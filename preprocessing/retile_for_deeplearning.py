#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math

import multiprocessing
import logging
from itertools import product, repeat
import optparse

import rasterio
from rasterio import windows
from rasterio.enums import Resampling

import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
from shapely import geometry

'''
This script will take a set of geotiffs and chip them into smaller pieces. 

THIS IS A WORK IN PROGRESS. It is finicky. It only take geotiffs. Those geotiffs must be of a height and width divisible by 512.
You will get jpegs in return. Only jpegs.
Those jpegs will only be of size 512x512, with 256 overlap (50% overlap)
The script will filter images of all 0,0,0 values.

Inputs: 
    file_list: a list of geotiff images to chip out.
    chip_size: number of pixels for the height/width of the image
    overlap: number of pixels of overlap
    out_dir: the directory to save all output files (chips, tile indexes, log files, etc.)

TODOS:
    1. Add support for multiple sizes. Currently only supports 512x512 chip size and 256x256 overlap (50% overlap). 
    2. Auto resize images based on user's desired chip size.
    3. Add ability to turn off filter no data.
    4. Add abiity to walk between formats other than input geotiffs and output jpegs.
    5. Allow annotations to be uploaded. Then we could filter the data set so only positives tiles are allowed in.
    6. Fix the logging.
'''

def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    #(lon, lat) of top left corner
    tl = geometry.Point(bbox[0], bbox[1])
    #(lon, lat) of bottom left corner
    bl = geometry.Point(bbox[2],bbox[1])
    #(lon, lat) of top right corner
    tr = geometry.Point(bbox[2],bbox[3])
    #(lon, lat) of bottom right corner
    br = geometry.Point(bbox[0],bbox[3])
    vertex_list = [tl, bl, tr, br]
    
    #logging.info(f'vertex list: {vertex_list}')
    polygon = geometry.Polygon([[v.x, v.y] for v in vertex_list])

    #print(f"polygon: {polygon}")
    
    return polygon

def make_gdf(polygons, attr_dict, out_crs):
    gs = gpd.GeoSeries(polygons)
    df = pd.DataFrame(data=attr_dict)

    gdf = gpd.GeoDataFrame(df, crs=out_crs, geometry=gs)
    
    return gdf

def grid_calc(width, height, stride):
    #get all the upper left (ul) pixel values of our chips. Combine those into a list of all the chip's ul pixels. 
    x_chips = width // stride
    x_ul = [stride * s for s in range(x_chips)]

    y_chips = height // stride
    y_ul = [stride * s for s in range(y_chips)]

    #xy_ul = product(x_ul, y_ul) #this is the final list of ul pixel values
    #print(f"xy_ul: {len(xy_ul)}")

    xy_ul = []
    for x in x_ul:
        for y in y_ul:
            xy_ul.append((x, y))

    #print(f"x/y ul len: {len(x_ul)} / {len(y_ul)}")
    #print(f"y * x = {len(x_ul) * len(y_ul)}")
    #print(f"xy_ul len: {len(xy_ul)}")

    return xy_ul

def is_nodata(in_data):
    #check to make sure our chip isn't made up of all zeros (nodata)
    zeros = np.zeros_like(in_data)
    #check if they're equal, return result
    if np.array_equal(in_data, zeros):
        return True
    else:
        return False

def write_jpeg(in_data, in_count, in_size, in_win_transform, in_src_crs, in_out_path):
    #building a custom jpeg profile for our chip due to some gdal/rasterio bugs in walking from input geotiff to output jpeg
    profile={'driver': 'JPEG',
        'count': in_count,
        'dtype': rasterio.ubyte,
        'height': in_size,
        'width': in_size,
        'transform': in_win_transform,
        'crs': in_src_crs}
        
    #write the chip
    with rasterio.open(in_out_path, 'w', **profile) as dst:
        dst.write(in_data)

def chip(args, in_tif):
    """
    Takes an input image, chips the image into a bunch of tiny, overlapping tiles. Returns a geopandas dataframe of all the chips
    with filenames in the attribute table.
    
    Inputs:
        args: a list of the following things:
            1. Chip Size- the desired size of the output chips in pixels (ie 512 results in 512x512 pixel images)
            2. Overlap- the desired overlap of the output chips in pixels (ie 256 results in 50% overlap on a 512x512 pixel image)
            3. Out Directory- where to write all the chips, tindex, etc.

    Outputs:
        Writes chipped images to Out Directory
            
    Returns:
        A geopandas dataframe chip tile index with the filename in the attributes table.
        
    """

    #unpack our args list. 
    size = args[0]
    overlap = args[1]
    out_dir = args[2]

    #calculate our stride
    stride = size - overlap

    #print(f"size: {size}")
    #print(f"overlap: {overlap}")
    #print(f"stride: {stride}")

    polygons = [] #this is to store geometries for exporting to a gpkg tindex
    attr_dict={} #this holds the attributes for exporting to a gpkg tindex
    basenames = [] #this holds the values of the 'filename' attribute

    #open our raster
    with rasterio.open(in_tif, 'r') as src:
        logging.info(f"Opened, {in_tif}")
        #print(f"Opened, {in_tif}")

        basename = os.path.splitext(os.path.basename(in_tif))[0]

        #get the image's coordinate reference system, width, and height
        src_crs = src.crs
        width = src.width
        height = src.height
        #print(f"h/w: {height}, {width}")

        #check that the size of the image is divisible by our chip size.
        if width%size != 0 or height%size != 0:
            logging.error(f"{in_tif} of height {height} and width {width}. Is not divisible by chosen chip size of {size}")
            print(f"{in_tif} of height {height} and width {width}. Is not divisible by chosen chip size of {size}")
        else:
            #come up with a grid of our the upper left chip corners.
            xy_ul = grid_calc(width, height, stride)

            #loop though our upper left corners, create a window view of that data
            for ul_pix in xy_ul:
                Window = windows.Window(
                    col_off=ul_pix[0],
                    row_off=ul_pix[1],
                    width=size, 
                    height=size)

                #get the affine matrix for our new shifted 512x512px chip
                win_transform = src.window_transform(Window)
                
                #read in our window's data
                data = src.read(window=Window)

                #pretty formating of our output chip filenames with column and row counts
                pretty_row_count = ul_pix[0] // size
                pretty_col_count = ul_pix[1] // size
                pretty_basename = f"{basename}_{pretty_row_count}_{pretty_col_count}"
                out_path = os.path.join(out_dir, pretty_basename + ".jpg")

                #check if a tile is composed completely of no data. if it is, skip it.
                if is_nodata(data) == True:
                    logging.info(f"{out_path} is no data (all zeros). Filtering out of data set.")
                else:
                    #write the valid data jpeg
                    write_jpeg(data, src.count, size, win_transform, src_crs, out_path)

                    #get the chip's bounding box geometry, convert it to a Shapely geometery, and append it to the polygon list. 
                    bounds = rasterio.windows.bounds(Window, src.transform)
                    polygon = generate_polygon(bounds)
                    polygons.append(polygon)

                    #print(f"{pretty_basename} bounds: {bounds}")
                    #print(f"bounds: {bounds}")

                    #print(f"polygons; {polygons}")
                    #get the name of the chip for the tindex's attribute table. Add all attributes to the attr_dict.
                    basenames.append(pretty_basename)

    #for name in basenames:
        #print(f"{name}")
    #print(f"len of basename: {len(basenames)}")
    #assemble our polygons and attr_dict into a geodataframe that represents our chip tindex
    attr_dict['filename'] = basenames
    gdf = make_gdf(polygons, attr_dict, src_crs)

    del basenames, attr_dict, gdf

    #write GeoDataFrame
    gdf.crs = src_crs
    #print(f"gdf crs: {gdf.crs}")
    gdf_path = os.path.join(out_dir, basename + '.gpkg')

    gdf.to_file(gdf_path, driver='GPKG')
    #print('wrote gdf')

    logging.info(f"CLOSED, {in_tif}")
    logging.info("/n")

    #return (gdf)

if __name__ == "__main__":
    #handle our input arguments
    parser = optparse.OptionParser()

    parser.add_option('-f', '--filelist',
        action="store", dest="file_list",
        type='string', help="A txt list of absolute file paths, one file per line. Must be tifs.")

    parser.add_option('-o', '--outdir',
        action='store', dest='usr_out_dir',
        type='string', help='An out directory to stash files. Images, tile indexes, logfiles, etc.')

    parser.add_option('-t', '--chipsize',
        action='store', dest='usr_size',
        type='int', default=512,
        help='Size of image chips in pixel (ie. 512 is 512x512px chips.)')

    parser.add_option('-s', '--stride',
        action='store', dest='usr_overlap',
        type='int', default=256,
        help='Amount of image chip overlap in pixels (ie 256 is 50 percent overlap with a chipsize of 512)')

    options, args = parser.parse_args()

    #setup logging
    log_name = r'log.txt'
    log_name = os.path.join(options.usr_out_dir, log_name)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", 
                        datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_name)])

    #do some checks to make sure we can find inputs and outputs.
    if os.path.exists(options.usr_out_dir):
        pass
    else:
        print('ERROR: Cannot find out directory. Abort.')
        logging.error('ERROR: Cannot find out directory. Abort.')
        sys.exit(0)

    if os.path.exists(options.file_list):
        pass
    else:
        print('ERROR: Cannot find input filelist. Abort.')
        logging.error('ERROR: Cannot find input filelist. Abort.')
        sys.exit(0)

    #open our file list and arguments. Zip them up into a list of arguments and a input file that feeds into multiprocessing's starmap.
    with open(options.file_list, 'r') as f:
        in_paths = [line.strip() for line in f]

    args = [[options.usr_size, options.usr_overlap, options.usr_out_dir]] * len(in_paths)
    zipped = zip(args, in_paths)

    #start the pool. Each entry in results will contain a gdf of all the resulting chips.
    pool=multiprocessing.Pool(processes=8)
    results = pool.starmap_async(chip, zipped)

    while not results.ready():
        print(f"retile_for_deeplearning.py | {results._number_left} of {len(in_paths)} files remain.")
        time.sleep(5)

    pool.close()
    pool.join()

'''
    print(f"Writing final geodataframe.")
    logging.info(f"Writing final geodataframe.")
    #merge all the pd.Dataframes, convert to gpd.GeoDataFrame
    results_df = pd.concat(results, ignore_index=True)
    results_gdf =gpd.GeoDataFrame(results_df, crs=out_crs, geometry='geometry')

    #write GeoDataFrame
    out_path = os.path.join(usr_out_dir, out_gpkg)
    results_gdf.to_file(out_path, driver='GPKG')

    print("Success!")
    logging.info(f"Writing final geodataframe.")'''