#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math

import multiprocessing
import logging
from itertools import product, repeat

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
    
    return polygon

def make_gdf(polygons, attr_dict):
    gs = gpd.GeoSeries(polygons)
    df = pd.DataFrame(data=attr_dict)

    gdf = gpd.GeoDataFrame(df, geometry=gs)
    
    return gdf

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

    #open our raster
    with rasterio.open(in_tif, 'r') as src:
        logging.info(f"Opened, {in_tif}")
        print(f"Opened, {in_tif}")
        basename = os.path.splitext(os.path.basename(in_tif))[0]
        polygons = [] #this is to store geometries for exporting to a gpkg tindex
        attr_dict={} #this holds the attributes for exporting to a gpkg tindex
        basenames = [] #this holds the values of the 'filename' attribute

        #get the image's coordinate reference system, width, and height
        src_crs = src.crs
        width = src.width
        height = src.height

        #check that the size of the image is divisible by our chip size.
        if width%size != 0 or height%size != 0:
            logging.error(f"{in_tif} of height {height} and width {width}. Is not divisible by chosen chip size of {size}")
            print(f"{in_tif} of height {height} and width {width}. Is not divisible by chosen chip size of {size}")
        else:
            #get all the upper left (ul) pixel values of our chips. Combine those into a list of all the chip's ul pixels. 
            x_chips = width // stride
            x_ul = [stride * s for s in range(x_chips)]
            #logging.info(f"x_ul: {x_ul}")

            y_chips = height // stride
            y_ul = [stride * s for s in range(y_chips)]

            xy_ul = product(x_ul, y_ul) #this is the final list of ul pixel values
        
        for ul_pix in xy_ul:
            Window = windows.Window(
                col_off=ul_pix[0],
                row_off=ul_pix[1],
                width=size, 
                height=size)
            #logging.info(Window)

            #get the affine matrix for our new shifted 512x512px chip
            win_transform = src.window_transform(Window)
            
            #read in our window
            data = src.read(window=Window)

            #check to make sure our chip isn't made up of all zeros (nodata)
            zeros = np.zeros_like(data)
            if np.array_equal(data, zeros):
                logging.info(f"{in_tif} is no data (all zeros). Filtering out of data set.")
            else:
                #building a custom jpeg profile for our chip due to some gdal/rasterio bugs in walking from input geotiff to output jpeg
                profile={'driver': 'JPEG',
                        'count': src.count,
                        'dtype': rasterio.ubyte,
                        'height': size,
                        'width': size,
                        'transform': win_transform,
                        'crs': src_crs}

                #pretty formating of our output chip filenames with column and row counts
                out_path = os.path.join(out_dir, f"{basename}_{ul_pix[0]}_{ul_pix[1]}.jpg")
                logging.info(out_path)
                
                #write the chip
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data)
                    logging.info(f"{out_path}, WROTE")
                    
                #get the chip's bounding box geometry, convert it to a Shapely geometery, and append it to the polygon list. 
                bounds = rasterio.windows.bounds(Window, src.transform)
                polygon = generate_polygon(bounds)
                polygons.append(polygon)

                #get the name of the chip for the tindex's attribute table. Add all attributes to the attr_dict.
                basenames.append(basename)
            
    attr_dict['filename'] = basenames
    gdf = make_gdf(polygons, attr_dict)
            
    return (gdf)

#### EDIT THESE #######################
file_list = r"/media/ross/ssd/00_kahoolawe_dar2015/00_256x256/tif_list.txt"
usr_out_dir = r"/media/ross/ssd/00_kahoolawe_dar2015/01_retile_for_deeplearning"
out_gpkg = r"kahoolawe_chip_tindex.gpkg"
log_name = r'log.txt'

#### STOP EDITING ####################
out_crs = {'init':'epsg:26904'}
usr_size=512
usr_overlap=256 #the results of this is 50% overlap (an intial 256x256 image, with 128 of padding on all sides.)

log_name = os.path.join(usr_out_dir, log_name)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(log_name)])
        #logging.StreamHandler()
    #])

#open our file list and arguments. Zip them up into a list of arguments and a input file that feeds into multiprocessing's starmap.
with open(file_list, 'r') as f:
    in_paths = [line.strip() for line in f]

args = [[usr_size, usr_overlap, usr_out_dir]] * len(in_paths)
zipped = zip(args, in_paths)

#start the pool. Each entry in results will contain a gdf of all the resulting chips.
pool=multiprocessing.Pool(processes=10)
results = pool.starmap(chip, zipped)

pool.close()
pool.join()

print(f"Writing final geodataframe.")
logging.info(f"Writing final geodataframe.")
#merge all the pd.Dataframes, convert to gpd.GeoDataFrame
results_df = pd.concat(results, ignore_index=True)
results_gdf =gpd.GeoDataFrame(results_df, crs=out_crs, geometry='geometry')

#write GeoDataFrame
out_path = os.path.join(usr_out_dir, out_gpkg)
results_gdf.to_file(out_path, driver='GPKG')

print("Success!")
logging.info(f"Writing final geodataframe.")