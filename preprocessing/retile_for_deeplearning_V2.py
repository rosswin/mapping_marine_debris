#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import math
import csv

import multiprocessing
import logging
from itertools import product, repeat
import optparse

import rasterio
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.mask import mask

import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
from shapely import geometry

#######################
#### FUNCTIONS ########
#######################

def grid_calc(width, height, stride):
    #get all the upper left (ul) pixel values of our chips. Combine those into a list of all the chip's ul pixels. 
    #NOTE: THESE ARE THE OPPOSITE OF WHAT I THINK THEY SHOULD BE (HEIGHT/WIDTH SWAPPED)
    #I DON"T KNOW WHY IT WORKS DONT TOUCH THIS FUNCTION UNLESS YOURE GOING TO OWN IT.
    x_chips = height // stride
    x_ul = [stride * s for s in range(x_chips)]

    y_chips = width // stride
    y_ul = [stride * s1 for s1 in range(y_chips)]

    #xy_ul = product(x_ul, y_ul) #this is the final list of ul pixel values
    print(f"x chips: {len(x_ul)}")
    print(f"y chips: {len(y_ul)}")

    xy_ul = []
    for x in x_ul:
        for y in y_ul:
            xy_ul.append((x, y))

    #print(f"x/y ul len: {len(x_ul)} / {len(y_ul)}")
    #print(f"y * x = {len(x_ul) * len(y_ul)}")
    #print(f"xy_ul len: {len(xy_ul)}")

    return xy_ul

def build_window(in_src, in_xy_ul, in_chip_size, in_chip_stride):

    out_window = windows.Window(col_off = in_xy_ul[0],
                                row_off = in_xy_ul[1],
                                width=in_chip_size,
                                height=in_chip_size)
    
    out_win_transform = windows.transform(out_window, in_src.transform)
    #print(out_win_transform)
    
    col_id = in_xy_ul[1] // in_chip_stride
    row_id = in_xy_ul[0] // in_chip_stride
    out_win_id = f'{col_id}_{row_id}'
    
    out_win_bounds = windows.bounds(out_window, out_win_transform)
    #print(out_win_bounds)
    
    return out_window, out_win_transform, out_win_bounds, out_win_id

def make_gdf(polygons, attr_dict, out_crs, out_gdf_path='none'):
    #print(f"out_gdf_path: {out_gdf_path}")
    gs = gpd.GeoSeries(polygons)
    
    df = pd.DataFrame(data=attr_dict)
    
    gdf = gpd.GeoDataFrame(df, geometry=gs)
    gdf.crs=out_crs
    
    #optionally write a file if the path was provided. NOTE: COULD EXPAND THIS TO HANDLE FORMATS OTHER THAN GPKG.
    if out_gdf_path != 'none':
        if os.path.exists(os.path.dirname(os.path.abspath(out_gdf_path))):
            print(f"Writing: {out_gdf_path}")
            
            gdf.to_file(out_gdf_path, driver='GPKG')
    
    #regardless of writing output, return GDF
    return gdf

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

def coords_2_pix(in_bounds, in_affine):
    xmin = in_bounds[0]
    ymin = in_bounds[1]
    xmax = in_bounds[2]
    ymax = in_bounds[3]
    
    xs = (xmin, xmax)
    ys = (ymin, ymax)
    
    pix_coords = rasterio.transform.rowcol(in_affine, xs, ys)
    
    pix_bounds = (pix_coords[0][1], pix_coords[1][1], pix_coords[0][0], pix_coords[1][0])
    
    return pix_bounds

def coords_2_pix_gdf(gdf):
    gdf['x_min'] = gdf.bounds.minx
    gdf['y_min'] = gdf.bounds.miny
    gdf['x_max'] = gdf.bounds.maxx
    gdf['y_max'] = gdf.bounds.maxy
    
    gdf['px_x_min'] = (gdf['x_min'] - gdf['a2']) // gdf['a0']
    gdf['px_x_max'] = (gdf['x_max'] - gdf['a2']) // gdf['a0']
    
    #NOTE: in coordinates we the origin as the bottom left corner. In pixel coordinates, we set the origin at the top left corner.
    #Anyways, we're flipping our pix min/max here so that our origin starts at the top right corner.
    gdf['px_y_max'] = (gdf['y_min'] - gdf['a5']) // gdf['a4']
    gdf['px_y_min'] = (gdf['y_max'] - gdf['a5']) // gdf['a4']
    
    return gdf

def pix_2_xy(in_bounds, in_affine):
    xmin = in_bounds[0]
    ymin = in_bounds[1]
    xmax = in_bounds[2]
    ymax = in_bounds[3]
    
    xs = (xmin, xmax)
    ys = (ymin, ymax)
    
    pix_coords = rasterio.transform.xy(in_affine, xs, ys)
    
    pix_bounds = (pix_coords[0][0], pix_coords[1][1], pix_coords[0][1], pix_coords[1][0] )
    #print(f"pix bounds: {pix_bounds}")
    return pix_bounds

def return_intersection(in_tindex, in_annotations, unique_annotation_id):
    inter = gpd.overlay(in_tindex, in_annotations)
    inter['intersect_area'] = inter['geometry'].area
    filter_partial_annotations = inter[inter['intersect_area'] == 1.0]
    remove_duplicates = filter_partial_annotations.drop_duplicates(subset=unique_annotation_id)
    
    return remove_duplicates

def create_cindex(in_file, in_size, in_stride, in_out_dir):
    basename = os.path.splitext(os.path.basename(in_file))[0]
    #print(basename)
    #print(f"{in_file}, {in_size}, {in_stride}, {in_out_dir}")
    gdfs = []

    with rasterio.open(in_file, 'r') as src:
        print(f"Initial Width/Height: {src.width}, {src.height}")
        #print(f"src height/width: {src.height}/{src.width}")
        #print(f"src bounds: {src.bounds}")
        #print(f"src transform: {src.transform}")
        
        upper_left_grid = grid_calc(src.width, src.height, in_stride)
        
        for ul in upper_left_grid:
            #note, we're currently working with slices because I can't make col_off, row_off work. Code needs to be reworked to naturally work with slices
            col_start = ul[0]
            col_stop = ul[0] + in_size
            row_start = ul[1]
            row_stop = ul[1] + in_size
            #slices = (col_start, row_start, col_stop, row_stop)
            colrow_bounds = (col_start, row_start, col_stop, row_stop)
                
            win, win_transform, win_bounds, win_id = build_window(src, ul, in_size, in_stride)

            #NOTE: I had to write my own affine lookup to get bounding boxs from windows (pix_2_coords). Rasterio's windows.bounds(win, win_transform) 
            #caused every overlpping tile to shift 256 pix in the x and y direction (removed overlap, doubled area covered by chip tindex)
            #therefore, the win_bounds variable above should not currently be used. I need to investigate further.
            
            #ret = pix_2_coords(slices, src.transform)

            ret = pix_2_xy(colrow_bounds, src.transform)
            #print(f"ret: {ret}")
            #create and store the chip's geometry (the bounding box of the image chip)
            envelope = geometry.box(*ret)
            geometries=[]
            geometries.append(envelope)
            
            #store the image basename. No real reason, just comes in handy alot.
            attr_basename=[]
            attr_basename.append(in_file)
            
            #store chip name as an attribute in the cindex attribute table
            chip_name = f"{basename}_{win_id}"
            attr_filename = []
            attr_filename.append(chip_name)
            
            # store affine values as attributes in the cindex attribute table
            px_width = []
            row_rot = []
            col_off = []
            col_rot = []
            px_height = []
            row_off = []
            
            px_width.append(win_transform[0])
            row_rot.append(win_transform[1])
            col_off.append(win_transform[2])
            col_rot.append(win_transform[3])
            px_height.append(win_transform[4])
            row_off.append(win_transform[5])
            
            #create a single chip feature with attributes and all
            attr_dict = {}
            attr_dict['basename'] = attr_basename
            attr_dict['filename'] = attr_filename
            attr_dict['a0'] = px_width
            attr_dict['a1'] = row_rot
            attr_dict['a2'] = col_off
            attr_dict['a3'] = col_rot
            attr_dict['a4'] = px_height
            attr_dict['a5'] = row_off
        
            chip_gdf = make_gdf(geometries, attr_dict, src.crs)

            #append our single chip feature to a list of all chips. Later we will merge all the single chips into a big chip index (cindex).
            gdfs.append(chip_gdf)
            
    #merge all those little chip features together into our master cindex for the input image 
    cindex_gdf_path = os.path.join(in_out_dir, f"{basename}_chip_tindex.gpkg")
    cindex_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    cindex_gdf.crs = src.crs
    cindex_gdf.to_file(cindex_gdf_path, driver='GPKG')

    return cindex_gdf

def write_annotations(in_gdf, out_path='none'):
    coord_gdf = coords_2_pix_gdf(in_gdf)
    
    #set our columns/column order
    out_gdf = coord_gdf[['filename', 'x_min', 'y_min', 'x_max', 'y_max', 
                         'px_x_min', 'px_y_min', 'px_x_max', 'px_y_max', 
                         'label_name', 'label', 'label_int']]
    
    if out_path != 'none':
        out_gdf.to_csv(out_path, index=False)
    
    return out_gdf

def mask_raster(in_poly, src_raster, in_out_path):
    try:
        with rasterio.open(src_raster, 'r') as src:
            out_data, out_transform = mask(src, [in_poly], crop=True)
    except:
        print("ERROR 1 in mask_raster:")
        print("Could not read cropped data/transform.")
        sys.exit(0)

    write_jpeg(out_data, src.count, out_data.shape[1], out_transform, src.crs, in_out_path)

def backbone(args, in_f):
    try:
        #print(f"backbone: {in_f}")
        logging.info(f"backbone: {in_f}")
        #unpack our args list.
        anno_path = args[0] 
        in_anno = gpd.read_file(anno_path)
        
        size = args[1]
        stride = args[2]
        out_dir = args[3]
        #print(f"{in_anno}, {size}, {stride}, {out_dir}")
    except:
        print("Error loading arguments into backbone. Check your args.")

    try:
        logging.info(f"cindex: {in_f}")
        #Chip out our image, return a cindex
        cindex = create_cindex(in_f, size, stride, out_dir)
    except:
        print("Error in cindex operation!")

    try:
        logging.info(f"intersect: {in_f}")
        #find all the annotations that intersect each chip. Filter chips with no annotations, filter annotations that are not fully contained within a chip.
        intersect = return_intersection(cindex, in_anno, 'unique_pt_id')
    except:
        print("Error in intersect operation!")

    try:
        logging.info(f"writing positive files: {in_f}")
        #generate a list of positive chips in the annotation database.
        pos_chips = intersect['filename'].unique().tolist()
        
        pos_chips_gdf = cindex[cindex['filename'].isin(pos_chips)]
        #print(pos_chips_gdf.head())

        for i, row in pos_chips_gdf[['geometry', 'basename','filename']].iterrows():
            polygon = row["geometry"]
            src_raster = row['basename']
            out_raster_path = os.path.join(out_dir, f"{row['filename']}.jpg")
            #print(f"{polygon}, {src_raster}, {out_raster_path}")
            #this also write our positive image chip to a jpeg located at out_raster_path
            mask_raster(polygon, src_raster, out_raster_path)
    except:
        print("Error when writing images!")
        print(f"polygon: {polygon}")
        print(f"src_raster: {src_raster}")
        print(f"out_raster_path: {out_raster_path}")

    logging.info(f"backbone COMPLETE: {in_f}")
    #return our annotations to be bound into a island-wide annotation data set.
    
    return intersect

#######################
#### END FUNCTIONS ####
#######################
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
        action='store', dest='usr_stride',
        type='int', default=256,
        help='Amount of image chip overlap in pixels (ie 256 is 50 percent overlap with a chipsize of 512)')

    parser.add_option('-a', '--annotations',
        action='store', dest='in_annotation',
        type='string',
        help='path to a geopackage containing marine debris annotation envelopes. Note: this probably wont work with all annotations. Use preapproved annos for now.')

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

    if os.path.exists(options.in_annotation):
        pass
    else:
        print('ERROR: Cannot find input annotations. Abort.')
        logging.error('ERROR: Cannot find input annotations. Abort.')
        sys.exit(0)

    #print("exists")
    '''
    with open(options.file_list, 'r') as f:
        in_paths = [line.strip() for line in f]
    
    for fi in in_paths:
        inter = backbone((options.in_annotation, 
                options.usr_size, 
                options.usr_stride, 
                options.usr_out_dir),
                fi)
    '''
    #open our file list and arguments. Zip them up into a list of arguments and a input file that feeds into multiprocessing's starmap.
    with open(options.file_list, 'r') as f:
        in_paths = [line.strip() for line in f]

    args = [[options.in_annotation, options.usr_size, options.usr_stride, options.usr_out_dir]] * len(in_paths)
    zipped = zip(args, in_paths)

    #print(f"args {args}")

    #start the pool. Each entry in results will contain a gdf of all the resulting chips.
    pool=multiprocessing.Pool(processes=8)
    map_results = pool.starmap_async(backbone, zipped)

    

    while not map_results.ready():
        print(f"retile_for_deeplearning_V2.py | {map_results._number_left} of {len(in_paths)} files remain.")
        time.sleep(5)

    pool.close()
    pool.join()

    results = map_results.get()

    print(f"Writing final annotations.")
    logging.info(f"Writing final annotations.")
    #merge all the pd.Dataframes, convert to gpd.GeoDataFrame
    results_df = pd.concat(results, ignore_index=True)
    results_gdf = gpd.GeoDataFrame(results_df, geometry='geometry')
    #print(results_df.head())
    
    #write annotations to csv
    out_path = os.path.join(options.usr_out_dir, 'final_annotations.csv')
    write_annotations(results_gdf, out_path)

    print("SUCCESS!")
    logging.info(f"SUCCESS!")

#out_df = results_df[['basename', 'filename', 
#'x_min', 'y_min', 'x_max', 'y_max', 
#'px_x_min', 'px_y_min', 'px_x_max', 'px_y_max', 
#'label_name', 'label', 'label_int']]
