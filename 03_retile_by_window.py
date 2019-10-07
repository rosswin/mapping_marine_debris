#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
#from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import rasterio
from rasterio import windows
from rasterio.enums import Resampling
import numpy as np

#### EDIT THESE ########################################################

file_list = f"/media/ross/ssd/00_2015_DAR_marinedebris/maui/03_gdalwarp_blocksize/tif_list.txt"
out_dir = f"/media/ross/ssd/00_2015_DAR_marinedebris/maui/04_window_retile"

#### STOP EDITING #####################################################

log_name = os.path.join(out_dir, 'log.txt')

#### START OF FCNs ########################################

def window_retile(in_path):
    with rasterio.open(in_path, 'r') as src:
        src_transform = src.transform
        src_crs = src.crs

        #Play with the file names, end up with our output path
        basename = os.path.basename(in_path)
        basename_no_ext = os.path.splitext(basename)[0]


        logging.info(f"{basename}, OPEN")

        assert len(set(src.block_shapes)) == 1, "block shapes != 1"

        for ji, current_window in src.block_windows():
            filename = f"{basename_no_ext}_{ji[-1]}_{ji[0]}.jpg"
            out_path = os.path.join(out_dir, filename)
            #get the affine matrix for the window we are writing
            win_transform = src.window_transform(current_window)



            #get the actual data from the window as a 3D numpy array
            data = src.read(window=current_window)
            #check if our data is all zeros (no data), if it is don't write the file
            zeros = np.zeros_like(data)

            if np.array_equal(data, zeros):
                logging.info(f"{filename}, NO DATA")
            else:
                #this only runs if we have valid data. Write a custom JPEG profile, and export our window to a file.
                profile={'driver': 'JPEG',
                        'count': src.count,
                        'dtype': rasterio.ubyte,
                        'height': current_window.height,
                        'width': current_window.width,
                        'transform': win_transform,
                        'crs': src_crs}

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data)
                logging.info(f"{out_path}, WROTE")

#### END OF FCNs ##########################################
#### START OF MAIN ########################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(log_name),
        logging.StreamHandler()],
    handlers[1].setLevel(logging.ERROR))

# open the file list, count the files and display count
with open(file_list, 'r') as f:
    in_paths = [line.strip() for line in f]

#fire up the processes. Count them as they finish and display the remaining files.
pool=multiprocessing.Pool()
result=pool.map_async(window_retile, in_paths, chunksize=1)

num_of_files = len(in_paths)

while not result.ready():
    print(f"{result._number_left}, {num_of_files}, REMAINING.")
    time.sleep(5)

pool.close()
pool.join()


'''
for path in in_paths:
    window_retile(path)
'''
'''
with ProcessPoolExecutor() as executor:
    executor.map(window_retile, in_paths)
'''
