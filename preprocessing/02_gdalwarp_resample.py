#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rasterio
from rasterio import windows
from rasterio.enums import Resampling

from concurrent.futures import ProcessPoolExecutor

def gdalwarp_resample(in_path):
    with rasterio.open(in_path, 'r') as src:
        basename = os.path.basename(in_path)
        print(basename)

        src_height = src.height
        src_width = src.width

        resample_height = min(multiples_of_size, key=lambda x:abs(x-src_height))
        resample_width = min(multiples_of_size, key=lambda x:abs(x-src_width))
        print(f"orig height: {src_height}, resample height: {resample_height}")
        print(f"orig width: {src_width}, resample width: {resample_width}")

        out_path = os.path.join(out_dir, basename)
        #cutline_path = os.path.join(cutline_dir, os.path.splitext(basename)[0]+'.shp')

        gdalwarp = f"gdalwarp -srcnodata 0 -dstnodata 0 -co \"TILED=YES\" -co \"BLOCKXSIZE={size}\" -co \"BLOCKYSIZE={size}\" -ts {resample_width} {resample_height} {in_path} {out_path} 2>&1"

        #gdalwarp_with_cutline = f"gdalwarp -srcnodata 0 -dstnodata 0 -crop_to_cutline -cutline {cutline_path} -co \"TILED=YES\" -co \"BLOCKXSIZE={size}\" -co \"BLOCKYSIZE={size}\" -overwrite -ts {resample_width} {resample_height} {in_path} {out_path} 2>&1"

        os.system(gdalwarp)
        print(f"warped: {basename}")


size = 512
desired_height_width = (size, size)
multiples_of_size = [size * i for i in range(1,100)]

file_list = f"/media/ross/ssd/06_maui/01_gto/tif_list.txt"
out_dir = f"/media/ross/ssd/06_maui/02_gis_images"
#cutline_dir = f"/media/ross/ssd/02_molokai_dar2015/tiles_shp/03_gdal_trace_outline_clean"

with open(file_list, 'r') as f:
    in_paths = [line.strip() for line in f]

num_of_jpgs = len(in_paths)
print(f"{num_of_jpgs} total images to process")

'''
for path in in_paths:
    gdalwarp_resample(path)

'''
with ProcessPoolExecutor(max_workers=10) as executor:
    executor.map(gdalwarp_resample, in_paths)
