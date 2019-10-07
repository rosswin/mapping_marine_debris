#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rasterio
from rasterio import windows
from rasterio.enums import Resampling

from concurrent.futures import ProcessPoolExecutor

desired_height_width = (512, 512)
multiples_of_512 = [512 * i for i in range(1,100)]

file_list = f"/media/ross/ssd/00_2015_DAR_marinedebris/molokai/01_gdaltranslate/tif_list.txt"
out_dir = f"/media/ross/ssd/00_2015_DAR_marinedebris/molokai/02_gdalwarp"
cutline_dir = f"/media/ross/ssd/00_2015_DAR_marinedebris/molokai/tiles_shp/03_gdal_trace_outline_clean"

def gdalwarp_resample(in_path):
    with rasterio.open(in_path, 'r') as src:
        src_height = src.height
        src_width = src.width

        resample_height = min(multiples_of_512, key=lambda x:abs(x-src_height))
        resample_width = min(multiples_of_512, key=lambda x:abs(x-src_width))

        basename = os.path.basename(in_path)
        out_path = os.path.join(out_dir, basename)
        cutline_path = os.path.join(cutline_dir, os.path.splitext(basename)[0]+'.shp')

        gdalwarp = f"gdalwarp -srcnodata 0 -dstnodata 0 -co \"TILED=YES\" -co \"BLOCKXSIZE=512\" -co \"BLOCKYSIZE=512\" -ts {resample_width} {resample_height} {in_path} {out_path}"

        gdalwarp_with_cutline = f"gdalwarp -srcnodata 0 -dstnodata 0 -crop_to_cutline -cutline {cutline_path} -co \"TILED=YES\" -co \"BLOCKXSIZE=512\" -co \"BLOCKYSIZE=512\" -ts {resample_width} {resample_height} {in_path} {out_path}"

        os.system(gdalwarp_with_cutline)
        #os.system(gdalwarp)
        print(f"warped: {basename}")

with open(file_list, 'r') as f:
    in_paths = [line.strip() for line in f]

num_of_jpgs = len(in_paths)
print(f"{num_of_jpgs} total images to process")

'''
for path in in_paths:
    gdalwarp_resample(path)

'''
with ProcessPoolExecutor() as executor:
    executor.map(gdalwarp_resample, in_paths)
