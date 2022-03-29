"""Script to convert .tif files to uint8 .png files. 
"""
import os
import cv2 as cv

START_PATH = './data/raw'
c = 1
for path,dirs,files in os.walk(START_PATH):
    for filename in files:
        if '.tif' in filename:
            print(f'Converting Image {c}/{len(files)}')
            i = cv.imread(f'./data/raw/{filename}', cv.IMREAD_COLOR)
            # i_8 = i.astype('uint8')
            i_8 = i * 4
            filename_out = filename.replace('tif', 'png')
            cv.imwrite(f'./data/{filename_out}', i_8)
            c += 1
