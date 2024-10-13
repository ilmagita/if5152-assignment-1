# myImageFilter
# Ilmagita Nariswari - 18221101

# ORIGINAL MATLAB FUNCTION:
# function [img1] = myImageFilter(img0, h)

import numpy as np
from helper import padImage

def myImageFilter(img0: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    # myImageFilter
    * takes an array of a greyscaleimage img0 and convolution filter stored in matrix h
    * outputs img1, which results from convolving im0 and h
    * assumes that matrix h is always odd-sized 
    """
    img_array = img0
    
    img_height = len(img0)
    img_width = len(img0[0])
    
    filter_height = len(h)
    filter_width = len(h[0])
    
    pad_height = filter_height // 2
    pad_width = filter_width // 2
    
    padded_image = padImage(img_array, pad_height, pad_width)
    
    # initialize output image
    img1 = [[0 for j in range(img_width)] for i in range (img_height)]
    
    # perform convolution - base it by origin point in image
    for i in range(img_height):
        for j in range(img_width):
            conv_sum = 0
            
            for ki in range(filter_height):
                for kj in range(filter_width):
                    # map the kernel position to the padded image
                    ni = i + ki
                    nj = j + kj
                    conv_sum += padded_image[ni][nj] * h[ki][kj]
            
            img1[i][j] = conv_sum
            
    return img1