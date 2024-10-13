# myHoughLines
# Ilmagita Nariswari - 18221101

# ORIGINAL MATLAB FUNCTION:
# function [rhos, thetas] = myHoughLines(H, nLines)

import numpy as np

def nms_all(img, kernel_size=3):
    """
    # nms_all: non maximal suppression for all neighbours
    * receives an input of an array for image and kernel size
    * outputs the dilated image
    """
    dilated_img = np.zeros(img.shape)
    pad = kernel_size // 2
    
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # get the current 3x3 area
            area = padded_img[i:i+kernel_size, j:j+kernel_size]
            dilated_img[i, j] = np.max(area)
    
    return dilated_img

def myHoughLines(H: np.ndarray, nLines: int):
    """
    # myHoughLines
    * takes H: transform accumulator, and nLines: number of lines to return
    * outputs rhos: array[nLines][1], and thetas: array[nLines][1]
    """
    img = H.copy()
    dilated_img = nms_all(img)
    
    rhos = []
    thetas = []
    
    for i in range(nLines):
        flat_index = np.argmax(dilated_img)
        max_index = np.unravel_index(flat_index, dilated_img.shape)
        
        dilated_img[max_index] = -999
        
        rhos.append(max_index[0])
        thetas.append(max_index[1])
    
    return rhos, thetas

    