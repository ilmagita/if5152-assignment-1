# myEdgeFilter
# Ilmagita Nariswari - 18221101

# ORIGINAL MATLAB FUNCTION:
# function [img1] = myEdgeFilter(img0, sigma)

import numpy as np
from myImageFilter import myImageFilter
from helper import generateGaussianFilter, generateSobelFilterX, generateSobelFilterY

def myEdgeFilter(img0: np.ndarray, sigma: int) -> np.ndarray:
    """
    # myEdgeFilter
    * receives an input of a greyscale image (array) and scalar (int)
    * scalar is the standard deviation of the Gaussian smoothing kernel to be used
    * outputs img1 or the edge magnitude image as an array
    """
    
    img_array = img0
    
    # convolute with Gaussian kernel
    gaussian_kernel = generateGaussianFilter(sigma)
    smoothed_img = myImageFilter(img_array, gaussian_kernel)
    
    # calculate image gradient in the x-direction by convoluting smoothed img with x-oriented Sobel filter
    sobelX = generateSobelFilterX()
    imgX = myImageFilter(smoothed_img, sobelX)
    
    # calculate image gradient in the y-direction by convoluting smoothed img with y-oriented Sobel filter
    sobelY = generateSobelFilterY()
    imgY = myImageFilter(smoothed_img, sobelY)
    
    imgX = np.array(imgX)
    imgY = np.array(imgY)
    
    # get the edge magnitude
    edge_magnitude = np.sqrt(imgX ** 2 + imgY ** 2)
    magnitude = (edge_magnitude / np.max(edge_magnitude)) * 255
    img1 = magnitude
    
    # implement non-maximum suppression
    nms_magnitude = np.zeros_like(magnitude)
    count_error = 0
    for i in range(1, len(img_array)):
        for j in range(1, len(img_array[0])):
            
            # calculate angle by taking the inverse tangent of the ratio of the gradient magnitude in the y-direction to the x-direction one
            angle = np.arctan2(imgY[i, j], imgX[i, j]) * (180/np.pi) % 180
            
            try:
                n1 = 0
                n2 = 0
                
                # determine angles to closest and look at two neighboring pixels 
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180): # 0 degrees -> _ -> right and left
                    n1 = magnitude[i, j + 1]
                    n2 = magnitude[i, j - 1]
                elif (22.5 <= angle < 67.5): # 45 degrees -> \ -> top left and bottom right
                    n1 = magnitude[i - 1, j - 1]
                    n2 = magnitude[i + 1, j + 1]
                elif (67.5 <= angle < 112.5): # 90 degrees -> | -> top and bottom
                    n1 = magnitude[i - 1, j]
                    n2 = magnitude[i + 1, j]
                elif (112.5 <= angle < 157.5): # 135 degrees -> / -> top right and bottom left
                    n1 = magnitude[i - 1, j + 1]
                    n2 = magnitude[i + 1, j - 1]
                    
                if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                    nms_magnitude[i, j] = magnitude[i, j]
                else:
                    nms_magnitude[i, j] = 0
                
            except IndexError as e:
                # print(f"Error calculating angle at pixel ({i}, {j}): {e}")
                nms_magnitude[i, j] = 0
                count_error += 1
            
    img1 = nms_magnitude
    return img1