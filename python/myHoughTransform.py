# myHoughTransform
# Ilmagita Nariswari - 18221101

# ORIGINAL MATLAB FUNCTION:
# function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)

import numpy as np

def myHoughTransform(Im: np.ndarray, threshold: int, rhoRes: int, thetaRes: int):
    """
    # myHoughTransform
    * implementation of the Hough Transform
    * takes the array of image, threshold, rhoRes, and thetaRes
    """
    # edge_magnitude_img = readImageAsArray('results/myEdgeFilter_res.jpg')
    
    img_height = len(Im)
    img_width = len(Im[0])
    
    # rhoMax in Hough transform is diagonal length of image because it is the farthest point from origin, thus enabling
    # accumulator array to capture all lines possible
    rhoMax = np.sqrt(img_height**2 + img_width**2)
    
    # initialize scales of minimum and maximum rho and theta
    rhoScale = np.arange(0, rhoMax, rhoRes)
    thetaScale = np.arange(0, 2 * np.pi, thetaRes)
    
    # initialize accumulator (H)
    H = np.zeros((len(rhoScale), len(thetaScale)))
    
    # threshold the edge image
    # get the indices of points above threshold
    edge_points = np.argwhere(Im > threshold)
    # print(f"Number of edge points: {len(edge_points)}")
    
    # for each point, try theta values and check obtained value of rho
    for point in edge_points:
        y, x = point[:2]
        
        for j, theta in enumerate(thetaScale):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            
            if rho >= 0:
                i = int(rho/rhoRes)
                H[i, j] += 1
            
    return H, rhoScale, thetaScale