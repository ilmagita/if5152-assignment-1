import sys
import os
import cv2
import numpy as np

# houghScript.py adapted from: https://github.com/Ali-Hasan-Khan28/Computer-Vision

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..', 'python'))

from myEdgeFilter import myEdgeFilter
from myHoughLines import myHoughLines
from myHoughTransform import myHoughTransform
from drawLine import drawLine

# Parameters
datadir = os.path.join(current_dir, 'data')  # Update to point to 'ec\data'
resultsdir = os.path.join(current_dir, 'results')

sigma = 2
threshold = 0.03
rhoRes = 2
thetaRes = np.pi / 90
nLines = 50
# end of parameters

# START
print('Starting script ec.py')

# Get list of images
imglist = [f for f in os.listdir(datadir) if f.endswith('.jpg')]

for imgname in imglist:
    print(f'Starting function for {imgname}')
    # Read in images
    img_path = os.path.join(datadir, imgname)
    img = cv2.imread(img_path)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float64) / 255.0

    # Actual Hough line code function calls
    Im = myEdgeFilter(img, sigma)
    H, rhoScale, thetaScale = myHoughTransform(Im, threshold, rhoRes, thetaRes)
    rhos, thetas = myHoughLines(H, nLines)
    lines = cv2.HoughLinesP((Im > threshold).astype(np.uint8), 1, thetaRes, threshold=0, minLineLength=10, maxLineGap=5)

    # Save outputs to files
    fname = os.path.join(resultsdir, f"{os.path.splitext(imgname)[0]}_01edge.png")
    cv2.imwrite(fname, np.sqrt(Im / np.max(Im)) * 255)

    fname = os.path.join(resultsdir, f"{os.path.splitext(imgname)[0]}_02threshold.png")
    cv2.imwrite(fname, (Im > threshold).astype(np.uint8) * 255)

    fname = os.path.join(resultsdir, f"{os.path.splitext(imgname)[0]}_03hough.png")
    cv2.imwrite(fname, (H / np.max(H) * 255).astype(np.uint8))

    fname = os.path.join(resultsdir, f"{os.path.splitext(imgname)[0]}_04lines.png")
    img_lines = np.dstack([img,img,img])
    
    # display line results from myHoughLines function in red
    for k in np.arange(nLines):
        a = np.cos(thetaScale[thetas[k]])
        b = np.sin(thetaScale[thetas[k]])
        
        x0 = a*rhoScale[rhos[k]]
        y0 = b*rhoScale[rhos[k]]
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(img_lines,(x1,y1),(x2,y2), (0,0,255),1)
    
    # display line segment results from cv2.HoughLinesP in green
    for line in lines:
        coords = line[0]
        cv2.line(img_lines, (coords[0], coords[1]), (coords[2], coords[3]), \
                    (0, 255, 0), 1)

    cv2.imwrite(fname, 255 * img_lines)
    
    print(f'Done applying functions to {imgname}.')