# drawLine.py
# ORIGINAL FUNCTION IN MATLAB:

'''
function [ img ] = drawLine( img, start, stop )

delta = 0.05 / norm(stop-start);

t = 0:delta:1;

x = start(1) + t * ( stop(1) - start(1) );
y = start(2) + t * ( stop(2) - start(2) );

if(ndims(img) == 2)
    img = repmat(img,[1 1 3]);
end

if (ischar(img))
    img = double(img) / 255;
end

x = round(x);
y = round(y);

idx = y + (x-1) * size(img,1);
img(idx) = 0;
img(idx +   size(img,1) * size(img,2)) = 1;
img(idx + 2*size(img,1) * size(img,2)) = 0;

end
'''

import numpy as np

def drawLine(img, start, stop):
    delta = 0.05 / np.linalg.norm(np.array(stop) - np.array(start))
    t = np.arange(0, 1, delta)

    x = start[0] + t * (stop[0] - start[0])
    y = start[1] + t * (stop[1] - start[1])

    if img.ndim == 2:  
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2) 

    if isinstance(img, str):  
        img = np.double(img) / 255.0  

    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    
    # Print for debugging
    print("X coordinates:", x)
    print("Y coordinates:", y)
    print("Image shape:", img.shape)

    x = np.clip(x, 0, img.shape[1] - 1) 
    y = np.clip(y, 0, img.shape[0] - 1) 

    idx = (y * img.shape[1]) + x  # y for height, x for width
    idx = np.clip(idx, 0, img.size - 1)

    img[idx, 0] = 0  
    img[idx, 1] = 1   
    img[idx, 2] = 0    

    return img