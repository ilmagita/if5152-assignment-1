# helper functions

from PIL import Image
import numpy as np
from scipy.signal import windows

# READ IMAGE FUNCTIONS

def readImageAsArray(image_file_path: str):
    """
    # readImageAsArray
    # takes an image file path and returns it as an array
    """
    image = Image.open(image_file_path)
    img_array = np.asarray(image)
    
    return img_array

def showImageFromArray(img_array: np.ndarray):
    img_array = img_array.astype(np.uint8) 
    image = Image.fromarray(img_array)
    image.show()
    
# PADDING IMAGE FUNCTIONS 

def ceil(x):
    if int(x) == x:
        return int(x)
    else:
        return int(x) + 1

def padImage(image, pad_height, pad_width):
    """
    # padImage
    # receives an array of image and pads it to the input parameters
    # pads it using the nearest neighbors method
    """
    img_array = image
    padded_image = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge') # pad with nearest neighbors
    return padded_image

# FILTERS

def generateGaussianFilter(sigma, normalised=True):
    """
    * generates a n x n matrix with a centered gaussian of standard deviation std centered on it.
    if normalised, its elements equals 1.'''
    """
    n = 2 * ceil(3 * sigma) + 1
    
    gaussian1D = windows.gaussian(n, sigma)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    
    if normalised:
        gaussian2D /= (2*np.pi*(sigma ** 2))
    return gaussian2D

def generateSobelFilterX():
    return [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    
def generateSobelFilterY():
    return [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]