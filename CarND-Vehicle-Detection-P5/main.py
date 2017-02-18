from skimage.feature import hog
import cv2
import numpy as np
from scipy.ndimage.measurements import label

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 1#2 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [500, None] # Min and max in y to search in slide_window()


def read_image(name):
#     print("Image Name: ", name)
    img = cv2.imread(name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = read_image("./bbox.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
img = img[400:500,400:500,:]

features, hog_image = hog(img[:,:, hog_channel], orientations=orient,
                                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                                  cells_per_block=(cell_per_block, cell_per_block),
                                                  transform_sqrt=True,
                                                  visualise=True, feature_vector=True)

print("Features: ", features.tolist())
print("Image: ", hog_image)