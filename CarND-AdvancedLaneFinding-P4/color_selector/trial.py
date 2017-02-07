import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, img_as_float
from skimage import exposure
import numpy as np

img = Image.open('test3.jpg')
img = np.asarray(img)
imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
imghsv[:,:,2] = [[max(pixel - 75, 0) if pixel < 190 else min(pixel + 75, 255) for pixel in row] for row in imghsv[:,:,2]]
img = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

cv2.imwrite('asd.jpg', img)

plt.imshow(img)
plt.show()


#
# arr = np.asarray(img)
# plt.imshow(arr, vmin=0, vmax=255)
# plt.show()
# #
# img = cv2.imread('test3.jpg')
#
# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#
# # equalize the histogram of the Y channel
# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#
# # convert the YUV image back to RGB format
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#
# cv2.imshow('Color input image', img)
# cv2.imshow('Histogram equalized', img_output)

# cv2.waitKey(0)
