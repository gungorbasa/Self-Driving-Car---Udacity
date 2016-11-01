#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# %matplotlib inline
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image

import math



def pipeline(img):
    gray = grayscale(img)
    y, x = gray.shape
    mask = cv2.inRange(gray, 180, 255)
    gauss = gaussian_blur(mask, kernel_size=5)


    edges = canny(mask, 60, 180)
    lines = hough_lines(edges, rho=1, theta=np.pi/180.0, threshold=20, min_line_len=20, max_line_gap=40)
    vertices = [np.array([ [(x//2) - 60,(y//2)+65], [(x//2) + 60,(y//2)+65],[x-60,y], [(x//2) - 340,y] ], np.int32)]
    roi = region_of_interest(lines, vertices)
    combined = weighted_img(roi, img, α=0.8, β=1., λ=0.)

    return combined

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

import operator

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    pos = []
    neg = []
    pos_lines = []
    neg_lines = []
    pos_slope, neg_slope = 0, 0
    pos_count, neg_count = 0, 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(1.0*(x2-x1))

            if x2 - x1 != 0:
                if slope > 0:
                    pos.append((x1, y1))
                    pos.append((x2, y2))
                    pos_lines.append(line)
                    pos_slope += slope
                    pos_count += 1
                elif slope < 0:
                    neg.append((x1, y1))
                    neg.append((x2, y2))
                    neg_lines.append(line)
                    neg_slope += slope
                    neg_count += 1
    min_pos_y = min(pos, key=operator.itemgetter(1))


    cv2.line(img, (300, 300), (10,10), color, thickness)

#     print(mean_pos_slope, mean_neg_slope)
#     for line in neg_lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

import os
images = os.listdir("test_images/")

i = 0
for fn in images:
    plt.figure(i)
    img = cv2.imread('test_images/' + fn)
    plt.imshow(pipeline(img))
    i += 1
plt.show()
