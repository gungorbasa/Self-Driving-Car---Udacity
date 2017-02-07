import numpy as np
import cv2


def find_sobel(img, orient='x', sobel_kernel=3):
    if np.shape(img)[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = np.absolute(sobel)

    return sobel


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    sobel = find_sobel(img, orient=orient, sobel_kernel=sobel_kernel)
    scaled = np.uint8(255 * sobel / np.max(sobel))
    mask = np.zeros_like(scaled)
    mask[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return mask


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = find_sobel(img, orient='x', sobel_kernel=sobel_kernel)
    sobely = find_sobel(img, orient='y', sobel_kernel=sobel_kernel)
    magxy = np.sqrt((np.square(sobelx) + np.square(sobely)))
    scaled_magnitude = np.uint8(255 * magxy / np.max(magxy))

    mask = np.zeros_like(scaled_magnitude)
    mask[(scaled_magnitude >= mag_thresh[0]) & (
        scaled_magnitude <= mag_thresh[1])] = 1

    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = find_sobel(img, orient='x', sobel_kernel=sobel_kernel)
    sobely = find_sobel(img, orient='y', sobel_kernel=sobel_kernel)
    direction = np.arctan2(sobely, sobelx)
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return mask
