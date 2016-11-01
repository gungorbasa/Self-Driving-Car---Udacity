import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('test_images/solidYellowLeft.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
lower_white = np.array([0, 220, 0])
upper_white = np.array([180, 255, 255])
# H = 50 - 70
# S = 72 - 100
# L = 70 - 87

lower_yellow = np.array([0,123,140])
upper_yellow = np.array([180,250,255])

mask_white = cv2.inRange(hsv, lower_white, upper_white)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
final_mask = cv2.bitwise_or(mask_white, mask_yellow)
res = cv2.bitwise_and(img, img, mask=mask_yellow)


plt.imshow(res)
plt.show()


#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#yellow_range = (np.array([103, 86, 65]), np.array([145, 133, 128]))

# plt.figure(1)

# cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(hsv)
# plt.show()
# plt.figure(2)
#yellow = cv2.inRange(img, yellow_range[0], yellow_range[1])
# plt.imshow(yellow)
#plt.imshow(cv2.inRange(gray, 180, 255), cmap='gray')
# plt.show()
