import numpy as np
import cv2

src = np.array([(215, 715), (1120, 715), (606, 431), (676, 431)])
dst = np.array([(200, 720), (1080, 720), (200, -500), (1080, -500)])
M = cv2.getPerspectiveTransform(src, dst)
