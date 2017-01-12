import cv2
import numpy as np
import math
import random


class Preprocessing():
    @staticmethod
    def augment_brightness_camera_images(image):
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        # print(random_bright)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        return image1

    @staticmethod
    def trans_image(image, steer, trans_range):
        shape = np.shape(image)
        cols, rows = shape[1], shape[0]
        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        steer_ang = steer + tr_x / trans_range * 2 * .2
        tr_y = 40 * np.random.uniform() - 40 / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

        return image_tr, steer_ang


    @staticmethod
    def add_random_shadow(image):
        r = random.randint(0, 1)
        alpha = random.uniform(0.3, 0.7)
        size = np.shape(image)
        left = [[0,0],[0,size[0]]]
        right = [[size[1], 0], [size[1], size[0]]]
        overlay = image.copy()
        output = image.copy()

        if r == 0:
            vertices = np.array([[left[0], left[1], [math.ceil(size[1] * np.random.uniform(0.3, 0.85)), left[1][1]],
                                  [math.ceil(size[1] * np.random.uniform(0.3, 0.85)), left[0][0]]]], dtype=np.int32)
        else:
            vertices = np.array([[math.ceil(size[1] * np.random.uniform(0.3, 0.85)), left[0][0]], [math.ceil(size[1] * np.random.uniform(0.3, 0.85)), left[1][1]],
                                 right[1], right[0]], dtype=np.int32)

        cv2.fillConvexPoly(overlay, vertices, (0,0,0))

        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                        0, output)
        return output

    @staticmethod
    def flip_image(img, steering_angle):
        return cv2.flip(img, 1), -1*steering_angle