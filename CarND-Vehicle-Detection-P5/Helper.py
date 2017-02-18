import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

class Helper(object):
    @classmethod
    def plot_images(self, fig, rows, cols, imgs, titles):
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i+1)
            plt.title(i+1)
            img_dims = len(img.shape)
            if img_dims < 3:
                # gray scale image
                plt.title(titles[i])
                plt.imshow(img, cmap='gray')

            else:
                plt.title(titles[i])
                plt.imshow(img)

        plt.figure(figsize=(20, 20))
        plt.show()

    @classmethod
    def draw_rectangle(self, img, locations, color=(255, 0, 0), thick=6):
        bboxes = locations
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    @classmethod
    def sliding_window(self, image, step_size=32, window_size=64):
        y, x = np.shape(image)[0], np.shape(image)[1]
        for x_start in range(0, x, step_size):
            for y_start in range(0, y, step_size):
                yield (x, y, image[y_start:y_start + window_size, x_start:x_start + window_size])

    # http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
    @classmethod
    def pyramid(self, image, scale=1.2, min_size=(512, 512)):
        yield image
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
                break
            yield image
