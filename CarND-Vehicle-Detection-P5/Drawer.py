import numpy as np
import cv2
import abc

class Drawer(object):
    def __init__(self):
        print("Drawer")

    @abc.abstractmethod
    def draw(self, img, locations, color=(0,0,255), thich=6):
        pass





class Rectangle(Drawer):
    # Define a function to draw bounding boxes
    def draw(self, img, locations, color=(255, 0, 0), thick=6):
        bboxes = locations
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

class Cube(Drawer):
    def draw(self, img, locations, color=(0, 0, 255), thick=6):
        pass
