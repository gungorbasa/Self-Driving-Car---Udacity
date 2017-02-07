from FindLanes import Lane
import numpy as np
import copy

class SmoothLane(object):
    def __init__(self, margin=100, minpix=50):
        self.old_lanes = []
        self.smooth_size = 20
        self.lane = Lane(margin=margin, minpix=minpix)
        self.left_fit = None
        self.right_fit = None
        self.fity = None
        self.current_image = None

    def search(self, binary_warped):
        self.current_image = binary_warped
        self.lane.search(binary_warped)
        
        if len(self.old_lanes) >= self.smooth_size:
            self.old_lanes.pop(0)

        self.old_lanes.append(copy.copy(self.lane))
        self.smooth_fit(binary_warped)


    def smooth_fit(self, binary_warped):
        self.current_image = binary_warped
        leftx, lefty = [], []
        rightx , righty = [], []

        for i, lane in enumerate(self.old_lanes):
            # print(lane.leftx[111])
            leftx.extend(lane.leftx)
            lefty.extend(lane.lefty)
            rightx.extend(lane.rightx)
            righty.extend(lane.righty)
            # leftx = np.concatenate(lane.leftx)
            # rightx = np.concatenate(lane.rightx)
            # lefty = np.concatenate(lane.lefty)
            # righty = np.concatenate(lane.righty)
        leftx = np.array(leftx)
        rightx = np.array(rightx)
        lefty = np.array(lefty)
        righty = np.array(righty)
        self.leftx, self.lefty = leftx, lefty
        self.rightx, self.righty = rightx, righty
        self.fity = np.linspace(0, self.current_image.shape[0] - 1, self.current_image.shape[0])
        # print(np.shape(lefty), np.shape(leftx))
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        self.fit_leftx = np.array(self.left_fit[0] * self.fity**2 + self.left_fit[1] * self.fity + self.left_fit[2])
        self.fit_rightx = np.array(self.right_fit[0] * self.fity**2 + self.right_fit[1] * self.fity + self.right_fit[2])



        self.lane._set_variables(leftx, rightx, lefty, righty, self.left_fit, self.right_fit, self.fit_leftx, self.fit_rightx)

    def calculate_curvature(self):
        return calculate_curvature(real=True)


    def translate_to_real_world_image(self, image, Minv):
        return self.lane.translate_to_real_world_image(image, Minv)

    def plot(self):
        self.lane.plot()

    def copy(self):
        return SmoothLane(**self.kw)







