import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import matplotlib

class Helper():
    @classmethod
    def plot_images(cls, count, images, cls_true, cls_pred=None, randomly=True):
        size = np.shape(images)[0]
        img_shape = (32, 32, 3)
        # Create figure with 3x3 sub-plots.
        if randomly:
            fig, axes = plt.subplots(count, count)
            fig.subplots_adjust(hspace=0.8, wspace=0.8)
        else:
            fig, axes = plt.subplots(count, 1)
            fig.subplots_adjust(hspace=0.8, wspace=0.8)

        for i, ax in enumerate(axes.flat):
            if randomly:
                r = random.randint(0, size - 1)
            else:
                r = i
            # Plot image.
            ax.imshow(images[r])

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[r])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[r], cls_pred[r])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()


    @classmethod
    def rgb_to_yuv(cls, images):
        size = np.shape(images)[0]
        for i in range(size):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2YUV)


    @classmethod
    def rgb_to_hls(cls, images):
        size = np.shape(images)[0]
        for i in range(size):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2HLS)


    @classmethod
    def transform_image(cls, img, ang_range, shear_range, trans_range):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over.

        A Random uniform distribution is used to generate different parameters for transformation

        '''
        # Rotation

        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols, ch = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        shear_M = cv2.getAffineTransform(pts1, pts2)

        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        img = cv2.warpAffine(img, Trans_M, (cols, rows))
        img = cv2.warpAffine(img, shear_M, (cols, rows))

        return img


    @classmethod
    def resample_class_helper(cls, data, class_size, max_size):
        size = max_size - class_size
        shape = np.shape(data)
        shape = (shape[0] + size, shape[1], shape[2], shape[3])
        #     images = np.zeros(shape=shape, dtype=np.uint)
        images = data.copy()

        #     plt.imshow(images[0])
        #     plt.show()
        #     plt.figure()


        for i in range(size):
            r = random.randint(0, class_size - 1)
            transformed_img = Helper.transform_image(data[r], 20, 10, 5)
            images = np.concatenate((images, transformed_img.reshape((1, 32, 32, 3))), axis=0)

        return images


    @classmethod
    def resample_data(cls, data, unique, counts):
        print("Resampling Data..")
        max_class = np.max(counts)
        images = None
        ys = []
        start = 0
        # Goes over all classes
        for i in range(len(counts)):
            imgs = Helper.resample_class_helper(data[start:start + int(counts[i])], int(counts[i]), max_class)
            start = start + int(counts[i])
            #         plt.imshow(imgs[0])
            #         plt.show()
            #         plt.figure()
            if images is None:
                images = imgs.copy()
            else:
                images = np.concatenate((images, imgs), axis=0)

        for i in range(len(unique)):
            for j in range(max_class):
                ys.append(i)

        print("Resampling is done..")
        ys = np.array(ys)

        return images, ys.T



    @classmethod
    def plot(cls, title, x, y):
        plt.title(title)
        plt.bar(x, y)
        plt.show()

    @classmethod
    def one_hot_encoder(cls, y, n_classes):
        return np.eye(n_classes)[y]


    @classmethod
    def randomize_data(cls, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels