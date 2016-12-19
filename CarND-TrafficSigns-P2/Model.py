import abc
import numpy as np
import cv2
import tensorflow as tf
import tqdm
from Helper import Helper
from sklearn.cross_validation import train_test_split
import math


class Model():
    __metaclass__ = abc.ABCMeta

    def __init__(self, x_train, y_train, x_test, y_test, epoch=100, learning_rate=1e-3, batch_size=128, normalized=False, sess=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_classes = 43
        self.learning_rate = learning_rate
        self.epoch = epoch
        img_size = np.shape(x_train)[1]
        num_channels = np.shape(x_train)[3]
        self.x_image = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels])
        self.y_true = tf.placeholder(tf.int64, shape=[None, self.n_classes], name='y_true')
        self.batch_size = batch_size
        if normalized:
            self.x_train = self.normalize(self.x_train)
            self.x_test = self.normalize(self.x_test)

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.accuracy = None
        self.probabilities = None
        self.predictions = None
        self.last_layer = None

    def divide_train_val(self):
        self.x_train, self.valid_features, self.y_train, self.valid_labels = train_test_split(
            self.x_train,
            self.y_train,
            test_size=0.05)

    def batching(self, X, Y, batch_size):
        num_it = math.ceil(len(Y) / batch_size)
        start = 0
        for i in range(num_it):
            if start + batch_size >= len(Y):
                yield X[start:], Y[start:]
            else:
                yield X[start:start + batch_size], Y[start:start + batch_size]
            start += batch_size


    def test_big_data(self, sess, accuracy, x_test=None, y_test=None):
        total = 0
        cnt = 0
        if x_test is None and y_test is None:
            x_test = self.x_test
            y_test = self.y_test
        for batch_x, batch_y in self.batching(x_test, y_test, self.batch_size):
            # print("Batch size: ", np.shape(batch_y))
            feed_dict_train = {self.x_image: batch_x, self.y_true: batch_y}
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            total += acc
            cnt += 1

        return (total * 1.0)/cnt

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(self, input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

            # Rectified Linear Unit (ReLU).
            # It calculates max(x, 0) for each input pixel x.
            # This adds some non-linearity to the formula and allows us
            # to learn more complicated functions.
            #     layer = tf.nn.dropout(layer, 0.5)
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    # Flattens the image to 1D array
    def flatten_layer(self, layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    # Creates fully connected layer
    def new_fc_layer(self, input,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError("Please Implement build method")

    def train(self, normalize=False):
        if normalize:
            self.normalize(self.x_train)
            self.normalize(self.x_test)

        accuracy, y_pred, optimizer = self.build()
        model = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.x_train, self.y_train = Helper.randomize_data(self.x_train, self.y_train)
        self.divide_train_val()

        self.sess.run(model)
        for i in tqdm.tqdm(range(self.epoch)):
            for batch_x, batch_y in self.batching(self.x_train, self.y_train, self.batch_size):
                # print("Batch size: ", np.shape(batch_y))
                feed_dict_train = {self.x_image: batch_x, self.y_true: batch_y}
                acc = self.sess.run(optimizer, feed_dict=feed_dict_train)
                #         sess.run(model)
            if i % 10 == 0:
                acc = self.sess.run(accuracy, feed_dict=feed_dict_train)

                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

                feed_dict_train = {self.x_image: self.valid_features, self.y_true: self.valid_labels}
                acc2 = self.sess.run(accuracy, feed_dict=feed_dict_train)
                msg2 = "Optimization Iteration: {0:>6}, Validation Accuracy: {1:>6.1%}"

                print(msg.format(i + 1, acc))
                print(msg2.format(i + 1, acc2))
        save_path = saver.save(self.sess, "./Models/model.ckpt")
        print("Model saved in file: %s" % save_path)

        test_acc = self.test_big_data(self.sess, accuracy)
        print("Test Accuracy: ", test_acc)

    def normalize(self, x):
        size = np.shape(x)[0]
        for i in range(size):
            norm = np.copy(x[i])
            x[i] = cv2.normalize(x[i], norm, 0., 1., cv2.NORM_MINMAX, cv2.CV_32F)
        return x


    def recover_model(self, sess, model_path):
        saver = tf.train.import_meta_graph(model_path + ".meta")
        checkpoint_dir = './'
        path = tf.train.get_checkpoint_state(checkpoint_dir)
        print(path.model_checkpoint_path)
        saver.restore(sess, path.model_checkpoint_path)
        all_vars = tf.trainable_variables()
        # sess = tf.Session()
        # saver = tf.train.import_meta_graph(path+".meta")
        # path = tf.train.get_checkpoint_state(checkpoint_dir)
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        # all_vars = tf.trainable_variables()
        return all_vars


    def predict(self, x, y=None):
        if y is None:
            feed_dict_train = {self.x_image: x}
            # Returns probabilities and end result classes
            return self.sess.run(self.probabilities, feed_dict=feed_dict_train), \
                   self.sess.run(self.predictions, feed_dict=feed_dict_train)
        else:
            # Returns probabilities, end class results, and accuracy
            feed_dict_train = {self.x_image: x, self.y_true:y}
            return self.sess.run(self.probabilities, feed_dict=feed_dict_train), \
                   self.sess.run(self.predictions, feed_dict=feed_dict_train), \
                   self.sess.run(self.accuracy, feed_dict=feed_dict_train)

    def Destruct(self):
        self.sess.close()

# class Shallow2(Model):



# 0.84 accuracy
class Shallow(Model):
    def build(self):
        layer_conv1, weights_conv1 = \
            self.new_conv_layer(input=self.x_image,
                                num_input_channels=3,
                                filter_size=3,
                                num_filters=64,
                                use_pooling=False)



        layer_flat, num_features = self.flatten_layer(layer_conv1)

        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                      num_inputs=num_features,
                                      num_outputs=512,
                                      use_relu=True)

        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                      num_inputs=512,
                                      num_outputs=512,
                                      use_relu=True)

        layer_fc3 = self.new_fc_layer(input=layer_fc2,
                                      num_inputs=512,
                                      num_outputs=43,
                                      use_relu=False)

        y_pred = tf.nn.softmax(layer_fc3)

        y_pred_cls = tf.argmax(y_pred, dimension=1)
        y_true_cls = tf.argmax(self.y_true, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                                labels=self.y_true)
        self.last_layer = layer_fc2
        self.probabilities = y_pred
        self.predictions = y_pred_cls

        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        # print(y_pred_cls, y_true_cls)

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy, y_pred, optimizer


# 0.93 ccuracy
class Shallow2(Model):
    def build(self):
        layer_conv1, weights_conv1 = \
            self.new_conv_layer(input=self.x_image,
                                    num_input_channels=3,
                                    filter_size=3,
                                    num_filters=64,
                                    use_pooling=False)


        layer_conv2, weights_conv2 = \
                self.new_conv_layer(input=layer_conv1,
                                    num_input_channels=64,
                                    filter_size=3,
                                    num_filters=128,
                                    use_pooling=True)

        layer_conv3, weights_conv3 = \
                self.new_conv_layer(input=layer_conv2,
                                    num_input_channels=128,
                                    filter_size=3,
                                    num_filters=128,
                                    use_pooling=True
                                    )


        layer_flat, num_features = self.flatten_layer(layer_conv3)
        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                          num_inputs=num_features,
                                          num_outputs=512,
                                          use_relu=True)
        layer_fc1 = tf.nn.dropout(layer_fc1, 0.5)


        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                          num_inputs=512,
                                          num_outputs=43,
                                          use_relu=False)

        y_pred = tf.nn.softmax(layer_fc2)

        y_pred_cls = tf.argmax(y_pred, dimension=1)
        y_true_cls = tf.argmax(self.y_true, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                    labels=self.y_true)
        self.last_layer = layer_fc2
        self.probabilities = y_pred
        self.predictions = y_pred_cls

        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy, y_pred, optimizer

# YUV
# 0.9375 accuracy with resampled
# 0.9252 without sampling

# RGB
# 0.9568 accuracy with resampled data

# HLS
# 0.9237 accuracy with resampled data

class Shallow3(Model):
    def build(self):
        layer_conv1, weights_conv1 = \
            self.new_conv_layer(input=self.x_image,
                                    num_input_channels=3,
                                    filter_size=3,
                                    num_filters=32,
                                    use_pooling=False)


        layer_conv2, weights_conv2 = \
                self.new_conv_layer(input=layer_conv1,
                                    num_input_channels=32,
                                    filter_size=3,
                                    num_filters=64,
                                    use_pooling=True)

        layer_conv3, weights_conv3 = \
                self.new_conv_layer(input=layer_conv2,
                                    num_input_channels=64,
                                    filter_size=3,
                                    num_filters=32,
                                    use_pooling=True
                                    )

        layer_conv4, weights_conv4 = \
            self.new_conv_layer(input=layer_conv3,
                                num_input_channels=32,
                                filter_size=5,
                                num_filters=32,
                                use_pooling=True
                                )

        layer_conv5, weights_conv5 = \
            self.new_conv_layer(input=layer_conv4,
                                num_input_channels=32,
                                filter_size=1,
                                num_filters=16,
                                use_pooling=True
                                )

        layer_flat, num_features = self.flatten_layer(layer_conv5)
        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                          num_inputs=num_features,
                                          num_outputs=512,
                                          use_relu=True)
        layer_fc1 = tf.nn.dropout(layer_fc1, 0.5)


        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                          num_inputs=512,
                                          num_outputs=43,
                                          use_relu=False)
        y_pred = tf.nn.softmax(layer_fc2)


        y_pred_cls = tf.argmax(y_pred, dimension=1)
        y_true_cls = tf.argmax(self.y_true, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                    labels=self.y_true)
        self.last_layer = layer_fc2
        self.probabilities = y_pred
        self.predictions = y_pred_cls

        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        return accuracy, y_pred, optimizer


