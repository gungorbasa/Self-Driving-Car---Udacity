import numpy as np
import math


def batching( X,batch_size):
    num_it = math.ceil(len(X) / batch_size)
    start = 0
    for i in range(num_it):
        if start + batch_size >= len(X):
            yield X[start:]
        else:
            yield X[start:start + batch_size]
        start += batch_size

A = np.arange(0,100)


def randomize_data(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


i, j = randomize_data(A, A)

print(i)
print(i)


