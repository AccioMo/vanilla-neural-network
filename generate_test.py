import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save the data as binary files
x_train.tofile('mnist_train_images.bin')
y_train.tofile('mnist_train_labels.bin')
x_test.tofile('mnist_test_images.bin')
y_test.tofile('mnist_test_labels.bin')
