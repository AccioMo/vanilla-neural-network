import tensorflow as tf
import matplotlib.pyplot as plt
import ssl

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Function to plot multiple images
def plot_mnist_images(images, labels, num_images=5, rows=1, cols=5):
    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.close()

# Plot first 5 training images
plot_mnist_images(x_train, y_train)

print("MNIST sample images saved as 'mnist_samples.png'")