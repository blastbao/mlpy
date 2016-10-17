import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import network2 as network


def plot(pixels):
    pixels = np.array(pixels, dtype='float32')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.imshow(pixels, cmap='gray')
    plt.show()


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
arr, label = test_data[-1]
net = network.Network([784, 100, 10])
net.default_weight_initializer()
net.stochastic_gradient_descent(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
print net.predict(arr)
