import mnist_loader
import network2 as network


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])
net.default_weight_initializer()
net.stochastic_gradient_descent(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
