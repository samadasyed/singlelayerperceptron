import numpy as np

class Perceptron:
    def __init__(self, input_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Network
        self.weights = np.zeros(input_size)
        self.bias = 0
