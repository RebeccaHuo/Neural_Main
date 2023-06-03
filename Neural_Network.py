# todo docstring
import time
from collections import deque
import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class Neuralnetwork:
    # todo docstring
    def __init__(self, layers, learning_rate):
        """ "Initialising all the variables"""

        self.all_weights = []  # List of weights
        self.delta_weights = []  # List of delta weights used to update weights
        self.length_input = 785 # Number of pixels
        self.learning_rate = learning_rate
        self.error = 0 # Initial error
        self.layers = layers
        self.activation_functions = ["ReLU"] * (len(layers) - 1) + ["sigmoid"] # Use ReLU for all layers except the last one because?
        self.ff = [] # A list to append feedforward from later in the message
        self.activation_list = [] # Empty list to append activations
        self.weight_list = [] # Empty list to append updated weights

    def normal_init(self, shape, mean=0, std=0.1):
        return np.random.normal(loc=mean, scale=std, size=shape)

    def initialise_weight(self):
        for layer in self.layers:
            self.all_weights.append(self.normal_init((self.length_input, layer)))
            self.length_input = layer

    def standardisation(self, data):
        # 
        return (data - np.mean(data)) / np.std(data)
