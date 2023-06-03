import unittest
import numpy as np
from scipy.special import expit
import sys
sys.path.append('..')

from src.MNIST_multi_classification import Neuralnetwork


class TestNeuralnetwork(unittest.TestCase):
    def setUp(self):
        self.nn = Neuralnetwork(layers=[10, 5, 2], learning_rate=0.1)

    def test_normal_init(self):
        shape = (3, 3)
        weights = self.nn.normal_init(shape)
        self.assertEqual(weights.shape, shape)

    def test_initialise_weight(self):
        self.nn.initialise_weight()
        self.assertEqual(len(self.nn.all_weights), len(self.nn.layers))

    def test_standardisation(self):
        data = np.array([1, 2, 3, 4, 5])
        standardized_data = self.nn.standardisation(data)
        expected_output = (data - np.mean(data)) / np.std(data)
        np.testing.assert_array_equal(standardized_data, expected_output)

if __name__ == "__main__":
    unittest.main()