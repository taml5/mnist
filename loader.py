"""Loads the MNIST data into numpy arrays to be fed into a neural network."""
import pickle
import gzip
import numpy as np


def load_data(filepath: str) -> tuple[list, list, list]:
    """Load the data from the given package at filepath.

    :returns: A tuple of three lists, corresponding to training data, validation data, and test data respectively.
              Each list is a list of 2-tuples, the first value of which is a 784-dimensional array representing an
              image of a handwritten digit, and the second a 10-dimensional array representing the digit of the
              corresponding image in unit vector form.
    """
    f = gzip.open(filepath, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [_vectorise(digit) for digit in training_data[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])

    return list(training_data), list(validation_data), list(test_data)


def _vectorise(digit: int) -> np.ndarray:
    """Turn the given digit into a 10-dimensional vector to be used in the network."""
    vec = np.zeros(10)
    vec[digit] = 1.0
    return np.reshape(vec, (10, 1))
