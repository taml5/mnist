"""Contains the neural network."""
from typing import Optional

import numpy as np
import random


class Neuron:
    """A neuron in the network.

    Attributes:
        n_inputs: The number of inputs from the previous layer or the input neurons.
        weights: An array storing the weights for each connection to a neuron in the previous layer.
        bias: The bias applied for this neuron.
    """
    n_inputs: int
    weights: np.ndarray
    bias: float

    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def feedforward(self, inputs: np.ndarray) -> float:
        """Return the corresponding output of this neuron when the weights and biases are applied
        to the input values.

        Precondition:
         - The dimensions of ``inputs`` is the same as the dimensions of ``self.weights``
        """
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class Layer:
    """A single layer of neurons in the network.

    Attributes:
        neurons: the neurons in this layer.
    """
    neurons: tuple[Neuron]

    def __init__(self, size: int, prev_layer_size: int):
        self.neurons = tuple(Neuron(prev_layer_size) for _ in range(0, size))

    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """Return the corresponding outputs of this layer when the input values are applied
        to each neuron in the layer.

        Preconditions:
         - The dimensions of ``inputs`` is the same as the dimensions of ``weights`` for every neuron
        """
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])


class Network:
    """The neural network.

    Attributes:
        layers: The neuron layers that make up this neural network.
    """
    layers: tuple[Layer]

    def __init__(self, sizes: list[int]):
        """Initialise the neural network with ``len(sizes)`` layers, and the number of neurons given by
        the corresponding index. ``sizes[0]`` represents the input layer, and ``sizes[len(sizes) - 1]``
        represents the output layer.
        """
        self.layers = tuple(Layer(sizes[i], sizes[i - 1]) for i in range(1, len(sizes)))

    def evaluate(self, test_data: list[tuple[np.ndarray, int]]) -> int:
        """Evaluate the performance of the neural network against test_data.

        :returns: The number of test inputs that the neural network outputs a correct result for.
        """
        test_results = [np.argmax(self.compute(x), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def compute(self, input: np.ndarray[int]) -> np.ndarray:
        """Apply the neural network to the given input.

        Preconditions:
         - The size of ``input`` is the same as the size of the input layer

        :param input: The input array for the network.
        :returns: The output of the neural network.
        """
        for layer in self.layers:
            input = layer.feedforward(input)

        return input

    def train(self,
              training_data: list[tuple[np.ndarray[np.ndarray[int]], np.ndarray[int]]],
              epochs: int,
              batch_size: int,
              learning_rate: float,
              test_data: Optional[list[tuple[np.ndarray, int]]] = None) -> None:
        """Train the neural network using stochastic gradient descent.

        :param training_data: The training data used to update the network. This is represented as a list of tuples
                              (x, y), where x is a numpy array of 784-dimensional numpy arrays, and y is a
                              10-dimensional numpy array representing the unit vector of the digit for x.
        :param epochs: The number of epochs that the training lasts for. Completing an epoch means that the network has
                       trained on the complete training data set.
        :param batch_size: The size of the mini-batches used to train the neural network.
        :param learning_rate: The learning rate of the neural network.
        :param test_data: The optional testing data used to evaluate the performance of the network. This is represented
                          as a list of tuples (x, y), where x is a numpy array of 784-dimensional numpy arrays, and y is
                          the corresponding classifaction of x.
        """
        for i in range(0, epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j + batch_size] for j in range(0, len(training_data), batch_size)]

            for mini_batch in mini_batches:
                self.train_on_batch(mini_batch, learning_rate)

            if test_data is not None:
                print(f"Epoch {i}: {self.evaluate(test_data)} out of {len(test_data)} inputs")
            else:
                print(f"Epoch {i} completed")

    def train_on_batch(self,
                       mini_batch: list[tuple[np.ndarray[np.ndarray[int]], np.ndarray[int]]],
                       learning_rate: float) -> None:
        """Update the weights and biases of the neural network via gradient descent using backpropagation
        on the given mini batch.

        :param mini_batch: The mini batch used to update the network. This is represented as a list of tuples
                           (x, y), where x is a numpy array of 784-dimensional numpy arrays, and y is a
                           10-dimensional numpy array representing the unit vector of the digit for x.
        :param learning_rate: The learning rate of the network.
        """
        return


def sigmoid(x: float) -> float:
    """Apply the sigmoid function to x."""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: float) -> float:
    """Apply the derivative of the sigmoid function to x."""
    return sigmoid(x) * (1 - sigmoid(x))
