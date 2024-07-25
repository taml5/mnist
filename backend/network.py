"""Contains the neural network."""
from typing import Optional

import random
from PIL import Image
import numpy as np


class Neuron:
    """A neuron in the network.

    Attributes:
        weights: An array storing the weights for each connection to a neuron in the previous layer.
        bias: The bias applied for this neuron.
    """
    weights: np.ndarray
    bias: float

    def __init__(self, n_inputs: int):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def feedforward(self, inputs: np.ndarray) -> float:
        """Return the corresponding output of this neuron when the weights and biases are applied
        to the input values.

        Precondition:
         - The dimensions of ``inputs`` is the same as the dimensions of ``self.weights``
        """
        total = np.dot(self.weights, inputs) + self.bias
        return total


class Layer:
    """A single layer of neurons in the network.

    Attributes:
        input_size: The number of weights from the input or previous layer.
        neurons: The neurons in this layer.
    """
    input_size: int
    neurons: tuple[Neuron]

    def __init__(self, size: int, input_size: int):
        self.input_size = input_size
        self.neurons = tuple(Neuron(input_size) for _ in range(0, size))

    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """Return the corresponding outputs of this layer when the input values are applied
        to each neuron in the layer.

        Preconditions:
         - The dimensions of ``inputs`` is the same as the dimensions of ``weights`` for every neuron
        """
        return np.array([sigmoid(neuron.feedforward(inputs)) for neuron in self.neurons])

    def weighted(self, inputs: np.ndarray) -> np.ndarray:
        """Return the weighted activation of this layer."""
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
        test_results = [(np.argmax(self.compute(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def test_one(self, testing_data: list[tuple[np.ndarray, int]]) -> tuple:
        """
        Open a random test image and apply the network to it.

        :param testing_data: The testing data. This is represented as a list of tuples (x, y), where x is a numpy
                             array of 784-dimensional numpy arrays, and y is the corresponding classifaction of x.
        :return: a 2-tuple (guess, ans), where guess is the output from the neural network, and ans is the label of the
                 testing image.
        """
        image_data = random.choice(testing_data)
        image = Image.fromarray((image_data[0].reshape(28, 28) * 255).astype(np.uint8), 'L')
        image.show()

        return np.argmax(self.compute(image_data[0])), image_data[1]

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
        print("Starting training...")
        for i in range(1, epochs + 1):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j + batch_size] for j in range(0, len(training_data), batch_size)]

            for mini_batch in mini_batches:
                self.train_on_batch(mini_batch, learning_rate)

            if test_data is not None:
                print(f"Epoch {i} of {epochs}: {self.evaluate(test_data)} out of {len(test_data)} inputs")
            else:
                print(f"Epoch {i} of {epochs} completed")

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
        # initialise matrices for gradients of weights and biases per layer
        nabla_w = [np.zeros((len(layer.neurons), layer.input_size)) for layer in self.layers]
        nabla_b = [np.zeros(len(layer.neurons)) for layer in self.layers]

        # apply backpropagation to get updated weights and biases
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagate(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb.transpose() for nb, dnb in zip(nabla_b, delta_nabla_b)]

        # update weights and biases based on mini batch
        for i in range(0, len(self.layers)):
            layer = self.layers[i]
            layer_weights, layer_biases = nabla_w[i], nabla_b[i]
            for j in range(0, len(layer.neurons)):
                neuron = layer.neurons[j]
                neuron.bias -= (learning_rate / len(mini_batch)) * layer_biases[0][j]
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= (learning_rate / len(mini_batch)) * layer_weights[j][k]

    def backpropagate(self, x: np.ndarray[np.ndarray[int]], y: np.ndarray[int]) -> tuple:
        """Apply the backpropagation algorithm to the neural network using the test input x and the
        label y.

        First, the weighted inputs and activations of each neuron are calculated and recorded through a single
        forward pass in the network. Once the output layer is reached, the backpropagation algorithms are applied
        backwards to the input layer to calculate the partial derivatives of the cost function
        with respect to the weights and biases. These partial derivatives are then recorded in matrices and returned.

        :param x: The test input, consisting of a 784-dimensional array of ints representing the brightness of a
                  pixel in the test image.
        :param y: The label, consisting of a 10-dimensional unit vector where non-zero index is the digit of the
                  image.
        :returns: A 2-tuple (nabla_w, nabla_b) representing the changes in weights and the changes in biases
                  respectively.
        """
        nabla_w = [np.zeros((layer.input_size, len(layer.neurons))) for layer in self.layers]
        nabla_b = [np.zeros(len(layer.neurons)) for layer in self.layers]

        # compute the activations and weighted inputs of each neuron through the network
        input = x
        activations = [x]  # first "activation" is the input layer aka the input
        zs = []  # no weighted input for the first layer
        # forward propagate: store weighted inputs and activations
        for layer in self.layers:
            z = layer.weighted(input)
            input = layer.feedforward(input)
            activations.append(input)
            zs.append(z)

        # at this point, input is the final output and zs[-1] is the final weighted input of the network.
        # we can use it to compute the error in the output layer.
        delta = cost_derivative(input, y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        # backpropagate delta through the layers
        for i in range(len(self.layers) - 2, -1, -1):
            # get delta and weight of (l + 1)th layer
            weights = np.array([[weight for weight in neuron.weights] for neuron in self.layers[i + 1].neurons])
            delta = np.dot(weights.transpose(), delta) * sigmoid_prime(zs[i])

            # apply backpropagation algorithms
            nabla_w[i] = np.dot(delta, activations[i].transpose())
            nabla_b[i] = delta

        return nabla_w, nabla_b


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    """Apply the sigmoid function to x."""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: float | np.ndarray) -> np.ndarray:
    """Apply the derivative of the sigmoid function to x."""
    return sigmoid(x) * (1 - sigmoid(x))


def cost_derivative(activations: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the gradient of the cost function with respect to the output activations."""
    return activations - y
