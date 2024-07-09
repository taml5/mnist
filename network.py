"""TODO: fill this docstring in"""
import numpy as np


def sigmoid(x: float) -> float:
    """TODO: fill this docstring in"""
    return 1 / (1 + np.exp(-x))


class Neuron:
    """TODO: fill this docstring in"""
    n_inputs: int
    weights: np.ndarray
    bias: float

    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def feedforward(self, inputs: np.ndarray) -> float:
        """TODO: fill this docstring in"""
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class Layer:
    """TODO: fill this docstring in"""
    neurons: tuple[Neuron]

    def __init__(self, size: int, prev_layer_size: int):
        self.neurons = tuple(Neuron(prev_layer_size) for _ in range(0, size))

    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """TODO: fill this docstring in"""
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])


class Network:
    """TODO: fill this docstring in"""
    layers: tuple[Layer]
    input_size: int

    def __init__(self, input_size: int, sizes: list[int]):
        self.input_size = input_size

        layers = [Layer(sizes[0], input_size)]
        for i in range(1, len(sizes)):
            layers.append(Layer(sizes[i], sizes[i - 1]))
        self.layers = tuple(layers)

    def evaluate(self, input: np.ndarray) -> np.ndarray:
        """TODO: fill this docstring in"""
        for layer in self.layers:
            input = layer.feedforward(input)

        return input
