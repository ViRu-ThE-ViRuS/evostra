import numpy as np
import _pickle as pickle


class FeedForwardNetwork(object):
    def __init__(self, layer_sizes):
        self.weights = []
        self.structure = layer_sizes
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.zeros(shape=(layer_sizes[index], layer_sizes[index+1])))

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for layer in self.weights:
            out = np.dot(out, layer)
            out = np.maximum(out, 0)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)

    def __str__(self):
        return self.structure.__str__()
