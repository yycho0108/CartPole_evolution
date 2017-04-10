import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class Net:
    def __init__(self, w):
        self.w = w
    def compute(self, inputs):
        output = inputs
        for w in self.w:
            output = np.tanh(np.dot(output, w))
        return output[0]
