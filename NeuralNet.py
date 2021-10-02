# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

# Testing data form NNFS
###############################################################################
#nnfs.init()
#X, y = spiral_data(100, 3)
###############################################################################

# Testing data from CSV files
###############################################################################
#xtrain = np.loadtxt('X_train.csv')
#normalized_xtrain = xtrain / np.linalg.norm(xtrain)
#ytrain = np.loadtxt('Y_train.csv')
#normalized_ytrain = ytrain / np.linalg.norm(ytrain)
#xtest = np.loadtxt('X_test.csv')
#normalized_xtest = xtest / np.linalg.norm(xtest)
#ytest = np.loadtxt('Y_test.csv')
#normalized_ytest = ytest / np.linalg.norm(ytest)
###############################################################################

# Manual entry of testing data
###############################################################################
# Vectors inputs
#inputs = [1, 2, 3, 2.5]
#
#weights1 = [0.2, 0.8, -0.5, 1.0]
#weights2 = [0.5, -0.91, 0.26, -0.5]
#weights3 = [-0.26, -0.27, 0.17, 0.87]
#
#bias1 = 2
#bias2 = 3
#bias3 = 0.5

# Matrix inputs
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2, 3, 0.5]


weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.33, 0.13]]

biases2 = [-1, 2, -0.5]
###############################################################################

def main():
    #layer1 = Layer(4,5)
    #activation1 = ActivationReLU()
    #layer1.forward(X)
    #layer1_output = np.dot(inputs, np.array(weights1).T) + biases1
    #layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
    #print(layer2_output)

    layer1 = Layer(4,5)
    layer2 = Layer(5,2)

    layer1.forward(inputs)
    print(layer1.output)

###############################################################################

# Layer class definition
###############################################################################
class Layer:
    def __init__ (self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(input, self.weights) + self.biases

class ActivationSigmoid:
    def activation(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs)) 
    def derivative(self, inputs):
        a = np.zeros(inputs.shape)

        num_row, run_col = inputs.shape
        for i in range(num_row):
            a[i,:] = self.activation(inputs[i,:]) * (1 - self.activation(inputs[i,:]))

        return a

class ActivationReLU:
    def activation(self, inputs):
        self.output = np.maximum(0, inputs)

    def derivative(self, inputs):
        inputs[inputs<=0] = 0
        inputs[inputs>0] = 1
        return inputs

class ActivationTanh:
    def activation(self, inputs):
        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def derivative(self, inputs):
        a = np.zeros(inputs.shape)

        num_row, run_col = inputs.shape
        for i in range(num_row):
            a[i,:] = 1 - np.square(self.activation(inputs[i,:]))

        return a

###############################################################################

# Run main function
###############################################################################
if __name__ == "__main__":
    main()
###############################################################################
