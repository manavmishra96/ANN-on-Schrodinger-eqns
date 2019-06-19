"""
Python code to train the Neural Network using numpy package.
It is a pure pythonic code using no exclusive third party library functions
"""

import numpy as np 
import json
import random as ra
from matplotlib.pylab import *

from load_data import load_data_nn

training_data, validation_data = load_data_nn()

class Neural_network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):
        a = np.array(a); a = a.reshape((a.shape[0], 1))
        for b,w in zip(self.biases[:-1], self.weights[:-1]):
            a = ReLU(np.dot(w,a) + b)
        #No activation function in the output layer
        a = np.dot(self.weights[-1], a) + self.biases[-1]    
        return a
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = np.array(x); activation = activation.reshape((activation.shape[0], 1))
        activations = []       #list to store all activations layer by layer
        activations.append(activation)
        zs = []                 #list to store all z vectors layer by layer
        for b,w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = ReLU(z)
            activations.append(activation)
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = z
        y = np.array(y); y = y.reshape((y.shape[0], 1))
        activations.append(activation)   
        #backward pass
        delta = (activations[-1] - y) * ReLU_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = ReLU_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)    

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Update weights and biases
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]    
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0):
        n_data = len(validation_data)
        n = len(training_data)
        validation_cost = []; validation_accuracy = []
        training_cost = []; training_accuracy = []
        epch = []
        for j in range(epochs):
            ra.shuffle(training_data)     
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]   
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data)) 
            print ("Epoch %s: training complete" % j)
            epch.append(j)
            #Print cost on the training_data
            costs = self.total_cost(training_data, lmbda)
            training_cost.append(costs)
            print ("Cost on training data: {}".format(costs))
            #Print cost on the validation_data
        plot(epch, training_cost)
        show()
        return validation_cost, validation_accuracy, training_cost, training_accuracy    

    def total_cost(self, data, lmbda):
        #Returns the total cost of the data set J(theta)
        cost = 0.0
        for x, y in data:
            x = np.array(x); x = x.reshape((x.shape[0], 1))
            y = np.array(y); y = y.reshape((y.shape[0], 1))
            a = self.feedforward(x)
            a = np.array(a)
            val = cost1(a, y)
            cost += 1.0 * val / len(data)
        cost += 0.5*(lmbda/len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost  
 
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data,f)
        f.close() 

    def load(self, filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        net = Neural_network(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net     



def cost1(a, y):
    """Return the cost associated with an output ``a`` and desired output``y``"""
    return 0.5 * np.linalg.norm(a-y)

def ReLU(z):
    """The ReLU function."""
    return z * (z > 0)

def ReLU_prime(z):
    """Derivative of the ReLU function."""
    return 1. * (z > 0)


if __name__ == '__main__':
    net = Neural_network([127, 200, 200, 127])
    #net.SGD(training_data, 1500, 300, 18.0, 0.1)
    #net.save("matrix.csv")
    net.load("matrix.csv")
    i = ra.randint(0, len(training_data)-1)
    x, y = training_data[i]
    var = [i+1 for i in range(127)]
    p = net.feedforward(x)
    plot(var, x, color = 'm')
    plot(var, y, color = 'r')
    plot(var, p, color = 'g')
    show()
    