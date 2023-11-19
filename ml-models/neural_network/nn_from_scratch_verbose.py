import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    # Layer has two methods - forward_propagation and backward_propagation.
    # The forward method takes in the input and returns the output.
    # The backward method takes in the derivative of the error wrt the output, i.e. output_gradient, and the learning rate. 
    # The backward method updates the parameters and returns the derivative of the error wrt the input of the layer.
    def forward_propagation(self, input):
        # Returns output
        pass

    def backward_propagation(self, output_gradient, learning_rate):
        # Updates parameters and returns input gradient
        pass

# In a dense (i.e. fully-connected) layer, each input neuron is connected to every output neuron.
# Each connection represents a weight (w_ji = weight connecting output neuron j to input neuron i).
# Every output value (i.e. at each neuron) is computed as the sum of all the inputs multiplied by all the 
# weights connecting them to that specific output, with an additional bias term. Just like the weights, the biases are trainable parameters.
# => Y (jx1) = W (jxi) . X (ix1) + B (jx1)
class Dense(Layer):
    # Inherits from base class Layer
    # input_size = no. of input neurons
    # output_size = no.of output neurons
    def __init__(self, input_size, output_size):
        # Weights and biases are initialised randomly with dimensions mentioned above.
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(output_size, 1) - 0.5
    
    # Forward propagation involves computing the linear combination of the input neurons.
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        # The forward method computes Y = W . X + B, outputting a matrix (1-d array) of linear combinations.
        return self.output
    
    # During backward propagation, we apply gradient descent to update our model's parameters.
    # i.e. w = w - learning_rate * dE/dw
    # From our derivations using chain rule, we have several equations to use in our backward propagation function:
    # (Aside: output_gradient = dE/dY)
    # weights_gradient = dE/dW = dE/dY * X^T
    # bias_gradient = dE/dB = dE/dY
    # input_gradient = dE/dX = W^T * dE/dY
    def backward_propagation(self, output_gradient, learning_rate):
        # dE/dW = dE/dY * X^T 
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Adjust weight via gradient descent, i.e. subtracting weights by learning_rate * weights_gradient
        # Adjust bias via gradient descent, i.e. subtracting bias by learning_rate * bias_gradient = learning_rate * output_gradient
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient

# The Activation Layer takes in input neurons (i) and passes them through an activation function.
# i.e. Y = f(X), f = Activation Function applied to every element of X
# From chain rule, we have dE/dX = dE/dY * f_prime(X), where * = element-wise multiplication.
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        # activation = activation function
        # activation_prime = the derivative of the activation function
        self.activation = activation
        self.activation_prime = activation_prime
    
    # Forward method applies activation to the input
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    # Backward method returns the deriviative of the error to the input
    # i.e. dE/dX = dE/dY * f_prime(X)
    def backward_propagation(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
# Suppose we use tanh(x) as our activation function. Its derivative will be 1-(tanh(x))^2.
# This function is non-linear, as required by our neural network, otherwise the model will essentially be a linear model.
class Tanh(Activation):
    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)

# Alternate activation function of ReLU
class ReLU(Activation):
    def relu(self, x):
        return np.maximum(x, 0)

    def relu_prime(self, x):
        return np.where(x > 0, 1, 0)

    def __init__(self):
        super().__init__(self.relu, self.relu_prime)

# For our loss function, we will use Mean Squared Error (MSE)
# Given y = actual output and y* = output of the neural network
# MSE = E = 1/n sum(y - y*)^2
# Recall: For each layer, we receive an input of dE/dY during backpropagation given by the next layer via dE/dX 
# (as the input to a layer is equivalent to the output from the previous layer).
# dE/dY = 2/n (Y - Y*) = output_gradient in final layer
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    
    # Add a layer to network
    def add(self, layer):
        self.layers.append(layer)
    
    # Set loss function to use
    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    # Predict output for given input
    def predict(self, input_data):
        n_samples = len(input_data)
        result = []

        # Run network over all samples
        for i in range(n_samples):
            # Forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        return result
    
    # Train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        n_samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(n_samples):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # Compute loss
                err += self.loss(y_train[j], output)

                # Backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                
            # Calculate average error on all samples
            err /= n_samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

