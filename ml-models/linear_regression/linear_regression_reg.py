import numpy as np

"""
f(x) = y* = xW + b
outcome = features * weights + bias

Using MSE as our cost/loss function: E = 1/n (y - y*)^2

Derivative wrt weights
dE/dW = 2/n (y - y*) (y - xW - b)'
      = 2/n (y - y*) (-x)
      = -2/n (y - y*) (x)

Derivative wrt bias
dE/dB = 2/n (y - y*) (y - xW - b)'
      = 2/n (y - y*) (-1)
      = -2/n (y - y*) => residuals
"""

"""
L2 Regularisation:
E = MSE + lambda * W^2
dE/dW = MSE' + 2 * lambda * W
      = -2/n (y - y*) (x) + 2 * lambda * W
"""

"""
L1 Regularisation:
E = MSE + lambda * |W|
dE/dW = MSE' + lambda * sign(W)     # sign(W) = 1 if W > 0, sign(W) = -1 if W <= 0
      = -2/n (y - y*) (x) + lambda * sign(W)
"""
class Regularisation:
    def __init__(self):
        pass
    
    def regularisation_penalty(self, weights):
        return
    
    def regularisation_gradient(self, weights):
        return

class L2Regularisation(Regularisation):
    def __init__(self, l2_lambda: float):
        self.l2_lambda = l2_lambda
    
    def regularisation_penalty(self, weights):
        return np.sum(self.l2_lambda * weights**2)
    
    def regularisation_gradient(self, weights):
        return 2 * self.l2_lambda * weights
      
class L1Regularisation(Regularisation):
    def __init__(self, l1_lambda: float):
      self.l1_lambda = l1_lambda
    
    def regularisation_penalty(self, weights):
        return np.sum(self.l1_lambda * weights)
    
    def regularisation_gradient(self, weights):
        return self.l1_lambda * np.where(weights >= 0, 1, -1)

class LinearRegression:
    def __init__(self, learning_rate: float, epochs: int, logging: bool):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.logging = logging

    def fit(self, features: np.ndarray, labels: np.ndarray, regularisation = None, regularisation_factor = 1): 
        """
        Fits the linear regression model.
        """
        self.features = features
        self.labels = labels
        self.regularisation = regularisation
        self.regularisation_factor = regularisation_factor

        num_samples, num_features = self.features.shape
        
        # Initialise weights and bias at 0
        self.weights, self.bias = np.zeros(num_features), 0

        # Instantiate regularisation
        if self.regularisation == 'l1':
            regularisation_parameter = L1Regularisation(regularisation_factor)
        elif self.regularisation == 'l2':      
            regularisation_parameter = L2Regularisation(regularisation_factor)
        else:
            regularisation_parameter = L2Regularisation(0)
        
        # Fit over epochs number of times using batch gradient descent
        for epoch in range(self.epochs):
            residuals = self.labels - self.predict(self.features)

            weights_gradient = -2/num_samples * residuals.dot(self.features) + regularisation_parameter.regularisation_gradient(self.weights)
            bias_gradient = -2/num_samples * residuals.sum()
            
            self.weights -= self.learning_rate * weights_gradient
            self.bias -= self.learning_rate * bias_gradient

            if self.logging:
                print(f"MSE Loss [{epoch}]: {np.mean(np.square(residuals)):.3f}")

    def predict(self, features: np.ndarray):
        """
        Uses features to predict an outcome, f(X) = XW + b
        """
        return features.dot(self.weights) + self.bias