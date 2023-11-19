import numpy as np
from dataclasses import dataclass

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

class LinearRegression:
    def __init__(self, learning_rate: float, epochs: int, logging: bool):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.logging = logging

    def fit(self, features: np.ndarray, labels: np.ndarray): 
        """
        Fits the linear regression model.
        """
        self.features = features
        self.labels = labels
        num_samples, num_features = self.features.shape
        
        # Initialise weights and bias at 0
        self.weights, self.bias = np.zeros(num_features), 0
        
        # Fit over epochs number of times using batch gradient descent
        for epoch in range(self.epochs):
            residuals = self.labels - self.predict(self.features)
            
            weights_gradient = -2/num_samples * residuals.dot(self.features)
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