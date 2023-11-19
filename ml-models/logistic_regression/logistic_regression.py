import numpy as np
from dataclasses import dataclass

"""
Linear regression: f(x) = xW + b
Logistic regression: g(x) = Sigmoid(f(x)) = Sigmoid(xW + b)
Linear regression generates values from (essentially) -inf to inf. For Logistic Regression, 
we need to map these values to a [0, 1] interval. We do this via the sigmoid activation function.

Sigmoid(x) = 1 / (1 + exp(-x))
Sigmoid(x)' = Sigmoid(x) * (1 - Sigmoid(x))
"""

"""
Log-loss = -(y log(g(x)) + (1-y) log(1-g(x)))   # g(x) = probability 
E = Mean log-loss = mean(log-loss)

(Log-loss)' = f'(x) * (h - y)
            = (xW + b)' * (h - y)

dE/dW = 1/n (X) * (Sigmoid(X) - Y)
dE/dB = 1/n (Sigmoid(X) - Y)
"""

@dataclass
class LogisticRegression:
    learning_rate: float
    epochs: int
    threshold: float    # e.g. threshold = 0.8, if prob = 0.5 < threshold => 0 ; prob = 0.9 > threshold => 1
    logging: bool

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def mean_log_loss(self, predictions: np.ndarray, labels: np.ndarray) -> np.float32:
        """
        Computes the mean Cross Entropy Loss (in binary classification, also called Log-loss)
        """    
        return -np.mean(labels * np.log(predictions) + (1-labels) * np.log(1 - predictions))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fits the logistic regression model.
        """
        num_samples, num_features = features.shape
        
        # Initialise weights and bias at 0
        self.weights, self.bias = np.zeros(num_features), 0
        
        # Fit over epochs number of times using batch gradient descent
        for epoch in range(self.epochs):
            predictions = self.predict_prob(features) 
            difference = predictions - labels

            weights_gradient = features.T.dot(difference) / num_samples
            bias_gradient = difference.sum() / num_samples
            
            self.weights -= self.learning_rate * weights_gradient
            self.bias -= self.learning_rate * bias_gradient

            if self.logging:
                print(f"Mean Log-loss [{epoch}]: {self.mean_log_loss(predictions, labels):.3f}")

    def predict_prob(self, features: np.ndarray) -> np.ndarray:
        """
        Returns the probability of each label using the given features.
        """    
        return self.sigmoid(features.dot(self.weights) + self.bias)

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Uses the given features to predict a label, 
        """
        return np.where(self.sigmoid(features.dot(self.weights) + self.bias) < self.threshold, 0, 1)