import numpy as np

class PCA:
    def __init__(self, n:int):
        self.n = n
        
    def standardise(self, X):
        """
        Standardise the data by subtracting the mean and scaling by the standard deviation for each feature.
        This ensures that all the features have a mean of zero and a standard deviation of one.
        z = (x - u)/s
        """
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        self.std_dev = np.std(X, axis=0)
        X /= self.std_dev

        return X

    def fit(self, X):
        # Standardise X
        X = self.standardise(X)
        
        # Get covariance. Transpose is required due to np.cov requirement.
        cov = np.cov(X.T)

        # Get eigenvectors, eigenvalues - are returned as column vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors in decreasing order 
        eigenvectors = eigenvectors.T
        ev_sorted_idx = np.argsort(eigenvalues)[::-1][:self.n]
        # eigenvalues = eigenvalues[ev_sorted_idx]
        # eigenvectors = eigenvectors[ev_sorted_idx]

        # Store first n eigenvectors
        self.components = np.array([eigenvectors[idx] for idx in ev_sorted_idx])
    
    def transform(self, X):
        # project data
        X = self.standardise(X)
        return np.dot(X, self.components.T)