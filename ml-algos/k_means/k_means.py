import numpy as np

from dataclasses import dataclass

@dataclass
class KMeans:
    k: int
    iterations: int
    # tol: float      

    def euclidean_distance(self, data_point: np.ndarray, centroids:np.ndarray):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
    
    def fit(self, samples: np.ndarray) -> None:
        """
        Clusters the data.
        """
        num_samples, num_features = samples.shape

        # Randomly select k data points to initialise the centroids
        self.centroids = samples[np.random.choice(num_samples, size=self.k, replace=False)]
        
        # Create array to track which cluster each sample is assigned to
        self.closest = np.zeros(num_samples)

        for iteration in range(self.iterations):
            # Save current closest cluster values
            old_closest = self.closest.copy()

            # Compute distances from every sample to each of the centroids
            # Distance calculated as L2/Euclidean norms (i.e. sqrt(x^2 + y^2 + ...))
            distances = [self.euclidean_distance(sample, self.centroids) for sample in samples]

            # Save the closest clusters
            self.closest = np.argmin(distances, axis=1)

            # Set new centroids to the mean values of the currently labelled samples (with the same label)
            for idx in range(self.k):
                self.centroids[idx] = (samples[self.closest == idx]).mean(axis=0)

            # if np.linalg.norm(self.closest - old_closest) < self.tol:
            #     break
            
            if np.array_equal(self.closest, old_closest):
                break
