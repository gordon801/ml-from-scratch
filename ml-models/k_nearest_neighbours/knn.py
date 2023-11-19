import numpy as np

class KNN:
    def __init__(self, features: np.ndarray, labels: np.ndarray, k: int):
        self.features = features
        self.labels = labels
        self.k = k

    def euclidean_distance(self, pt_1: np.ndarray, pt_2:np.ndarray):
        return np.sqrt(np.sum((pt_1 - pt_2)**2))
    
    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Performs inference using the given features
        """
        num_samples, num_features = samples.shape

        # Initialise array for predictions. We update the indices of this array for efficiency (instead of appending).
        predictions = np.empty(num_samples)

        # For each sample in our (test) sample set
        for idx, test_sample in enumerate(samples):
            # We calculate the distance between this sample and each of the samples in our training dataset
            distances = [self.euclidean_distance(test_sample, train_sample) for train_sample in self.features]

            # Sort the distances to get the indices of the k closest points
            k_sorted_idxs = np.argsort(distances)[:self.k]

            # Use bincount to count the frequency of each label, and argmax to get the (index) value of the most frequent label in these k points.
            most_common_label = np.bincount([self.labels[idx] for idx in k_sorted_idxs]).argmax()

            # Set the label of the current prediction to this most common label
            predictions[idx] = most_common_label
        
        return predictions