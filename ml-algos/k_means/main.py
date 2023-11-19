from k_means import KMeans
import numpy as np

if __name__ == "__main__":

    features = np.random.rand(1_000, 2)

    kmeans = KMeans(k=4, iterations=16)
    kmeans.fit(features)