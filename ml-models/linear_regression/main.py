from linear_regression import LinearRegression
import numpy as np

if __name__ == "__main__":
    # Training dataset
    train_features = np.arange(0, 250).reshape(-1, 1)
    train_labels = np.arange(0, 500, 2)

    # Testing dataset
    test_features = np.arange(300, 400, 8).reshape(-1, 1)
    test_labels = np.arange(600, 800, 16)

    linear_regression = LinearRegression(epochs=20, learning_rate=1e-5, logging=True)
    linear_regression.fit(train_features, train_labels)
    predictions = linear_regression.predict(test_features).round()

