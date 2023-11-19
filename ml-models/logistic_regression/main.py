from logistic_regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Prepare the data
    data = load_breast_cancer()

    # Train/test split
    train_features, test_features, train_labels, test_labels = train_test_split(
        data.data, data.target, test_size=0.33, random_state=0  
    )

    logistic_regression = LogisticRegression(
        learning_rate=2e-5,
        epochs=256,
        threshold=0.5,
        logging=True,
    )
    logistic_regression.fit(train_features, train_labels)
    # predictions = logistic_regression.predict_labels(test_features) 