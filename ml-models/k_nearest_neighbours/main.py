from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from knn import KNN

if __name__ == "__main__":
    iris = datasets.load_iris()

    train_features, test_features, train_labels, test_labels = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=0  
    )

    knn = KNN(train_features, train_labels, k=3)
    predictions = knn.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {recall:.3f}")
    print(f"Recall:    {precision:.3f}")
    print(f"F-score:   {fscore:.3f}")