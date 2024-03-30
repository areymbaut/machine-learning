import numpy as np
from numpy.typing import NDArray
from typing import Union
from collections import Counter


class BaseKNN:
    """
    The k-nearest-neighbor (kNN) algorithm can be used for classification
    or regression purposes. During inference, it identifies for each
    test data point its k nearest neighbors in the training data
    ('nearest' in the sense of a given distance metric, here euclidian).

    As a classifier (see class KNeighborsClassifier below), kNN assigns
    a label corresponding to the mode of the labels found across
    these nearest neighbors (majority voting).

    As a regressor (see class KNeighborsRegressor below), kNN assigns
    the average of the targets found across these nearest neighbors.
    """
    
    def __init__(self, k: int = 5) -> None:
        """
        Arg:
        - k (int): number of nearest neighbors to consider
        for classification or regression.
        """

        self.k = k

    def fit(self, X: NDArray, y: NDArray) -> None:
        # Store the training feature array and training labels
        self.X_train = X
        self.y_train = y

    def predict(self, X: NDArray) -> NDArray:
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x: NDArray) -> Union[int, float]:
        # Compute the distances
        distances = [self._euclidian_distance(x, x_train)
                     for x_train in self.X_train]

        # Get the closest k points in the training dataset
        k_nearest_neighbor_idx = np.argsort(distances)[:self.k]
        k_nearest_neighbor_y = [self.y_train[i]
                                for i in k_nearest_neighbor_idx]
        return self._get_prediction_from_neighbors(k_nearest_neighbor_y)

    def _euclidian_distance(self, x0: NDArray, x1: NDArray) -> NDArray:
        return np.sqrt(np.sum((x0 - x1)**2))
    
    def _get_prediction_from_neighbors(y_neighbors: NDArray):
        raise NotImplementedError


class KNeighborsClassifier(BaseKNN):
    def _get_prediction_from_neighbors(self, y_neighbors: NDArray):
        return Counter(y_neighbors).most_common(1)[0][0]  # Label mode


class KNeighborsRegressor(BaseKNN):
    def _get_prediction_from_neighbors(self, y_neighbors: NDArray):
        return np.mean(y_neighbors)  # Average target


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Classification
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    idx_correct = (predictions == y_test)
    idx_incorrect = np.logical_not(idx_correct)

    accuracy = np.sum(idx_correct)/len(y_test)*100
    print(f'Accuracy of kNN classifier = {accuracy:.2f} %')

    color_list = ['#FF0000', '#00FF00', '#0000FF']  # [RGB]
    cmap_1 = ListedColormap([color_list[i] for i in np.unique(y_train)])
    cmap_2 = ListedColormap(
        [color_list[i] for i in np.unique(predictions[idx_correct])])
    cmap_3 = ListedColormap(
        [color_list[i] for i in np.unique(predictions[idx_incorrect])])
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X[:, 2], X[:, 3], c=y, cmap=ListedColormap(color_list),
               marker='o', edgecolor='k', s=30)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Original data')

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap=cmap_1, alpha=0.2,
                marker='o', edgecolor='k', s=30, label='Training data')
    if np.any(idx_incorrect):
        ax.scatter(X_test[idx_correct, 2], X_test[idx_correct, 3],
                   c=predictions[idx_correct], cmap=cmap_2,
                   marker='^', edgecolor='k', s=30,
                   label='Testing data correctly predicted')
        ax.scatter(X_test[idx_incorrect, 2], X_test[idx_incorrect, 3],
                   c=predictions[idx_incorrect], cmap=cmap_3,
                   marker='s', edgecolor='k', s=30,
                   label='Testing data incorrectly predicted')
    else:
        ax.scatter(X_test[idx_correct, 2], X_test[idx_correct, 3],
                   c=predictions[idx_correct], cmap=cmap_2,
                   marker='^', edgecolor='k', s=30,
                   label='Testing data')
    ax.set_title(f'kNN classifier (k = {classifier.k})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.show()


    # Regression
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 1 * (0.5 - np.random.rand(8))

    regressor = KNeighborsRegressor()
    regressor.fit(X, y)
    y_ = regressor.predict(T)

    plt.subplot(1, 1, 1)
    plt.scatter(X, y, edgecolor="black", c="cornflowerblue",
                label="Training data")
    plt.plot(T, y_, color="g", label="Prediction")
    plt.axis("tight")
    plt.xlabel('Data')
    plt.ylabel('Target')
    plt.legend()
    plt.title(f"kNN regressor (k = {regressor.k})")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
