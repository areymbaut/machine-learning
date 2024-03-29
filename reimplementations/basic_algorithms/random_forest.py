import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from collections import Counter
from decision_tree import (BaseDecisionTree,
                           DecisionTreeClassifier,
                           DecisionTreeRegressor)


class BaseRandomForest:
    """
    The random-forest algorithm is an ensemble technique that takes advantage
    of bagging (bootstrap aggregating), as it generates a set of weak learners
    from subsets of the data obtained via bootstrapping with replacement.
    There is, however, a twist to this concept of bagging for the random-forest
    algorithm, as it also selects a random set of features to choose from
    when splitting nodes.

    The idea behind this algorithm is the reduction of variance. When pooling
    the predictions from multiple (weak) learners, a reduction of variance is
    expected, yet limited by the amount of correlation present between the
    different predictions. By adding the random selection of splitting
    features, the random forest aims to decorrelate these predictions
    to further reduce variance.
    """
    
    def __init__(self,
                 n_trees: int = 20,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int = None) -> None:
        """
        Args:
        - n_trees (int): number of trees making up the random forest.
        - min_samples_split (int): minimal number of samples required to split
        a node.
        - max_depth (int): maximal depth of the decision tree.
        - n_features (int): number of features to consider when splitting
        a node. If None, all available features will be considered.
        """

        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

        # To later contain all the trees
        self.trees: list[BaseDecisionTree] = []

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.trees = []
        for _ in range(self.n_trees):
            tree = self._get_tree()

            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self,
                           X: NDArray,
                           y: NDArray) -> Tuple[NDArray, NDArray]:
        # Perform bootstrap with replacement
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def predict(self, X: NDArray) -> NDArray:
        # This is a list of lists wherein all the sample predictions
        # (second dimension) are listed for each tree (first dimension)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # We need to swap axes to have the samples in the first dimension
        # and the trees in the second dimension
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)

        return self._get_predictions_from_trees(tree_predictions)
    
    def _get_tree(self) -> BaseDecisionTree:
        raise NotImplementedError

    def _get_predictions_from_trees(self,
                                    tree_predictions: NDArray) -> NDArray:
        raise NotImplementedError


class RandomForestClassifier(BaseRandomForest):
    def __init__(self,
                 n_trees: int = 20,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int = None) -> None:
        super().__init__(n_trees, min_samples_split, max_depth, n_features)
        self.trees: list[DecisionTreeClassifier] = []

    def _get_tree(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            min_samples_split = self.min_samples_split,
            max_depth = self.max_depth,
            n_features = self.n_features
            )

    def _get_predictions_from_trees(self,
                                    tree_predictions: NDArray) -> NDArray:
        # Return label mode across tree predictions (majority voting)
        return np.array([Counter(t_pred).most_common(1)[0][0]
                         for t_pred in tree_predictions])
    

class RandomForestRegressor(BaseRandomForest):
    def __init__(self,
                 n_trees: int = 20,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int = None) -> None:
        super().__init__(n_trees, min_samples_split, max_depth, n_features)
        self.trees: list[DecisionTreeRegressor] = []

    def _get_tree(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            min_samples_split = self.min_samples_split,
            max_depth = self.max_depth,
            n_features = self.n_features
            )

    def _get_predictions_from_trees(self,
                                    tree_predictions: NDArray) -> NDArray:
        # Return average target across tree predictions
        return np.array([np.mean(t_pred) for t_pred in tree_predictions])


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    np.random.seed(0)

    # Classification
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    n_features = X.shape[1]
    n_features_for_splitting = np.floor(np.sqrt(n_features)).astype(int)
    classifier = RandomForestClassifier(
        n_features = n_features_for_splitting
        )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of random-forest classifier = {accuracy:.2f} %')


    # Regression
    # Create a random dataset
    rng = np.random.RandomState(0)
    X = np.sort(5*rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3*(0.5 - rng.rand(16))

    # Fit regression model
    max_depth_1 = 2
    max_depth_2 = 5
    regr_1 = RandomForestRegressor(max_depth=max_depth_1)
    regr_2 = RandomForestRegressor(max_depth=max_depth_2)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="cornflowerblue")
    plt.plot(X_test, y_1, color="g",
             label="Max depth = " + str(max_depth_1), linewidth=1)
    plt.plot(X_test, y_2, color="r",
             label="Max depth = " + str(max_depth_2), linewidth=1)
    plt.xlabel("Data")
    plt.ylabel("Target")
    plt.title(f"Random Forest Regression (n_trees = {regr_1.n_trees})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
