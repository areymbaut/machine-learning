import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from collections import Counter
from decision_tree import Node, DecisionTreeRegressor


class ProbabilisticTreeRegressor(DecisionTreeRegressor):
    """
    Decision-tree regressor that can handle targets being "probabilities"
    computed from the log of the odds. This class is a subtle twist on the
    DecisionTreeRegressor class, as previous probabilities need to be passed
    down the tree to alter the output leaf values.
    """

    def __init__(self,
                 min_samples_split: int = 2,
                 max_depth: int = 4,
                 n_features: int = None,
                 *, previous_probabilities: NDArray = None) -> None:
        """
        Args:
        - min_samples_split (int): minimal number of samples required to split
        a node.
        - max_depth (int): maximal depth of the decision tree.
        - n_features (int): number of features to consider when splitting
        a node. If None, all available features will be considered.
        - previous_probabilities (NDArray): previous probabilities
        associated with each sample.
        """
        super().__init__(min_samples_split, max_depth, n_features)
        self.previous_probabilities = previous_probabilities

    def fit(self, X: NDArray, y: NDArray) -> None:
        # Make sure the number of features for the tree
        # does not overshoot the number of features in the data
        # Also set self.n_features to the maximal number of features
        # if it was None
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)

        # Initialize self.previous_probabilities uniformly if None
        if self.previous_probabilities is None:
            n_samples = X.shape[0]
            self.previous_probabilities = np.ones_like(y)/n_samples

        # Grow tree
        self.root = self._grow_tree(X, y, self.previous_probabilities)

    def _grow_tree(self,
                   X: NDArray,
                   y: NDArray,
                   previous_probabilities: NDArray,
                   depth: int = 0) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            # Leaf node
            return Node(
                leaf_value = self._get_leaf_value(y, previous_probabilities))

        # Selecting the features of interest for splitting
        if self.n_features == n_features:
            # Consider all features
            feature_idx = np.arange(n_features)
        else:
            # Randomly select a subset of features without replacement
            feature_idx = np.random.choice(n_features,
                                           self.n_features,
                                           replace=False)
            
        # Find the best split
        best_feature_idx, best_threshold \
            = self._find_best_split(X, y, feature_idx)

        # Create child nodes
        left_idx, right_idx = self._split_samples(X[:, best_feature_idx],
                                                  best_threshold)
        left_node = self._grow_tree(X[left_idx, :],
                                    y[left_idx],
                                    previous_probabilities[left_idx],
                                    depth + 1)
        right_node = self._grow_tree(X[right_idx, :],
                                     y[right_idx],
                                     previous_probabilities[right_idx],
                                     depth + 1)

        return Node(best_feature_idx, best_threshold, left_node, right_node)

    def _get_leaf_value(self, y: NDArray, previous_probabilities: NDArray):
        return np.sum(y)/np.sum(
            [p*(1 - p) for p in previous_probabilities])


class GradientBoostingTreeClassifier():
    """
    Gradient-boosted trees classify based on an initial leaf node
    (i.e., log of the odds of the positive class) and a series of
    subsequent trees that predict the residuals of the previous trees
    in the series. These residuals are computed in terms of the probability
    associated with the log of the odds (i.e., sigmoid(log of the odds)).
    
    The final log-of-the-odd prediction is equal to:
    initial leaf value + learning_rate*sum(tree_predictions).

    The final prediction is obtained by taking the sigmoid of the final
    log-of-the-odd prediction, thresholded at 0.5 to get a binary
    classification output.
    """    
    
    def __init__(self,
                 n_trees: int = 20,
                 min_samples_split: int = 2,
                 max_depth: int = 4,
                 n_features: int = None,
                 learning_rate: float = 0.1) -> None:
        """
        Args:
        - n_trees (int): number of trees making up the set of residual trees.
        - min_samples_split (int): minimal number of samples required to split
        a node.
        - max_depth (int): maximal depth of the decision tree.
        - n_features (int): number of features to consider when splitting
        a node. If None, all available features will be considered.
        - learning_rate (float): learning rate.
        """

        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.learning_rate = learning_rate

        # To later store the initial leaf value
        self.initial_leaf_value: float = None

        # To later contain all the trees
        self.trees: list[ProbabilisticTreeRegressor] = []

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.trees = []

        # Set initial leaf value and compute initial residuals
        self._set_initial_leaf_value(y)
        residuals = self._compute_initial_residuals(y)

        # Initialize previous probabilities
        previous_probabilities \
            = self._sigmoid(self.initial_leaf_value)*np.ones_like(y)
        
        # Generate trees to progressively shrink the residuals
        for _ in range(self.n_trees):
            tree = ProbabilisticTreeRegressor(
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth,
                n_features = self.n_features,
                previous_probabilities = previous_probabilities
            )

            # Train decision tree to fit the residuals and store it
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update residuals and probabilities
            residuals, previous_probabilities \
                = self._update_residuals_and_probabilities(X, y)

    def predict(self, X: NDArray) -> NDArray:
        tree_predictions = self._gather_tree_predictions(X)
        return self._get_predictions_from_trees(tree_predictions)
    
    def _gather_tree_predictions(self, X: NDArray) -> NDArray:
        # This is a list of lists wherein all the sample predictions
        # (second dimension) are listed for each tree (first dimension)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # We need to swap axes to have the samples in the first dimension
        # and the trees in the second dimension
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        return tree_predictions

    def _set_initial_leaf_value(self, y: NDArray) -> None:
        self.initial_leaf_value = self._log_of_the_odds(y)

    def _log_of_the_odds(self, y: NDArray) -> float:
        n = len(y)
        n_1 = np.sum(y == 1)
        n_0 = np.sum(y == 0)

        # Check if we are properly dealing with a binary classification
        if (n_1 == 0):
            raise ValueError('There is no class associated with the label 1')
        elif (n_0 == 0):
            raise ValueError('There is no class associated with the label 0')
        elif (n_0 + n_1 != n):
            raise ValueError('There are more than two classes in the dataset')
        
        return np.log(n_1/n_0)

    def _compute_initial_residuals(self, y: NDArray) -> NDArray:
        # This function assumes that the class labels (y) are 0 and 1,
        # which is checked in _log_of_the_odds, used to compute
        # self.initial_leaf_value

        # "Probability" associated with the log of the odds
        probability = self._sigmoid(self.initial_leaf_value)
        return (y - probability)
    
    def _sigmoid(self, x: NDArray) -> NDArray:
        return 1/(1 + np.exp(-x))
    
    def _update_residuals_and_probabilities(self,
                                            X: NDArray,
                                            y: NDArray
                                            ) -> Tuple[NDArray, NDArray]:
        # Sum residuals through the tree predictions
        residual_predictions = self._gather_tree_predictions(X)
        final_residuals = np.sum(residual_predictions, axis=1)

        # Get predicted log of the odds and convert to probability
        predicted_log_of_the_odds = (self.initial_leaf_value
                                     + self.learning_rate*final_residuals)
        predicted_probabilities = self._sigmoid(predicted_log_of_the_odds)

        residuals = y - predicted_probabilities
        return residuals, predicted_probabilities

    def _get_predictions_from_trees(self,
                                    tree_predictions: NDArray) -> NDArray:
        # Sum residuals through the tree predictions
        final_residuals = np.sum(tree_predictions, axis=1)

        # Get predicted log of the odds and convert to probability
        predicted_log_of_the_odds = (self.initial_leaf_value
                                     + self.learning_rate*final_residuals)
        predicted_probabilities = self._sigmoid(predicted_log_of_the_odds)

        # Return label mode across tree predictions (majority voting)
        return np.array([1 if (p > 0.5) else 0
                         for p in predicted_probabilities])
    

class GradientBoostingTreesRegressor():
    """
    Gradient-boosted trees perform regression based on an initial leaf node
    (i.e., the average target) and a series of subsequent trees that predict
    the residuals of the previous trees in the series. The final prediction
    is equal to:
    initial leaf value + learning_rate*sum(tree_predictions).
    """  

    def __init__(self,
                 n_trees: int = 20,
                 min_samples_split: int = 2,
                 max_depth: int = 4,
                 n_features: int = None,
                 learning_rate: float = 0.1) -> None:
        """
        Args:
        - n_trees (int): number of trees making up the set of residual trees.
        - min_samples_split (int): minimal number of samples required to split
        a node.
        - max_depth (int): maximal depth of the decision tree.
        - n_features (int): number of features to consider when splitting
        a node. If None, all available features will be considered.
        - learning_rate (float): learning rate.
        """

        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.learning_rate = learning_rate

        # To later store the initial leaf value
        self.initial_leaf_value: float = None

        # To later contain all the trees
        self.trees: list[DecisionTreeRegressor] = []

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.trees = []

        # Set initial leaf value and compute initial residuals
        self._set_initial_leaf_value(y)
        residuals = self._compute_initial_residuals(y)
        
        # Generate trees to progressively shrink the residuals
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth,
                n_features = self.n_features
            )

            # Train decision tree to fit the residuals and store it
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update residuals
            residuals = self._update_residuals(X, y)

    def predict(self, X: NDArray) -> NDArray:
        tree_predictions = self._gather_tree_predictions(X)
        return self._get_predictions_from_trees(tree_predictions)
    
    def _set_initial_leaf_value(self, y: NDArray) -> None:
        self.initial_leaf_value = np.mean(y)
    
    def _compute_initial_residuals(self, y: NDArray) -> NDArray:
        return (y - self.initial_leaf_value)
    
    def _update_residuals(self, X: NDArray, y: NDArray) -> NDArray:
        # Sum residuals through the tree predictions
        residual_predictions = self._gather_tree_predictions(X)
        final_residuals = np.sum(residual_predictions, axis=1)

        # Get predicted target values
        predicted_values = (self.initial_leaf_value
                            + self.learning_rate*final_residuals)
        return (y - predicted_values)
    
    def _gather_tree_predictions(self, X: NDArray) -> NDArray:
        # This is a list of lists wherein all the sample predictions
        # (second dimension) are listed for each tree (first dimension)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # We need to swap axes to have the samples in the first dimension
        # and the trees in the second dimension
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        return tree_predictions

    def _get_predictions_from_trees(self,
                                    tree_predictions: NDArray) -> NDArray:
        return (self.initial_leaf_value
                + self.learning_rate*np.sum(tree_predictions, axis=1))


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Classification
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    n_features = X.shape[1]
    n_features_for_splitting = np.floor(np.sqrt(n_features)).astype(int)
    classifier = GradientBoostingTreeClassifier(
        n_trees=3,
        learning_rate=0.1,
        n_features=n_features_for_splitting)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of gradient-boosted classifier = {accuracy:.2f} %')


    # Regression
    # Create a random dataset
    rng = np.random.RandomState(0)
    X = np.sort(5*rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3*(0.5 - rng.rand(16))

    # Fit regression model
    regr_1 = GradientBoostingTreesRegressor(max_depth=2)
    regr_2 = GradientBoostingTreesRegressor(max_depth=5)
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
             label="Max depth = 2", linewidth=1)
    plt.plot(X_test, y_2, color="r",
             label="Max depth = 5", linewidth=1)
    plt.xlabel("Data")
    plt.ylabel("Target")
    plt.title(f"Gradient-Boosted Regression (n_trees = {regr_1.n_trees})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
