import numpy as np
from numpy.typing import NDArray
from typing import Union, List, Tuple
from collections import Counter


class Node:
    def __init__(self,
                 feature_idx: int = None,
                 threshold: Union[float, int] = None,
                 left_node = None,
                 right_node = None,
                 *, leaf_value: Union[float, int] = None) -> None:
        """
        Args:
        - feature_idx (int): index of the feature being split in the node.
        - threshold (float | int): threshold value splitting the node.
        - left_node (Node): left node associated with feature < threshold.
        - right_node (Node): right node associated with feature >= threshold.
        - value (float | int): value of the node, if the node is a leaf node.

        NB:
        - The arguments left_node and right_node are Node objects (even though
        they appear in the Node class definition) because the decision tree
        is grown recursively in the DecisionTree class (see _grow_tree).
        - Given the asterisk before the value argument, this argument has to
        be provided by name, making it obvious when we are dealing with
        a leaf node.
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.leaf_value = leaf_value

    def _is_leaf_node(self):
        return (self.leaf_value is not None)


class BaseDecisionTree:
    """
    A decision tree is a hierarchical set of nodes that can be used for
    classification (see class DecisionTreeClassifer below) or
    regression (see class DecisionTreeRegressor below) purposes. 
    
    During training, a binary decision is made at each node, splitting samples
    based on whether one of their features is lesser or greater than a given
    threshold. The node features and thresholds are determined at each depth
    of the tree to provide the "best" split (i.e., that which reduces the
    child nodes' impurity):
    - for classification purposes, the best split is the one associated with
    the largest information gain between the parent node and the
    child nodes. 
    - for regression purposes, the best split is the one associated with the
    lowest sum of squared residuals across the child nodes (a tree can be
    seen as a piecewise constant approximation).
    
    The final nodes of the tree, called leaf nodes, provide the result
    of the inference:
    - for classification purposes, a leaf node is associated with
    the mode of the sample labels within it.
    - for regression purposes, a leaf node is associated with
    the average of the sample targets within it.

    At inference time, new samples are simply run through the decision tree.
    """

    def __init__(self,
                 min_samples_split: int = 2,
                 max_depth: int = 100,
                 n_features: int = None) -> None:
        """
        Args:
        - min_samples_split (int): minimal number of samples required to split
        a node.
        - max_depth (int): maximal depth of the decision tree.
        - n_features (int): number of features to consider when splitting
        a node. If None, all available features will be considered
        (typical for a decision tree, less so when building a random forest).
        """

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None  # To later store the full tree from the root node

    def fit(self, X: NDArray, y: NDArray) -> None:
        # Make sure the number of features for the tree
        # does not overshoot the number of features in the data
        # Also set self.n_features to the maximal number of features
        # if it was None
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)

        # Grow tree
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: NDArray, y: NDArray, depth: int = 0) -> Node:
        """
        Function that recursively grows a regression tree.
        """

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            # Leaf node
            return Node(leaf_value = self._get_leaf_value(y))

        # Selecting the features of interest for splitting
        if self.n_features == n_features:
            # Consider all features
            feature_idx = np.arange(n_features)
        else:
            # Randomly select a subset of features without replacement
            # (useful when building random forests from this class)
            feature_idx = np.random.choice(n_features,
                                           self.n_features,
                                           replace=False)
            
        # Find the best split
        best_feature_idx, best_threshold \
            = self._find_best_split(X, y, feature_idx)

        # Create child nodes
        left_idx, right_idx = self._split_samples(X[:, best_feature_idx],
                                                  best_threshold)
        left_node = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_node = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feature_idx, best_threshold, left_node, right_node)

    def _get_leaf_value(self, y: NDArray):
        raise NotImplementedError

    def _find_best_split(self,
                         X: NDArray,
                         y: NDArray,
                         feature_idx: List[int]
                         ) -> Tuple[int, Union[float, int]]:
        raise NotImplementedError

    def _information_gain(self,
                          X_feature: NDArray,
                          y: NDArray,
                          split_threshold: Union[float, int]) -> float:
        raise NotImplementedError

    def _entropy(self, y: NDArray) -> float:
        raise NotImplementedError

    def _regression_loss(self,
                         X_feature: NDArray,
                         y: NDArray,
                         split_threshold: Union[float, int]
                         ) -> float:
        raise NotImplementedError

    def _split_samples(self,
                       X_feature: NDArray,
                       split_threshold: Union[float, int]
                       ) -> Tuple[List[int], List[int]]:
        """
        Find indices of the samples going into the left and right nodes.
        """

        left_idx = np.argwhere(X_feature < split_threshold).flatten()
        right_idx = np.argwhere(X_feature >= split_threshold).flatten()
        return left_idx, right_idx

    def predict(self, X: NDArray) -> NDArray:
        # Traverse the tree to predict
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: NDArray, node: Node) -> Union[float, int]:
        """
        Recursive function that traverses the decision tree
        for a given sample until a leaf node is reached.
        """

        if node._is_leaf_node():
            return node.leaf_value
        
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left_node)
        else:
            return self._traverse_tree(x, node.right_node)


class DecisionTreeClassifier(BaseDecisionTree):
    def _get_leaf_value(self, y: NDArray):
        return Counter(y).most_common(1)[0][0]  # Label mode
        
    def _find_best_split(self,
                         X: NDArray,
                         y: NDArray,
                         feature_idx: List[int]
                         ) -> Tuple[int, int]:
        # Initialize
        split_feature_idx, split_threshold = None, None

        # Find splitting feature and threshold
        best_gain = -1
        for f_idx in feature_idx:
            X_feature = X[:, f_idx]

            # Get thresholds as unique values of X_feature
            thresholds = np.unique(X_feature)

            for thr in thresholds:
                # Calculate the information gain
                gain = self._information_gain(X_feature, y, thr)

                # Store information if the information gain has increased
                if gain > best_gain:
                    best_gain = gain
                    split_feature_idx = f_idx
                    split_threshold = thr

        return split_feature_idx, split_threshold

    def _information_gain(self,
                          X_feature: NDArray,
                          y: NDArray,
                          split_threshold: Union[float, int]) -> float:
        # Create children
        left_idx, right_idx = self._split_samples(X_feature, split_threshold)

        # Check if there is actually a split
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # Parent entropy
        parent_entropy = self._entropy(y)

        # Calculate the weighted entropy of the children
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)
        entropy_left = self._entropy(y[left_idx])
        entropy_right = self._entropy(y[right_idx]) 
        children_entropy = (n_left*entropy_left + n_right*entropy_right)/n

        # Calculate information gain
        information_gain = parent_entropy - children_entropy
        return information_gain

    def _entropy(self, y: NDArray) -> float:
        # Compute information entropy
        probabilities = np.bincount(y)/len(y)
        entropy = -np.sum([p*np.log2(p) for p in probabilities if p != 0])
        return entropy


class DecisionTreeRegressor(BaseDecisionTree):
    def _get_leaf_value(self, y: NDArray):
        return np.mean(y)  # Average target
    
    def _find_best_split(self,
                         X: NDArray,
                         y: NDArray,
                         feature_idx: List[int]
                         ) -> Tuple[int, Union[float, int]]:
        # Initialize
        split_feature_idx, split_threshold = None, None

        # Find splitting feature and threshold
        lowest_loss = float('inf')  # Very large number
        for f_idx in feature_idx:
            X_feature = X[:, f_idx]

            # Get thresholds as middle points between
            # the unique values of X_feature 
            thresholds = np.unique(X_feature)
            diff_thresholds = np.diff(thresholds)
            thresholds = thresholds[:-1] + diff_thresholds/2.

            for thr in thresholds:
                # Calculate loss
                loss = self._regression_loss(X_feature, y, thr)

                # Store information if the loss has decreased
                if loss < lowest_loss:
                    lowest_loss = loss
                    split_feature_idx = f_idx
                    split_threshold = thr

        return split_feature_idx, split_threshold
    
    def _regression_loss(self,
                         X_feature: NDArray,
                         y: NDArray,
                         split_threshold: Union[float, int]
                         ) -> float:
        # Create children
        left_idx, right_idx = self._split_samples(X_feature, split_threshold)

        # Check if there is actually a split
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # Compute sum of squared residuals
        sum_squared_residuals_left \
            = np.sum((y[left_idx] - np.mean(y[left_idx]))**2)
        sum_squared_residuals_right \
            = np.sum((y[right_idx] - np.mean(y[right_idx]))**2)
        sum_squared_residuals \
            = sum_squared_residuals_left + sum_squared_residuals_right

        return sum_squared_residuals


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Classifier
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    classifier = DecisionTreeClassifier(max_depth=10)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of decision tree classifier = {accuracy:.2f} %')


    # Regression
    # Create a random dataset
    rng = np.random.RandomState(0)
    X = np.sort(5*rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3*(0.5 - rng.rand(16))

    # Fit regression model
    max_depth_1 = 2
    max_depth_2 = 5
    regr_1 = DecisionTreeRegressor(max_depth=max_depth_1)
    regr_2 = DecisionTreeRegressor(max_depth=max_depth_2)
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
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
