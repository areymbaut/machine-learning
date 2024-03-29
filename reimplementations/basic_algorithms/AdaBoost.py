import numpy as np
from numpy.typing import NDArray
from typing import Union


class DecisionStumpClassifier:
    """
    Instead of using full decision trees, one can define so-called decision
    "stumps" for AdaBoost, i.e., decision trees with a root node and
    two leaf nodes.

    The decision stump can be parametrized by the following class attributes:
    - feature_idx (int): index of the feature that is split by the stump.
    - threshold (float | int): threshold that splits the feature.
    - polarity (int): 1 (the stump assigns the +1 class to the right leaf node)
    or -1 (the stump assigns the +1 class to the left leaf node).
    - performance (float): measure that corresponds to how well the stump
    classified the data, used to update the sample weights and to obtain
    the final AdaBoost predictions.
    """

    def __init__(self) -> None:
        self.feature_idx: int = None
        self.threshold: Union[int, float] = None
        self.polarity: int = 1
        self.performance: float = None

    def predict(self, X: NDArray) -> NDArray:
        n_samples = X.shape[0]
        X_feature = X[:, self.feature_idx]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            # Stump has -1 class in the left leaf node
            predictions[X_feature < self.threshold] = -1
        else:
            # Stump has -1 class in the right leaf node
            predictions[X_feature >= self.threshold] = -1

        return predictions


class AdaBoostClassifier:
    """
    The Adaptive Boosting (AdaBoost) algorithm takes advantage of boosting,
    sequentially training weak learners (here, tree stumps) with each
    subsequent learner attempting to correct for the deficiencies of
    previous learner.

    This is done by modifying the weights assigned to data samples in the
    computation of missclassifications, more strongly penalizing data samples
    that were missclassified by the previous learner.
    """
    
    def __init__(self, n_stumps: int = 5) -> None:
        """
        Arg:
        - n_stumps (int): number of stumps to be generated.
        """

        self.n_stumps: int = n_stumps
        self.stumps: list[DecisionStumpClassifier] = []

    def fit(self, X: NDArray, y: NDArray) -> None:
        n_samples, n_features = X.shape

        # Make sure binary labels are -1 and 1
        y_modified = np.where(y <= 0, -1, 1)

        # Initialize weights uniformly
        weights = np.ones(n_samples)/n_samples

        # Generate the stumps
        for _ in range(self.n_stumps):
            stump = DecisionStumpClassifier()

            min_error = float('inf')  # Very large number
            for feature_idx in range(n_features):
                X_feature = X[:, feature_idx]
                thresholds = np.unique(X_feature)
                for thr in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_feature < thr] = -1

                    # Compute missclassification error
                    missclassified_weights = weights[y_modified != predictions]
                    error = np.sum(missclassified_weights)

                    if error > 0.5:
                        # Flip error and decision
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        min_error = error
                        stump.feature_idx = feature_idx
                        stump.threshold = thr
                        stump.polarity = polarity

            # Performance of the stump
            stump.performance = self._get_stump_performance(min_error)

            # Update weights
            predictions = stump.predict(X)
            update_exponent = stump.performance*y_modified*predictions
            weights *= np.exp(-update_exponent)/np.sum(weights)

            # Append stump
            self.stumps.append(stump)

    def _get_stump_performance(self, missclassification_error: float) -> float:
        EPS = 1e-10  # Very small number to avoid definition issues
        ratio = (1 - missclassification_error)/(missclassification_error + EPS)
        return 0.5*np.log(ratio)

    def predict(self, X: NDArray) -> NDArray:
        stump_predictions = [stump.performance*stump.predict(X)
                             for stump in self.stumps]
        return np.sign(np.sum(stump_predictions, axis=0))


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0)

    classifier = AdaBoostClassifier(n_stumps=20)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of AdaBoost = {accuracy:.2f} %')


if __name__ == '__main__':
    main()
