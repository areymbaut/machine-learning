import numpy as np
from typing import Any
from numpy.typing import NDArray


class NaiveBayes:
    """
    Given the class labels y and the feature array X,
    Bayes theorem states that P(y|X) = P(X|y)*P(y)/P(X).

    At inference time, the Naive Bayes classifier assigns to each sample
    the class that maximizes the numerator of the above equation,
    i.e., P(X|y)*P(y). 
     
    The prior probability P(y) is obtained using a frequentist approach
    within the training dataset,
    i.e., P(y) = (number of times y is observed) / (number of samples).

    As for P(X|y), the Naive Bayes classifier is naive in the sense
    that it asssumes features are independent, so that
    P(X|y) = P(x_1|y)*P(x_2|y)*P(x_3|y)*...,
    where x_i is the i-th feature of a given sample.

    Each probability P(x_i|y) is modeled by a Gaussian distribution with mean
    and variance equal to those of x_i within the training samples assigned
    to class y.

    To avoid underflow issues, it is often preferred to maximize
    log(P(X|y)*P(y)) = log(P(X|y)) + log(P(y)) instead of P(X|y)*P(y).

    NB: There is no need for a __init__ function as this classifier
    does not have hyperparameters.
    """

    def fit(self, X: NDArray, y: NDArray) -> None:
        n_samples, n_features = X.shape

        # Unique class labels and number of classes
        self._classes = np.unique(y)
        n_classes: int = len(self._classes)

        # Compute the mean and variance of each feature within
        # each class, and the prior probability of each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for idx_class, c in enumerate(self._classes):
            X_class = X[y == c]
            self._mean[idx_class, :] = X_class.mean(axis=0)
            self._var[idx_class, :] = X_class.var(axis=0)
            self._priors[idx_class] = X_class.shape[0]/float(n_samples)

    def predict(self, X: NDArray) -> NDArray:
        return np.array([self._predict(x) for x in X])

    def _predict(self, x: NDArray) -> int:
        # Calculate the log of the posterior probabilities for each class
        log_posteriors: list[float] = []
        for idx_class in range(len(self._classes)):
            log_prior = np.log(self._priors[idx_class])
            log_posterior = np.sum(np.log(self._gaussian_pdf(x, idx_class)))
            log_posteriors.append(log_posterior + log_prior)

        # Return class with highest posterior probability
        return self._classes[np.argmax(log_posteriors)]

    def _gaussian_pdf(self, x: NDArray, idx_class: int) -> NDArray:
        # Compute Gaussian probability distribution function
        mean = self._mean[idx_class]
        var = self._var[idx_class]
        gaussian_pdf = np.exp(-(x - mean)**2/(2*var))/np.sqrt(2*np.pi*var)
        return gaussian_pdf


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    # Load data
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Infer
    classifier = NaiveBayes()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of Naive Bayes classifier = {accuracy:.2f} %')


if __name__ == '__main__':
    main()
