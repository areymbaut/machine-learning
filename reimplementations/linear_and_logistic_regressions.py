import numpy as np
from numpy.typing import NDArray


class BaseRegression:
    """
    Given that the linear and logistic regressors are very similar, this class
    contains all the functions that are common to both. The helper functions
    _model and _predict are undefined here, as they are overwritten by those
    found in the LinearRegression and LogisticRegression classes below.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 n_iters: int = 1000) -> None:
        """
        Args:
        - learning_rate (float): learning rate.
        - n_iters (int): maximal number of iterations.
        """

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: NDArray, y: NDArray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = self._model(X, self.weights, self.bias)

            # Gradients
            grad_w = (1/n_samples)*np.dot(X.T, (y_predicted - y))
            grad_b = (1/n_samples)*np.sum(y_predicted - y)

            # Updates
            self.weights -= self.learning_rate*grad_w
            self.bias -= self.learning_rate*grad_b

    def predict(self, X: NDArray) -> NDArray:
        return self._predict(X, self.weights, self.bias)
    
    def _model(self, X: NDArray, weights: NDArray, bias: float):
        raise NotImplementedError
    
    def _predict(self, X: NDArray, weights: NDArray, bias: float):
        raise NotImplementedError


class LinearRegression(BaseRegression):
    def _model(self, X: NDArray, weights: NDArray, bias: float) -> NDArray:
        linear_model = np.dot(X, weights) + bias
        return linear_model
    
    def _predict(self, X: NDArray, weights: NDArray, bias: float) -> NDArray:
        return self._model(X, weights, bias)


class LogisticRegression(BaseRegression):
    def _model(self, X: NDArray, weights: NDArray, bias: float) -> NDArray:
        linear_model = np.dot(X, weights) + bias
        return self._sigmoid(linear_model)
    
    def _predict(self, X: NDArray, weights: NDArray, bias: float) -> NDArray:
        y_predicted = self._model(X, weights, bias)
        return np.array([1 if (p > 0.5) else 0 for p in y_predicted])
    
    def _sigmoid(self, x: NDArray) -> NDArray:
        return 1/(1 + np.exp(-x))


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    np.random.seed(0)

    # Linear regression
    X, y = datasets.make_regression(n_samples=100,
                                    n_features=1,
                                    noise=20,
                                    random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = np.mean((y_test - predictions)**2)
    print(f'Mean squared error of linear regression = {mse:.2f}')

    predicted_line = regressor.predict(X)
    plt.figure()
    plt.scatter(X_train, y_train, c='g', marker='o', edgecolor='k', s=30,
                label='Training data', alpha=0.2)
    plt.scatter(X_test, y_test, c='r', marker='o', edgecolor='k', s=30,
                label='Testing data')
    plt.plot(X, predicted_line, color='k', lw=2, label='Prediction')
    plt.xlabel('Data')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


    # Logistic regression
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of logistic regression = {accuracy:.2f} %')


if __name__ == '__main__':
    main()
