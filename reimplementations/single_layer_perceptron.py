import numpy as np
from numpy.typing import NDArray


class PerceptronClassifier:
    """
    The perceptron is one of the first and one of the simplest types of
    artificial neural networks, as it consists of a single node or neuron.
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
        self.activation_function = self._unit_step_function
        self.weights = None
        self.bias = None

    def _unit_step_function(self, x: NDArray) -> NDArray:
        return np.where(x > 0, 1, 0)

    def fit(self, X: NDArray, y: NDArray) -> None:
        n_features = X.shape[1]

        # Initialize parameters
        self.weights = np.random.rand(n_features)
        self.bias = 0

        y_activated = self.activation_function(y)

        # Learn weights
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Perceptron update (gradient descent)
                gradient_w = (y_predicted - y_activated[idx])*x
                gradient_b = (y_predicted - y_activated[idx])
                self.weights -= self.learning_rate*gradient_w
                self.bias -= self.learning_rate*gradient_b

    def predict(self, X: NDArray) -> NDArray:
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    np.random.seed(0)

    X, y = datasets.make_blobs(n_samples=150,
                               n_features=2,
                               centers=2,
                               cluster_std=1.05,
                               random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    classifier = PerceptronClassifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of single-layer perceptron = {accuracy:.2f} %')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', edgecolors='k',
                c=y_train, alpha=0.2, label='Training data')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='^', edgecolors='k',
                c=y_test, label='Testing data')

    x_min = np.amin(X_train[:, 0])
    x_max = np.amax(X_train[:, 0])
    y_decision_boundary_1 = -(classifier.weights[0]*x_min
                              + classifier.bias)/classifier.weights[1]
    y_decision_boundary_2 = -(classifier.weights[0]*x_max
                              + classifier.bias)/classifier.weights[1]
    
    ax.plot([x_min, x_max], [y_decision_boundary_1, y_decision_boundary_2],
            'k', label='Decision boundary')

    y_min = np.amin(X_train[:, 1])
    y_max = np.amax(X_train[:, 1])

    ax.set_xlim([x_min - 3, x_max + 3])
    ax.set_ylim([y_min - 3, y_max + 3])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Single-layer perceptron')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()