import numpy as np
from numpy.typing import NDArray


class LinearSVM:
    """
    The support vector machine (SVM) algorithm aims to identify a hyperplane
    that distinguishably teases apart data points belonging to different
    classes. The hyperplane is localized in such a manner that the
    largest margin separates the classes under consideration.

    THe coordinates x of the hyperplane satisfy dot(x, weights) + bias = 0,
    and the associated margins satisfy dot(x, weights) + bias = (+1 or -1).
    The width of the margin is given by 2/norm(weights).

    These margins are such that the samples x_i of the class y_i = 1 satisfy
    dot(x, weights) + bias >= 1 and the samples x_i of the class y_i = -1
    satisfy dot(x, weights) + bias <= -1. These two conditions can be combined
    as y_i*(dot(x, weights) + bias) >= 1.

    The above justifies the use of the so-called Hinge loss function:
    l = max(0, 1-y_i*(dot(x, weights) + bias)).

    Besides, maximizing the margin width can be achieved by adding
    an L2 (Ridge) regularization term to the loss function minimization,
    so as to minimize the norm of the weights.
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 lambda_regularization: float = 0.01,
                 n_iters: int = 1000) -> None:
        """
        Args:
        - learning_rate (float): learning rate.
        - lambda_regularization (float): parameter setting the strength of
        the L2 (Ridge) regularization.
        - n_iters (int): maximal number of iterations.
        """

        self.learning_rate = learning_rate
        self.lambda_regularization = lambda_regularization
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: NDArray, y: NDArray) -> None:
        n_features = X.shape[1]

        # Make sure binary labels are -1 and 1 
        y_modified = np.where(y <= 0, -1, 1)

        # Initialize parameters
        self.weights = np.random.rand(n_features)
        self.bias = 0

        # Learn weights and bias
        regularization_term = 2*self.lambda_regularization*self.weights
        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                # Update weights and bias according to appropriate condition 
                condition = (
                    y_modified[idx]*(np.dot(x, self.weights) + self.bias) >= 1)

                if condition:
                    grad_w = regularization_term
                    grad_b = 0
                else:
                    grad_w = regularization_term - np.dot(x, y_modified[idx])
                    grad_b = -y_modified[idx]

                self.weights -= self.learning_rate*grad_w
                self.bias -= self.learning_rate*grad_b

    def predict(self, X: NDArray) -> NDArray:
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)  # Labels that are either -1 or 1


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    np.random.seed(0)

    # Load data
    X, y = datasets.make_blobs(n_samples=100,
                               n_features=2,
                               centers=2,
                               cluster_std=1.05,
                               random_state=40)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Infer
    classifier = LinearSVM(n_iters=1500)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = np.sum(predictions == y_test)/len(y_test)*100
    print(f'Accuracy of linear SVM = {accuracy} %')

    # Visualize
    cmap = ListedColormap(['#FF0000', '#0000FF'])  # [RB]

    def get_hyperplane_coord(x: float,
                             weights: NDArray,
                             bias: float,
                             offset: float):
        return (-weights[0]*x - bias + offset)/weights[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", edgecolors='k',
                c=y_train, cmap=cmap, alpha=0.2, label='Training data')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="^", edgecolors='k',
                c=y_test, cmap=cmap, label='Testing data')

    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])

    w = classifier.weights
    b = classifier.bias

    y_decision_boundary_1 = get_hyperplane_coord(x_min, w, b, 0)
    y_decision_boundary_2 = get_hyperplane_coord(x_max, w, b, 0)

    y_decision_boundary_1_m = get_hyperplane_coord(x_min, w, b, -1)
    y_decision_boundary_2_m = get_hyperplane_coord(x_max, w, b, -1)

    y_decision_boundary_1_p = get_hyperplane_coord(x_min, w, b, 1)
    y_decision_boundary_2_p = get_hyperplane_coord(x_max, w, b, 1)

    ax.plot([x_min, x_max],
            [y_decision_boundary_1, y_decision_boundary_2], "k-",
            label='Decision boundary')
    ax.plot([x_min, x_max],
            [y_decision_boundary_1_m, y_decision_boundary_2_m], "k--",
            label='Decision margins')
    ax.plot([x_min, x_max],
            [y_decision_boundary_1_p, y_decision_boundary_2_p], "k--")

    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])
    ax.set_ylim([y_min - 3, y_max + 3])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Support Vector Machine')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
