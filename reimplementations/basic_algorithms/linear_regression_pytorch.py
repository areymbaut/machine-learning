import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, n_features: int) -> None:
        """
        Based on torch.nn.Linear, this class performs linear regression
        for an input containing n_features features.
        """
        super().__init__()
        # Define layer
        self.lin = nn.Linear(in_features=n_features,
                             out_features=n_features)

    def forward(self, x: torch.Tensor):
        # Apply layer
        return self.lin(x)


def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Linear regression
    X, y = datasets.make_regression(n_samples=100,
                                    n_features=1,
                                    noise=20,
                                    random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Convert from numpy to torch
    X_torch = torch.tensor(X, dtype=torch.float32)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Define model
    n_features = X_train.shape[1]
    model = LinearRegression(n_features=n_features)

    # Training
    learning_rate = 0.05
    n_epochs = 100

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        # Predict = forward pass
        y_pred = model(X_train_torch)

        # Compute loss
        l = loss(y_train_torch, y_pred)
        
        # Backward pass - Backpropagation based on gradient computations
        l.backward()

        # Update weights
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        if (epoch + 1)%10 == 0:
            w, b = model.parameters()
            print(f'Epoch {epoch+1}: '
                  + f'w = {w[0][0].item()}, b = {b.item()}, loss = {l.item()}')

    # Inference
    with torch.no_grad():
        y_pred = model(X_test_torch).numpy()

        mse = np.mean((y_test - y_pred)**2)
        print(f'Mean squared error of linear regression = {mse:.2f}')

        predicted_line = model(X_torch).numpy()
        plt.figure()
        plt.scatter(X_train, y_train, c='g', marker='o', edgecolor='k', s=30,
                    label='Training data', alpha=0.2)
        plt.scatter(X_test, y_test, c='r', marker='o', edgecolor='k', s=30,
                    label='Testing data')
        plt.plot(X, predicted_line, color='k', lw=2, label='Prediction')
        plt.xlabel('Data')
        plt.ylabel('Target')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
