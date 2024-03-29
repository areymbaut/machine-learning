import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Fully connected neural network with one hidden layer.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_classes: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x  # No activation or softmax at the end


def main():
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    # (28*28 images of handwritten digits, from 0 to 9, so 10 classes)
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)

    # Data loader
    batch_size = 100
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    
    # Show some examples from the test dataset
    example_data = next(iter(test_loader))[0]
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap='gray')
        plt.tick_params(axis='both',
                        which='both',
                        left=False,
                        labelleft=False,
                        bottom=False,
                        labelbottom=False)
    plt.show()

    # Define model
    input_size = 28*28  # 28*28 images
    num_classes = 10  # 10 types of digits
    hidden_size = 500
    model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

    # Training
    num_epochs = 3
    learning_rate = 0.001

    cross_entropy_loss = nn.CrossEntropyLoss()  # Multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    n_training_samples = len(train_loader.dataset)
    print(f'Training over {n_training_samples} images...')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # Forward pass and loss calculation
            outputs = model(images)
            loss = cross_entropy_loss(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}]: '
              + f'loss = {running_loss/n_training_samples:.4f}')

    # Inference
    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(images)

            # max returns (output_value, index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        accuracy = n_correct/n_samples*100
        print(f'Accuracy of the neural network on the {n_samples} '
              + f'test images: {accuracy:.2f} %')


if __name__ == '__main__':
    main()
