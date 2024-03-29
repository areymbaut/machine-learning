import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Convolutional neural network.
    """
    def __init__(self,
                 input_size: int,
                 pixel_size: int,
                 num_classes: int) -> None:
        super().__init__()

        # Hyperparameters
        conv_kernel_size = 3
        pool_size = 2

        hidden_size_1 = pixel_size
        hidden_size_2 = 2*hidden_size_1

        # IMPORTANT
        # We will keep track of output_size below, to know which input size
        # input_size_linear to set for the first classification layer self.fc1
        output_size = pixel_size
        
        # CONV 1
        self.conv_1 = nn.Conv2d(input_size, hidden_size_1, conv_kernel_size)
        output_size -= (conv_kernel_size - 1)

        # POOL 1
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        output_size \
            = np.floor((output_size - pool_size)/pool_size + 1).astype(int)
        
        # CONV 2
        self.conv_2 = nn.Conv2d(hidden_size_1, hidden_size_2, conv_kernel_size)
        output_size -= (conv_kernel_size - 1)

        # POOL 2
        output_size \
            = np.floor((output_size - pool_size)/pool_size + 1).astype(int)

        # CONV 3
        self.conv_3 = nn.Conv2d(hidden_size_2, hidden_size_2, conv_kernel_size)
        output_size -= (conv_kernel_size - 1)
        
        # Fully connected layers for classification
        input_size_linear = hidden_size_2*output_size*output_size
        self.fc_1 = nn.Linear(input_size_linear, hidden_size_2)
        self.fc_2 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x: torch.Tensor):
        # x has size (N=batch_size, 3, 32, 32)
        # (Output sizes indicating below assuming
        #  kernel_size = 3 and pool_size = 2)
        x = F.relu(self.conv_1(x))  # (N, 32, 30, 30)
        x = self.pool(x)            # (N, 32, 15, 15)
        x = F.relu(self.conv_2(x))  # (N, 64, 13, 13)
        x = self.pool(x)            # (N, 64, 6, 6)
        x = F.relu(self.conv_3(x))  # (N, 64, 4, 4)
        x = torch.flatten(x, 1)     # (N, 1024)
        x = F.relu(self.fc_1(x))    # (N, 64)
        x = self.fc_2(x)            # (N, num_classes)
        return x


def main():
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset has PILImage images of range [0, 1]
    # We transform them to tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # CIFAR10:
    # 60_000 32x32 color images in 10 classes, with 6000 images per class
    input_size = 3  # Color channels in the images
    pixel_size = 32  # Images are 32x32 pixels
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                transform=transform,
                                                download=True)

    # Data loader
    batch_size = pixel_size
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    
    # Classes to consider
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    
    # Show one batch of random training images
    def show_images(imgs: torch.Tensor):
        imgs = imgs/2 + 0.5  # Unnormalize images
        plt.imshow(np.transpose(imgs.numpy(), (1, 2, 0)))
        plt.tick_params(axis='both',
                        which='both',
                        left=False,
                        labelleft=False,
                        bottom=False,
                        labelbottom=False)
        plt.show()

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
    show_images(img_grid)

    # Define model
    model = ConvolutionalNeuralNetwork(input_size,
                                       pixel_size,
                                       num_classes).to(device)

    # Training
    num_epochs = 10
    learning_rate = 0.001

    cross_entropy_loss = nn.CrossEntropyLoss()  # Multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_training_samples = len(train_loader.dataset)
    print(f'Training over {n_training_samples} images...')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
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
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # max returns (output_value, index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        accuracy = n_correct/n_samples*100
        print('Accuracy of the convolutional neural network '
              + f'on the {n_samples} test images: {accuracy:.2f} %')


if __name__ == '__main__':
    main()
