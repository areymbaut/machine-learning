import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BasicLSTM(nn.Module):
    """
    Basic Long Short-Term Memory neural network.
    """
    def __init__(self) -> None:
        super().__init__()

        # Weights are initialized as normally-distributed values
        # with zero mean and unit standard deviation
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # Parameters involved in the forget gate
        self.w_fg0 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.w_fg1 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.b_fg0 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # Parameters involved in the input gate
        self.w_ig0 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.w_ig1 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.b_ig0 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # Parameters involved in the input node
        self.w_in0 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.w_in1 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.b_in0 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # Parameters involved in the output gate
        self.w_og0 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.w_og1 = nn.Parameter(torch.normal(mean=mean, std=std),
                                  requires_grad=True)
        self.b_og0 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def lstm_unit(self, input, long_memory, short_memory):
        # Percentage of the long memory that is remembered,
        # as computed within the forget gate
        long_memory_remember_percent \
            = torch.sigmoid(short_memory*self.w_fg0
                            + input*self.w_fg1
                            + self.b_fg0)
        
        # Potential long memory, as computed within the input node
        potential_long_memory = torch.tanh(short_memory*self.w_in0
                                           + input*self.w_in1
                                           + self.b_in0)

        # Percentage of the potential long memory that is remembered,
        # as computed within the input gate
        potential_long_memory_remember_percent \
            = torch.sigmoid(short_memory*self.w_ig0
                            + input*self.w_ig1
                            + self.b_ig0)
        
        # Updated long memory
        updated_long_memory \
            = (long_memory*long_memory_remember_percent
               + potential_long_memory*potential_long_memory_remember_percent)

        # Potential short memory
        potential_short_memory = torch.tanh(updated_long_memory)

        # Percentage of the potential short memory that is remembered,
        # as computed within the output gate
        potential_short_memory_remember_percent \
            = torch.sigmoid(short_memory*self.w_og0
                            + input*self.w_og1
                            + self.b_og0)
        
        # Updated short memory
        updated_short_memory \
            = potential_short_memory*potential_short_memory_remember_percent

        return updated_long_memory, updated_short_memory

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input should be of size (n_samples, n_sequence) 
        short_memory_samples = torch.empty((input.shape[0], 1),
                                           dtype=torch.float32)
        for i, sequence in enumerate(input):
            long_memory = torch.tensor(0.)
            short_memory = torch.tensor(0.)
            for seq in sequence:
                long_memory, short_memory = self.lstm_unit(seq,
                                                           long_memory,
                                                           short_memory)
            short_memory_samples[i][0] = short_memory
        return short_memory_samples


from torch.utils.data import Dataset
class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


def main():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    # Seeds
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    duration = 2*360
    time = np.arange(duration)
    signal = ((np.sin(3/2*time*np.pi/180)
               + 0.05*np.random.randn(duration))*np.exp(-time/(duration/2))
              + (0.5/duration)*time)

    # Prepare data for LSTM
    # (Model is trained over time_window to predict the next time point)
    X = []
    y = []
    time_window = 10
    for i in range(duration - time_window):
        X.append(signal[i:(i + time_window)])
        y.append(signal[i + time_window])
    X = np.array(X)
    y = np.array(y)

    train_size = 0.6
    idx_split = np.floor(train_size*len(y)).astype(int)
    idx_test = idx_split + time_window
    X_train = X[:idx_split]
    y_train = y[:idx_split]
    X_test = X[idx_split:]
    y_test = y[idx_split:]

    train_dataset = timeseries(X_train, y_train)
    test_dataset = timeseries(X_test, y_test)

    batch_size = 256
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    plt.figure()
    y_min = -1.1
    y_max = 1.1
    plt.axhline(y=0, ls='-', color='k', lw=0.5)
    plt.plot(time, signal, '-b', label = 'Training data')
    plt.plot(time[idx_test:], signal[idx_test:], '-g', label = 'Testing data')
    plt.xlim((time[0], time[-1]))
    plt.ylim((y_min, y_max))
    plt.xlabel('Data')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

    # Define model
    model = BasicLSTM().to(device)

    # Training
    num_epochs = 100
    learning_rate = 0.01

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    n_training_samples = len(train_loader.dataset)
    print(f'Training over {n_training_samples} samples...')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, targets in train_loader:
            signals = signals.to(device)
            targets = targets.view(-1, 1).to(device)

            # Forward pass and loss calculation
            outputs = model(signals)
            loss = mse_loss(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        if (epoch+1)%10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]: '
                  + f'loss = {running_loss/n_training_samples:.6f}')
            
            # Visualize training progress on testing data
            with torch.no_grad():
                outputs = model(test_dataset.x.to(device))
                targets = test_dataset.y.view(-1, 1).to(device)

                plt.figure()
                y_min = -1.1
                y_max = 1.1
                plt.axhline(y=0, ls='-', color='k', lw=0.5)
                plt.plot(time, signal, '-b', label = 'Training data')
                plt.plot(time[idx_test:], signal[idx_test:],
                        '-g', label = 'Testing data')
                plt.plot(time[idx_test:], outputs.numpy(),
                        '-r', label = 'Prediction')
                plt.xlim((time[0], time[-1]))
                plt.ylim((y_min, y_max))
                plt.legend()
                plt.show()

    # Inference
    with torch.no_grad():
        outputs = model(test_dataset.x.to(device))

        plt.figure()
        y_min = -1.1
        y_max = 1.1
        plt.axhline(y=0, ls='-', color='k', lw=0.5)
        plt.plot(time, signal, '-b', label = 'Training data')
        plt.plot(time[idx_test:], signal[idx_test:],
                 '-g', label = 'Testing data')
        plt.plot(time[idx_test:], outputs.numpy(), '-r', label = 'Prediction')
        plt.xlim((time[0], time[-1]))
        plt.ylim((y_min, y_max))
        plt.xlabel('Data')
        plt.ylabel('Target')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
