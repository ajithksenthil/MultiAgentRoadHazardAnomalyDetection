import torch
import torch.nn as nn
# contains the neural architectures for the DAI algorithm and SAC

class PolicyNetwork(nn.Module):
    """
    Neural network to represent the stochastic selection policy.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class BootstrappedEFENetwork(nn.Module):
    """
    Neural network to represent the bootstrapped Expected Free Energy (EFE).
    """
    def __init__(self, input_dim, hidden_dim):
        super(BootstrappedEFENetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Outputs a single scalar representing EFE
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class BootstrappedEFENetwork(nn.Module):
#     """
#     Neural network to represent the bootstrapped Expected Free Energy (EFE).
#     """
#     def __init__(self, input_dim, hidden_dim):
#         super(BootstrappedEFENetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1) # Outputs a single scalar representing EFE
    
#     def forward(self, pi, action):
#         # pi is the belief state
#         # action is the action taken
#         # Combine the belief state and action in a way that makes sense for your network
#         # This might be concatenation, or the action might modulate the input in some way
#         x = torch.cat((pi, action), dim=-1)
        
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Define some convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        # Add a fully connected layer to get the belief vector
        # The output size depends on the size of the input image and the architecture.
        # This is just a placeholder and might need adjustment.
        self.fc = nn.Linear(128 * 8 * 8, 256)  # Adjust the size as per the output of conv layers

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        
        # Flatten and pass through FC to get belief vector
        x = x.view(x.size(0), -1)
        pi = torch.relu(self.fc(x))
        return pi
