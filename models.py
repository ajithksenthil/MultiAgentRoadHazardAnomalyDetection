import torch
import torch.nn as nn
# contains the neural architectures for the DAI algorithm

import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.image_feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_feature_extractor = nn.Sequential(*list(self.image_feature_extractor.children())[:-1])  # Remove the last FC layer

    def forward(self, image):
        image_features = self.image_feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        return image_features


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.softmax(self.fc3(x))
        return x




class BootstrappedEFENetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(BootstrappedEFENetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class BeliefUpdateNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(BeliefUpdateNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.softmax(self.fc3(x))
        return x



class MaxLogitAnomalyDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(MaxLogitAnomalyDetector, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add a softmax layer for multi-class classification
        self.classifier = nn.Linear(self.resnet.fc.in_features, 13)  # 13 classes for StreetHazards

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features

        # Classification
        outputs = self.classifier(x)

        return outputs



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
