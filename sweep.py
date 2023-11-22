import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
from data_loader import data_loader, val_loader, StreetHazardsDataset
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/baselineproj')

import wandb
wandb.login()

# Initialize Weights & Biases
wandb.finish()
# wandb.init(project="baselineproject", entity="ajsen")

from torch.profiler import profile, record_function, ProfilerActivity


import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
      'name': 'Loss/val',
      'goal': 'minimize'   
    },
    'parameters': {
        'lr': {
            'min': 0.0001,
            'max': 0.1
        },
        'hidden_dim': {
            'values': [64, 128, 256, 512]
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'dropout': {
            'min': 0.0,
            'max': 0.5
        }
        # Add other hyperparameters here
    }
}





class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.5, hidden_dim=256):
        super(ResNetBinaryClassifier, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Define the fully connected layers for binary classification
        num_features = self.resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features

        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Binary Cross-Entropy Loss
def binary_loss(output, target):
    return nn.functional.binary_cross_entropy_with_logits(output, target)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize wandb with a new set of hyperparameters
    wandb.init()
    config = wandb.config

    learning_rate = config.lr
    dropout_rate = config.dropout
    batch_size = config.batch_size
    optimizer_choice = config.optimizer
    hidden_dim = config.hidden_dim
    num_epochs = 25

    # Modify the model initialization to use the dropout rate from the sweep
    model = ResNetBinaryClassifier(dropout_rate=dropout_rate, hidden_dim=hidden_dim).to(device)

    # Choose optimizer based on sweep config
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_choice == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    # Add other optimizers if needed
    # Dataset and DataLoader
    image_dir = 'train/images/training/t1-3'
    annotation_dir = 'train/annotations/training/t1-3'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Validation dataset
    image_val_dir = 'train/images/validation/t4'
    annotation_val_dir = 'train/annotations/validation/t4'
    val_dataset = StreetHazardsDataset(image_val_dir, annotation_val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Rest of your training loop comes here
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, _, anomaly_masks) in enumerate(data_loader):
            images = images.to(device)
            targets = (anomaly_masks.sum(dim=[1, 2, 3]) > 0).float().to(device)  # Binary labels

            # Forward pass
            outputs = model(images).squeeze()

            loss = binary_loss(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Logging
            wandb.log({"Loss/train": loss.item(), "epoch": epoch})
            print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for images, _, anomaly_masks in val_loader:
                images = images.to(device)
                targets = (anomaly_masks.sum(dim=[1, 2, 3]) > 0).float().to(device)
                outputs = model(images).squeeze()
    
                loss = binary_loss(outputs, targets)

                val_loss += loss.item()
                predictions = torch.sigmoid(outputs).round().squeeze().cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')

        # Log validation metrics
        wandb.log({"Loss/val": val_loss, "Accuracy/val": accuracy, "Precision/val": precision, "Recall/val": recall, "F1/val": f1, "epoch": epoch})

        print(f"Epoch: {epoch+1} - Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    wandb.finish()

    print("Training and validation completed.")


# Sweep function
def sweep():
    sweep_id = wandb.sweep(sweep_config, project="baselineproject", entity="ajsen")
    wandb.agent(sweep_id, train)

# Call the sweep function
if __name__ == "__main__":
    sweep()
    