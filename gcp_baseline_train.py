import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np

import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class StreetHazardsDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        annotation = Image.open(annotation_path)
        segmentation_mask = np.array(annotation, dtype=np.int64)  # Load full segmentation mask
        anomaly_mask = (segmentation_mask == 13)  # Anomalies are labeled with 13 in annotations
        
        if self.transform:
            image = self.transform(image)
            # Convert segmentation_mask to uint8 for PIL processing
            segmentation_mask_uint8 = segmentation_mask.astype(np.uint8)
            segmentation_mask_pil = Image.fromarray(segmentation_mask_uint8)
            segmentation_mask_pil = segmentation_mask_pil.resize((224, 224), Image.NEAREST)  # Use nearest neighbor to avoid interpolation artifacts
            segmentation_mask = torch.tensor(np.array(segmentation_mask_pil), dtype=torch.long)  # Convert back to tensor

            # Resize anomaly_mask as a PIL image to maintain consistency
            anomaly_mask_pil = Image.fromarray(anomaly_mask.astype(np.uint8) * 255)
            anomaly_mask_pil = anomaly_mask_pil.resize((224, 224), Image.NEAREST)
            anomaly_mask = torch.tensor(np.array(anomaly_mask_pil), dtype=torch.float32).unsqueeze(0) / 255

        return image, segmentation_mask, anomaly_mask


class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBinaryClassifier, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Add a new fully connected layer for binary classification
        num_features = self.resnet.fc.in_features  # Get the number of input features to the last layer
        self.resnet.fc = nn.Linear(num_features, 1)  # Replace with a new linear layer for binary classification

    def forward(self, x):
        return self.resnet(x)


# Binary Cross-Entropy Loss
def binary_loss(output, target):
    return nn.functional.binary_cross_entropy_with_logits(output, target)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 32

    # Initialize model, optimizer, and TensorBoard
    model = ResNetBinaryClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # Add DataLoader for the validation dataset
    image_val_dir = 'train/images/validation/t4'
    annotation_val_dir = 'train/annotations/validation/t4'
    val_dataset = StreetHazardsDataset(image_val_dir, annotation_val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

  
        print(f"Epoch: {epoch+1} - Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


    print("Training and validation completed.")

    # After training, save the model's state dictionary
    save_path = 'trained_model.pth'  # Define the path for saving the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")