import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
        anomaly_mask = np.array(annotation) == 13  # Anomalies are labeled with 13 in annotations
        
        if self.transform:
            image = self.transform(image)
            anomaly_mask = torch.tensor(anomaly_mask, dtype=torch.float32)
            
        return image, anomaly_mask

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset
image_dir = 'train/images/training/t1-3'
annotation_dir = 'train/annotations/training/t1-3'
dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

# Create data loader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Visualize an example
example_image, example_mask = dataset[0]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(example_image.permute(1, 2, 0).numpy())
plt.title('Image')
plt.subplot(1, 2, 2)
plt.imshow(example_mask.numpy(), cmap='gray')
plt.title('Anomaly Mask')
plt.show()
