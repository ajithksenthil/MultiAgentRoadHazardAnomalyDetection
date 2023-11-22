import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
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



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.image_feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_feature_extractor = nn.Sequential(*list(self.image_feature_extractor.children())[:-1])  # Remove the last FC layer

    def forward(self, image):
        image_features = self.image_feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        return image_features

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create training dataset
image_dir = 'train/images/training/t1-3'
annotation_dir = 'train/annotations/training/t1-3'
dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

# validation data set
image_validation_dir = 'train/images/validation/t4'
annotation_validation_dir = 'train/annotations/validation/t4'
val_dataset = StreetHazardsDataset(image_validation_dir, annotation_validation_dir, transform=transform)

# Create data loader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create an instance of your dataset
    dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

    # Visualization function
    def visualize_data(image, segmentation_mask, anomaly_mask):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title('Image')
        plt.subplot(1, 3, 2)
        plt.imshow(segmentation_mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(anomaly_mask[0], cmap='gray')
        plt.title('Anomaly Mask')
        plt.show()

    # Check and visualize a few examples
    for i in range(3):
        image, segmentation_mask, anomaly_mask = dataset[i]
        print(f"Image {i}:")
        print(f"  - Image shape: {image.shape}, type: {image.dtype}")
        print(f"  - Segmentation mask shape: {segmentation_mask.shape}, unique values: {torch.unique(segmentation_mask)}")
        print(f"  - Anomaly mask shape: {anomaly_mask.shape}, unique values: {torch.unique(anomaly_mask)}")
        visualize_data(image, segmentation_mask, anomaly_mask)

    # DataLoader check
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, segmentation_masks, anomaly_masks = next(iter(data_loader))

    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of segmentation masks shape: {segmentation_masks.shape}")
    print(f"Batch of anomaly masks shape: {anomaly_masks.shape}")

    # Feature extraction check
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    with torch.no_grad():
        images = images.to(device)
        features = feature_extractor(images)
        print(f"Extracted features shape: {features.shape}")
        print(f"Features statistics - Mean: {features.mean()}, Std: {features.std()}")