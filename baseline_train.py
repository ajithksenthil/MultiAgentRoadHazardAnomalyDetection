import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, jaccard_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from PIL import Image
import numpy as np
from models import MaxLogitAnomalyDetector
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

        # Map the anomaly class from 13 to 12
        segmentation_mask[segmentation_mask == 13] = 12

        anomaly_mask = (segmentation_mask == 12)  # Update anomaly mask

        if self.transform:
            image = self.transform(image)
            segmentation_mask_uint8 = segmentation_mask.astype(np.uint8)
            segmentation_mask_pil = Image.fromarray(segmentation_mask_uint8)
            segmentation_mask_pil = segmentation_mask_pil.resize((224, 224), Image.NEAREST)
            segmentation_mask = torch.tensor(np.array(segmentation_mask_pil), dtype=torch.long)

            anomaly_mask_pil = Image.fromarray(anomaly_mask.astype(np.uint8) * 255)
            anomaly_mask_pil = anomaly_mask_pil.resize((224, 224), Image.NEAREST)
            anomaly_mask = torch.tensor(np.array(anomaly_mask_pil), dtype=torch.float32).unsqueeze(0) / 255

        return image, segmentation_mask, anomaly_mask

    
    """ 
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
    """

class MaxLogitAnomalyDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(MaxLogitAnomalyDetector, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2]) # retain up to the last convolutional layer

        # Upsample and classifier layers
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.classifier = nn.Conv2d(512, 13, 1)  # 13 classes for StreetHazards

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Upsample to input size and classify
        x = self.upsample(x)
        outputs = self.classifier(x)

        return outputs



# max logit loss
def max_logit_loss(output, target):
    return nn.functional.cross_entropy(output, target)

def compute_iou(preds, labels, num_classes=13):
    iou_list = []
    preds = preds.flatten()  # Flatten the array if not already
    labels = labels.flatten()  # Flatten the array if not already

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            iou_list.append(float('nan'))  # Exclude classes not present in ground truth
        else:
            iou_list.append(intersection / union)

    return np.nanmean(iou_list)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 15 # change to 20
    batch_size = 32
    num_classes = 13

    # Initialize model, optimizer, and TensorBoard
    model = MaxLogitAnomalyDetector().to(device)
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
        train_loss = 0
        for batch_idx, (images, segmentation_masks, _) in enumerate(data_loader):
            images, segmentation_masks = images.to(device), segmentation_masks.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = max_logit_loss(outputs, segmentation_masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}")
        avg_train_loss = train_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        ious, pixel_accuracies, precision_list, recall_list = [], [], [], []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, segmentation_masks, _ in val_loader:
                images, segmentation_masks = images.to(device), segmentation_masks.to(device)
                outputs = model(images)

                val_loss += max_logit_loss(outputs, segmentation_masks).item()

                preds = torch.argmax(outputs, dim=1).view(-1).cpu().numpy()
                masks_np = segmentation_masks.cpu().numpy()

                # Flatten arrays for metric calculations
                preds_flat = preds.flatten()
                masks_flat = masks_np.flatten()

                # Pixel Accuracy
                pixel_accuracies.append(np.mean(preds_flat == masks_flat))

                # Mean IoU
                ious.append(compute_iou(preds, masks_np, num_classes=13))

                # Prepare for AUPR calculation (focusing on the anomaly class)
                anomaly_class_idx = 12  # Assuming anomaly class is indexed at 12
                anomaly_preds = (preds_flat == anomaly_class_idx)
                anomaly_truth = (masks_flat == anomaly_class_idx)

                precision, recall, _ = precision_recall_curve(anomaly_truth, anomaly_preds)
                precision_list.append(precision)
                recall_list.append(recall)
             

        # Average the metrics over the validation set
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = np.nanmean(ious)
        avg_pixel_accuracy = np.mean(pixel_accuracies)

        # AUPR calculation (average AUPR if there are multiple batches)
        avg_aupr = np.mean([auc(recall_list[i], precision_list[i]) for i in range(len(precision_list))])

        print(f"Epoch: {epoch+1} - Val Loss: {avg_val_loss:.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}, mIoU: {avg_iou:.4f}, AUPR: {avg_aupr:.4f}")

    print("Training and validation completed.")