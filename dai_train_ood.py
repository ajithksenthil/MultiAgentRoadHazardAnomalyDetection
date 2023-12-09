# contains the training loop for the DAI algorithm
import segmentation_models_pytorch as smp
from torchvision import transforms
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ood_models import CNNEncoder, PolicyNetwork, BootstrappedEFENetwork, BeliefUpdateNetwork, FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, jaccard_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from PIL import Image
import os
import csv

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

            # Resize both image and segmentation mask to 512x512
            segmentation_mask_pil = segmentation_mask_pil.resize((256, 256), Image.NEAREST)
            segmentation_mask = torch.tensor(np.array(segmentation_mask_pil), dtype=torch.long)

            anomaly_mask_pil = Image.fromarray(anomaly_mask.astype(np.uint8) * 255)
            anomaly_mask_pil = anomaly_mask_pil.resize((256, 256), Image.NEAREST)
            anomaly_mask = torch.tensor(np.array(anomaly_mask_pil), dtype=torch.float32).unsqueeze(0) / 255

        return image, segmentation_mask, anomaly_mask



class PSPNetResNet101(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(PSPNetResNet101, self).__init__()
        self.model = smp.PSPNet(
            encoder_name="resnet101", 
            encoder_weights="imagenet", 
            classes=13, 
            activation=None
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Extract features using the encoder
        features = self.model.encoder(x)

        # Apply dropout to the highest resolution features only
        if len(features) > 0 and isinstance(features[-1], torch.Tensor):
            features[-1] = self.dropout(features[-1])

        # Decode features and get logits
        x = self.model.decoder(*features)
        logits = self.model.segmentation_head(x)
        return logits

class UnetPlusPlusFeatureExtractor(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=13):
        super(UnetPlusPlusFeatureExtractor, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=classes, 
            activation=None
        )

    def forward(self, x):
        # Unet++ maintains spatial resolution in its output
        x = self.model(x)
        return x


def compute_target_belief(segmentation_masks, anomaly_class_idx):
    """
    Compute the target belief tensor for pixel-wise anomaly detection.
    
    Parameters:
    - segmentation_masks (torch.Tensor): Segmentation masks for the batch.
    - anomaly_class_idx (int): Index of the anomaly class.
    
    Returns:
    - torch.Tensor: Pixel-wise binary mask indicating anomaly presence.
    """
    anomaly_mask = (segmentation_masks == anomaly_class_idx).float()
    return anomaly_mask


def get_segmentation_prediction(logits):
    return torch.argmax(logits, dim=1)


def compute_belief_from_segmentation(segmentation_logits):
    """
    Convert segmentation logits to belief states for each class, including the anomaly class.

    Parameters:
    - segmentation_logits (torch.Tensor): Logits from the segmentation model. Shape: [batch_size, num_classes, H, W]

    Returns:
    - torch.Tensor: Belief states for each pixel. Shape: [batch_size, num_classes, H, W]
    """
    # Convert logits to probabilities for each pixel and each class
    probabilities = torch.softmax(segmentation_logits, dim=1)
    return probabilities



def compute_actual_efe(current_belief, target_belief):
    """
    Compute the actual expected free energy (EFE) for pixel-wise anomaly detection.

    Parameters:
    - current_belief (torch.Tensor): The current belief state. Shape: [batch_size, num_classes, H, W]
    - target_belief (torch.Tensor): The target belief state. Shape: [batch_size, H, W] (binary mask)

    Returns:
    - torch.Tensor: The EFE map for each pixel. Shape: [batch_size, 1, H, W]
    """
    # Expanding target_belief to match the shape of current_belief
    target_belief_expanded = target_belief.unsqueeze(1).float().expand_as(current_belief)

    # Convert current_belief to log probabilities
    log_current_belief = torch.log_softmax(current_belief, dim=1)

    # Compute pixel-wise KL divergence
    kl_div = nn.KLDivLoss(reduction='none')
    efe = kl_div(log_current_belief, target_belief_expanded)

    # Mask out regions belonging to the anomaly class
    anomaly_mask = target_belief == 12  # Assuming the anomaly class index is 12 (modify as needed)
    anomaly_mask = anomaly_mask.unsqueeze(1).float().expand_as(efe)
    efe = efe * torch.logical_not(anomaly_mask)

    # Summing over the class dimension
    efe = efe.sum(dim=1)

    return efe.unsqueeze(1)



def compute_multiclass_entropy(probabilities):
    """
    Compute the entropy for multi-class probability distributions.
    
    Parameters:
    - probabilities (torch.Tensor): Probability distributions for each class.
    
    Returns:
    - torch.Tensor: Entropy values for the probability distributions.
    """
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)


def compute_multiclass_free_energy(q, Q_actions_batch):
    """
    Compute the free energy for multi-class policy distributions.

    Parameters:
    - q (torch.Tensor): Policy distributions for the current state.
    - Q_actions_batch (torch.Tensor): Prior distributions for actions.

    Returns:
    - torch.Tensor: Free energy values for the policy distributions.
    """
    # Reshape q and Q_actions_batch if necessary
    if q.shape != Q_actions_batch.shape:
        q = q.permute(0, 3, 1, 2)  # Change from [B, H, W, C] to [B, C, H, W] if necessary
        Q_actions_batch = Q_actions_batch.unsqueeze(-1).unsqueeze(-1).expand_as(q)  # Expand Q_actions_batch to match q's shape

    entropy_q = compute_multiclass_entropy(q)
    kl_term = torch.sum(q * torch.log((q + 1e-10) / (Q_actions_batch + 1e-10)), dim=1)
    return -entropy_q - kl_term



def policy_loss(F):
    """
    Define the policy loss as the negative free energy.
    
    Parameters:
    - F (torch.Tensor): Free energy values for the policy distributions.
    
    Returns:
    - torch.Tensor: Policy loss values.
    """
    return torch.mean(F)

def efe_loss(G_phi, G):
    """
    Define the expected free energy (EFE) loss as the L2 norm between predicted and actual EFE.
    
    Parameters:
    - G_phi (torch.Tensor): Predicted EFE values.
    - G (torch.Tensor): Actual EFE values.
    
    Returns:
    - torch.Tensor: EFE loss value.
    """
    return torch.mean(torch.norm(G_phi - G, p=2))



def get_target_belief(anomaly_mask):
    """
    Given masks, compute the target beliefs indicating the presence of an anomaly for each pixel.
    
    Parameters:
    - masks (torch.Tensor): The mask (annotation) images for the batch. Shape: [batch_size, 1, H, W]
        
    Returns:
    - target_beliefs (torch.Tensor): Pixel-wise beliefs indicating the presence (1) or absence (0) of anomalies. Shape: [batch_size, H, W]
    """
    # Convert to binary labels for each pixel
    target_beliefs = (anomaly_mask > 0).float()
    return target_beliefs.squeeze(1)


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


def update_belief(current_belief, observation, belief_update_network):
    """
    Update the belief state based on new observation using the belief update network.

    Parameters:
    - current_belief (torch.Tensor): The current belief state. Shape: [batch_size, num_classes, H, W]
    - observation (torch.Tensor): The new observation to update the belief. Shape: [batch_size, num_classes, H, W]
    - belief_update_network (nn.Module): The network to update beliefs.

    Returns:
    - torch.Tensor: The updated belief state. Shape: [batch_size, num_classes, H, W]
    """
    updated_belief = belief_update_network(observation)
    combined_belief = 0.7 * current_belief + 0.3 * updated_belief
    return combined_belief




if __name__ == "__main__":

    # Set default CUDA device
    # if torch.cuda.is_available():
        # torch.cuda.set_device(1)  # Set default device in case of multiple GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    hidden_dim = 356  # Adjusted for task complexity
    learning_rate = 0.005
    num_epochs = 1
    num_classes = 13  # 13 classes including the anomaly class (13th)
    # Assuming the feature extractor outputs a feature vector of size 2048
    # H = 512
    H = 256
    W = 256
    # W = 512
    output_dim_belief = num_classes  # 12 normal classes + 1 anomaly class, might be incorrect
    input_dim_efe = num_classes * 2  # 26 for 13 classes might be incorrect
    
    # Initialize feature extractor with PSPNet for high-resolution feature extraction
    # feature_extractor = PSPNetFeatureExtractor(encoder_name="resnet101", encoder_weights="imagenet").to(device) # alternate
    feature_extractor = UnetPlusPlusFeatureExtractor(encoder_name="resnet34", encoder_weights="imagenet").to(device)
    # decided to use Unet++ instead to maintain spatial resolution

    # Initialize belief update network with adjusted input dimension
    # input_dim = 13, hidden_dim = 356, output_dim = 13
    belief_update = BeliefUpdateNetwork(input_channels=num_classes, hidden_channels=hidden_dim, output_channels=output_dim_belief).to(device)

    # Initialize policy network for multi-class decision-making
    # input_dim = 13, hidden_dim = 356, output_dim = 13
    policy_network = PolicyNetwork(input_channels=num_classes, hidden_channels=hidden_dim, output_channels=num_classes).to(device)

    # Initialize bootstrapped EFE network for multi-class EFE calculation
    # input_dim = 13, hidden_dim = 356
    efe_network = BootstrappedEFENetwork(input_channels=input_dim_efe, hidden_channels=hidden_dim).to(device)
    torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        feature_extractor = nn.DataParallel(feature_extractor).to(device)
        belief_update = nn.DataParallel(belief_update).to(device)
        policy_network = nn.DataParallel(policy_network).to(device)
        efe_network = nn.DataParallel(efe_network).to(device)
 

    # Define optimizers for policy and EFE networks
    optimizer_policy = optim.AdamW(policy_network.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer_efe = optim.AdamW(efe_network.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define transformations and create datasets
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_dir = 'train/images/training/t1-3'
    annotation_dir = 'train/annotations/training/t1-3'
    dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

    # Create data loader
    batch_size = 2 # changing for testing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Add DataLoader for the validation dataset
    image_val_dir = 'train/images/validation/t4'
    annotation_val_dir = 'train/annotations/validation/t4'
    val_dataset = StreetHazardsDataset(image_val_dir, annotation_val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize class distribution Q for active inference
    class_counts = torch.zeros(num_classes)
    for _, segmentation_masks, _ in data_loader:
        segmentation_masks_flat = segmentation_masks.view(-1)
        class_counts += torch.bincount(segmentation_masks_flat.long(), minlength=num_classes)
    Q = class_counts / class_counts.sum() # Tensor shape: [num_classes]

    print("Q:", Q)
    print("Q shape:", Q.shape)
    

    # torch.autograd.set_detect_anomaly(True)

    # Proceed to the training and validation loop...
    # validation and training loop
    # Hyperparameters for updating Q_actions
    learning_adjustment = 0.01 

    
    for epoch in range(num_epochs):
        # Set models to training mode
        feature_extractor.train()
        belief_update.train()
        policy_network.train()
        efe_network.train()

        # Initialize Q_actions with a uniform distribution for each pixel at the start of each epoch
        Q_actions_epoch = torch.full((batch_size, num_classes, H, W), fill_value=1.0 / num_classes, device=device)

        

        for batch_idx, (images, segmentation_masks, anomaly_masks) in enumerate(data_loader):
            images = images.to(device)  # [batch_size, 3, H, W]
            segmentation_masks = segmentation_masks.to(device)  # [batch_size, H, W]
            anomaly_masks = anomaly_masks.to(device)  # [batch_size, 1, H, W]
            print("images shape", images.shape, "segmentation_masks shape", segmentation_masks.shape, "anomaly_masks shape", anomaly_masks.shape)
            segmentation_logits = feature_extractor(images)  # [batch_size, num_classes, H, W]
            print("segmentation_logits shape", segmentation_logits.shape) # ([8, 13, 512, 512]) PSP: [8, 512, 64, 64]
            current_belief = compute_belief_from_segmentation(segmentation_logits)  # [batch_size, num_classes, H, W]
            print("current_belief shape", current_belief.shape) # ([8, 13, 512, 512]) PSP: [8, 512, 64, 64]

            updated_belief = update_belief(current_belief, segmentation_logits, belief_update)  # [batch_size, num_classes, H, W]
            print("updated_belief shape", updated_belief.shape) # updated_belief shape [8, 13, 512, 512]
            # Compute the target belief for multi-class classification
            target_belief = compute_target_belief(segmentation_masks, num_classes) # Tensor shape: [batch_size, H, W]
            print("target_belief shape", target_belief.shape) # target_belief shape torch.Size([8, 512, 512])
            # Calculate policy network output
            q = policy_network(updated_belief) # Tensor shape: [batch_size, num_classes], Intended Shape: [batch_size, num_classes, H, W]
            print("q shape", q.shape)
            # Concatenate belief state and action for EFE input
            input_to_efe = torch.cat([updated_belief, q], dim=1) # Tensor shape: [batch_size, num_classes * 2, H, W]
            print("input_to_efe shape", input_to_efe.shape) # ([8, 26, 512, 512])
            # Calculate the expected free energy
            G_phi = efe_network(input_to_efe) # Tensor shape: [batch_size, 1, H, W]
            print("G_phi shape", G_phi.shape) # [8, 1, 512, 512])

            G = compute_actual_efe(updated_belief, target_belief) # Tensor shape: [batch_size, 1, H, W]
            print("G shape", G.shape) # shape [batch_size] = [8]
            # Update Q_actions based on the actions chosen

            print("Q_actions_epoch shape", Q_actions_epoch.shape) # [num_classes] Intended Shape: [batch_size, num_classes, H, W]
            # Update Q_actions_epoch based on prior belief state
            for i, action in enumerate(range(num_classes)):
                # Select pixels belonging to the current action class
                action_mask = (target_belief == action).float()
                
                # Extract class probabilities from prior belief
                # updated_belief shape torch.Size([8, 13, 512, 512])
                action_probs = updated_belief[:, i, :, :]

                # Assign higher Q-values to pixels with higher class probabilities
                Q_actions_epoch[:, i, :, :] = torch.where(action_mask > 0, action_probs * learning_adjustment, Q_actions_epoch[:, i, :, :])


            # Normalize Q_actions_batch to get probability distribution for each pixel
            print("Q_actions_epoch shape", Q_actions_epoch.shape) # [num_classes] Intended Shape: [batch_size, num_classes, H, W]
            Q_actions_batch = torch.nn.functional.normalize(Q_actions_epoch, p=1, dim=1) # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)


            print("Q_actions_batch.shape", Q_actions_batch.shape) # [num_classes] Intended Shape for Pixel-wise Decisions: [batch_size, num_classes, H, W]
            print("q shape", q.shape) # [num_classes]
            # Calculate free energy for the policy network
            F = compute_multiclass_free_energy(q, Q_actions_batch) # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
            print("F shape", F.shape)
            # Compute the total loss
            loss_policy = policy_loss(F)
            print("loss_policy", loss_policy)
            loss_efe = efe_loss(G_phi, G)
            print("loss_efe", loss_efe)
            total_loss = loss_policy + loss_efe
            print("total_loss", total_loss)
            # Backpropagation
            optimizer_policy.zero_grad()
            optimizer_efe.zero_grad()
            print("zeroed out gradients now its time for backwards")
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(efe_network.parameters(), max_norm=1.0)

            # Optimizer step
            print("step")
            optimizer_policy.step()
            optimizer_efe.step()

            print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {total_loss.item()}")

        # Reset Q_actions for the next epoch
        Q_actions_epoch.fill_(1.0 / num_classes)

        print(f"Epoch {epoch+1} training completed.")

        # Proceed to the validation loop...


        print("Training completed.")    

        print("skipping validation")
        # to debug train, temporary
        break
        # validation loop, complete the validation code here
        # During validation, set models to evaluation mode
        # Validation Loop
        for epoch in range(num_epochs):
            # Set models to evaluation mode
            feature_extractor.eval()
            belief_update.eval()
            policy_network.eval()
            efe_network.eval()

            val_loss = 0.0
            ious, pixel_accuracies, precision_list, recall_list = [], [], [], []
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for images, segmentation_masks, anomaly_masks in val_loader:
                    images = images.to(device)
                    segmentation_masks = segmentation_masks.to(device)
                    anomaly_masks = anomaly_masks.to(device)

                    # Forward pass through the feature extractor
                    segmentation_logits = feature_extractor(images)

                    # Update belief based on segmentation predictions
                    updated_belief = compute_belief_from_segmentation(segmentation_logits)

                    # Compute IoU for segmentation accuracy
                    segmentation_pred = get_segmentation_prediction(segmentation_logits)
                    iou = compute_iou(segmentation_pred, segmentation_masks, num_classes=13)
                    ious.append(iou)

                    # Compute metrics for OoD detection
                    target_belief = get_target_belief(anomaly_masks)
                    anomaly_preds = updated_belief >= 0.5  # Threshold belief for anomaly detection
                    pixel_accuracies.append(np.mean(anomaly_preds.cpu().numpy() == anomaly_masks.cpu().numpy()))

                    # Prepare data for AUROC, AUPR, and FPR95 calculations
                    anomaly_truth_flat = anomaly_masks.view(-1).cpu().numpy()
                    anomaly_preds_flat = anomaly_preds.view(-1).cpu().numpy()
                    precision, recall, _ = precision_recall_curve(anomaly_truth_flat, anomaly_preds_flat)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    fpr, tpr, _ = roc_curve(anomaly_truth_flat, anomaly_preds_flat)
                    if np.where(tpr >= 0.95)[0].size > 0:
                        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]] * 100
                    else:
                        fpr95 = 100.0  # Default to 100% if TPR never reaches 95%

                    auroc = roc_auc_score(anomaly_truth_flat, anomaly_preds_flat) * 100

                # Average the metrics over the validation set
                avg_val_iou = np.nanmean(ious)
                avg_val_pixel_accuracy = np.mean(pixel_accuracies)
                avg_val_aupr = np.mean([auc(recall_list[i], precision_list[i]) for i in range(len(precision_list))] * 100)
                avg_val_fpr95 = fpr95
                avg_val_auroc = auroc

                print(f"Epoch: {epoch+1} - Val mIoU: {avg_val_iou:.4f}, Pixel Accuracy: {avg_val_pixel_accuracy:.4f}, AUPR: {avg_val_aupr:.2f}%, FPR95: {avg_val_fpr95:.2f}%, AUROC: {avg_val_auroc:.2f}%")

