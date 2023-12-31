# contains the training loop for the DAI algorithm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Add more imports as necessary, for example, for your dataset, models, etc.
from gcp_data_loader import StreetHazardsDataset, transform
from gcp_data_loader import val_loader # data_loader should be imported and taken out of the training script 
# from gcp_data_loader_(1) import StreetHazardsDataset, transform
# from gcp_data_loader_(1) import val_loader # data_loader should be imported and taken out of the training script 
from models import CNNEncoder, PolicyNetwork, BootstrappedEFENetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import logging

logging.basicConfig(level=logging.INFO)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.image_feature_extractor = models.resnet18(pretrained=True)
        self.image_feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_feature_extractor = nn.Sequential(*list(self.image_feature_extractor.children())[:-1])  # Remove the last FC layer
        print(self.image_feature_extractor)  # Debug: Print the model architecture

    def forward(self, image, mask):
        image_features = self.image_feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        # print(f"Image features shape: {image_features.shape}")  # Debug print
        print(f"Image features shape: {image_features.shape}")  # Debug print
        print(f"Image features mean: {image_features.mean()}, std: {image_features.std()}")  # Debug stats
        
        mask_hist = []
        for m in mask:
            hist = torch.histc(m.float(), bins=13, min=0, max=12)
            mask_hist.append(hist)
        mask_features = torch.stack(mask_hist)
        # print(f"Mask features shape: {mask_features.shape}")  # Debug print
        print(f"Mask histogram shape: {mask_features.shape}")  # Debug print
        print(f"Mask histogram sample: {mask_features[0]}")  # Example histogram
        
        combined_features = torch.cat([image_features, mask_features], dim=1)
        print(f"Combined features shape: {combined_features.shape}")  # Debug print
        # print(f"Combined features shape: {combined_features.shape}")  # Debug print
        return combined_features


class BeliefUpdateNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BeliefUpdateNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


def compute_entropy(pi):
    return -torch.sum(pi * torch.log(pi + 1e-10), dim=1)


def compute_free_energy(q, Q_actions_batch):
    entropy_q = compute_entropy(q)
    kl_term = torch.sum(q * torch.log((q + 1e-10) / (Q_actions_batch + 1e-10)), dim=1)
    return -entropy_q - kl_term


def policy_loss(F):
    return F

def efe_loss(G_phi, G):
    return torch.norm(G_phi - G, p=2)

def get_observation(action, image, mask, feature_extractor):
    observations = []
    batch_size, _, h, w = image.shape
    for i in range(batch_size):  # Loop over the batch
        act = action[i].item()  # Get the action as a Python integer
        img = image[i]  # Get the i-th image in the batch
        msk = mask[i].unsqueeze(0)  # Get the i-th mask in the batch
        
        # Depending on the action, select a different quadrant of the image
        if act == 0:  # top-left quadrant
            roi_image = img[:, :h//2, :w//2].unsqueeze(0)  # Unsqueeze to add the batch dimension back
            roi_mask = msk[:, :h//2, :w//2].unsqueeze(0)
        elif act == 1:  # top-right quadrant
            roi_image = img[:, :h//2, w//2:].unsqueeze(0)
            roi_mask = msk[:, :h//2, w//2:].unsqueeze(0)
        elif act == 2:  # bottom-left quadrant
            roi_image = img[:, h//2:, :w//2].unsqueeze(0)
            roi_mask = msk[:, h//2:, :w//2].unsqueeze(0)
        elif act == 3:  # bottom-right quadrant
            roi_image = img[:, h//2:, w//2:].unsqueeze(0)
            roi_mask = msk[:, h//2:, w//2:].unsqueeze(0)
        else:
            raise ValueError(f"Invalid action: {act}")

        # Obtain features for the selected ROI using the feature extractor
        observation = feature_extractor(roi_image, roi_mask)
        observations.append(observation)

    # Stack all observations along the batch dimension
    observations = torch.cat(observations, dim=0)
    return observations




def update_belief(current_belief, observation):
    updated_belief = belief_update(observation)
    # Combine the current belief and the new belief (you can adjust the weights)
    combined_belief = 0.7 * current_belief + 0.3 * updated_belief
    return combined_belief



def compute_actual_efe(current_belief, target_belief):
    # Ensure target_belief is a valid probability distribution
    target_belief = target_belief.clamp(min=1e-10)
    target_belief = target_belief / target_belief.sum(dim=1, keepdim=True)

    # Use log_softmax for current_belief
    log_current_belief = torch.log_softmax(current_belief, dim=1)

    # Compute KL divergence
    kl_div = nn.KLDivLoss(reduction='batchmean')  # change reduction to 'batchmean' if not already
    efe = kl_div(log_current_belief, target_belief)

    return efe


def get_target_belief(actions, masks):
    """
    Given actions and masks, compute the target beliefs for those actions.
    
    Parameters:
    - actions (torch.Tensor): The actions taken, determining the region of the image for each instance in the batch.
    - masks (torch.Tensor): The mask (annotation) images for the batch.
    
    Returns:
    - target_beliefs (torch.Tensor): The computed beliefs (distribution of classes) for the action's region for each instance in the batch.
    """
    target_beliefs = []
    
    for i in range(actions.size(0)):  # Loop over each example in the batch
        action = actions[i].item()  # Get the action for the current example
        mask = masks[i]  # Get the mask for the current example
        
        # Squeeze out the channel dimension if present
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)

        h, w = mask.shape[-2], mask.shape[-1]  # Height and width of the mask
        
        # Determine the region of interest based on the action
        if action == 0:  # top-left quadrant
            roi_mask = mask[:h//2, :w//2]
        elif action == 1:  # top-right quadrant
            roi_mask = mask[:h//2, w//2:]
        elif action == 2:  # bottom-left quadrant
            roi_mask = mask[h//2:, :w//2]
        elif action == 3:  # bottom-right quadrant
            roi_mask = mask[h//2:, w//2:]
        else:
            raise ValueError("Invalid action!")

        if roi_mask.numel() == 0:
            # Handle the empty ROI case appropriately
            # One option could be to return a uniform distribution over classes
            target_belief = torch.ones(13) / 13
        else:
            roi_mask = roi_mask.long()  # Convert to long to get integer values
        
            # Count the number of pixels for each class in the region of interest
            class_counts = torch.bincount(roi_mask.view(-1), minlength=13)  # Assuming 13 classes

            # Compute the belief for the current example
            target_belief = class_counts.float() / class_counts.sum()
        
        # Check for NaN values just in case
        if torch.isnan(target_belief).any():
            raise ValueError("NaN values encountered in target_belief.")
        
        target_beliefs.append(target_belief)
    
    return torch.stack(target_beliefs)  # Combine all target beliefs into a single tensor


if __name__ == "__main__":
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    hidden_dim = 256
    learning_rate = 0.001
    num_epochs = 100
    cnn_output_dim = 512  # For ResNet-18 without the final FC layer
    histogram_bins = 13  # Number of histogram bins for mask features (should match the number of classes)
    input_dim = cnn_output_dim + histogram_bins  # 525, Total input dimension for networks that follow feature extractor 
    output_dim_policy = 4  # For the policy network if you have 4 actions
    output_dim_belief = 13  # If belief update network outputs a belief distribution over classes

    if torch.cuda.is_available():
        torch.cuda.empty_cache()   

    
    # number of actions
    num_actions = 4

    # Initialize networks with the defined dimensions
    feature_extractor = FeatureExtractor().to(device)
    belief_update = BeliefUpdateNetwork(input_dim, hidden_dim, output_dim=13).to(device)
    combined_input_dim = cnn_output_dim + output_dim_belief
 
    # Correct initialization of the policy network
    policy_network = PolicyNetwork(input_dim=output_dim_belief, hidden_dim=hidden_dim, output_dim=output_dim_policy).to(device)

    # Correct initialization of the efe network
    efe_network = BootstrappedEFENetwork(output_dim_belief + num_actions, hidden_dim).to(device)

    optimizer_policy = optim.Adam(policy_network.parameters(), lr=learning_rate)
    optimizer_efe = optim.Adam(efe_network.parameters(), lr=learning_rate)
    
    # Create dataset
    image_dir = 'train/train/images/training/t1-3'
    annotation_dir = 'train/train/annotations/training/t1-3'
    dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

    # Create data loader
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Compute Q from the training data
    class_counts = torch.zeros(13)  # Assuming 13 classes
    for _, mask in data_loader:
        class_counts += torch.bincount(mask.view(-1).long(), minlength=13)
    Q = class_counts / class_counts.sum()

    print("Q shape:", Q.shape)  # Should be (num_classes,)

    # Q_actions = torch.full((num_actions,), fill_value=1.0/num_actions)  # Assuming a fixed uniform prior over actions

    test_batches = 2  # Only run 2 batches for each epoch during testing
    # Enable anomaly detection to find the operation that produces nan
    torch.autograd.set_detect_anomaly(True)
    # Simplified training loop for testing
    for epoch in range(num_epochs):
        # During training, set models to training mode
        feature_extractor.train()
        belief_update.train()
        policy_network.train()
        efe_network.train()
        for batch_idx, (image, mask) in enumerate(data_loader):
            mask = mask.unsqueeze(1)  # Add a channel dimension if not already present
            
            # if batch_idx >= test_batches:  # Only run a few batches for testing
            #     break
            
            image = image.to(device)
            mask = mask.to(device)
            # Forward pass through the network
            features = feature_extractor(image, mask)
            # print("Features shape:", features.shape)
            pi_prev = belief_update(features)
            # print("pi_prev shape:", pi_prev.shape)

            q = policy_network(pi_prev)
            # print("q shape:", q.shape)  # Should be (batch_size, num_actions)

            action = torch.argmax(q, dim=1)  

            # convert actions to one hot encoding 
            action_one_hot = torch.nn.functional.one_hot(action, num_classes=num_actions)
        
            observation = get_observation(action, image, mask, feature_extractor)
            pi_current = update_belief(pi_prev, observation)
            # Concatenate along the appropriate dimension if needed
            input_to_efe = torch.cat((pi_current, action_one_hot), dim=1)
        
            G_phi = efe_network(input_to_efe)
            target_belief = get_target_belief(action, mask)
            # print("pi_current shape:", pi_current.shape)
            # print("target_belief shape:", target_belief.shape)  

            G = compute_actual_efe(pi_current, target_belief)
            
            Q_batch = Q.unsqueeze(0).repeat(image.size(0), 1)  # Expand Q for each batch
            # print("Q_batch shape:", Q_batch.shape)  # Should be (batch_size, num_classes)

            Q_actions = torch.full((num_actions,), fill_value=1.0/num_actions)  # Assuming a dynamic uniform prior over actions
            Q_actions_batch = Q_actions.expand(q.size(0), -1)  # Expand Q_actions to match the batch size of q


            # Now Q_batch is aligned with q
            F = compute_free_energy(q, Q_actions_batch)

            # print("current_belief:", pi_current)
            # print("target_belief:", target_belief)
            assert not torch.isnan(target_belief).any(), "NaNs in target_belief"

            # Before loss calculation
            # print("G_phi before loss:", G_phi)
            # print("G before loss:", G)

            # Checking for NaNs or infs in tensors
            assert not torch.isnan(G_phi).any(), "NaNs in G_phi"
            assert not torch.isnan(G).any(), "NaNs in G"
            assert not torch.isinf(G_phi).any(), "Infs in G_phi"
            assert not torch.isinf(G).any(), "Infs in G"

            # Loss calculation
            loss_policy = policy_loss(F).mean()  # Ensure it's a scalar by taking the mean
            loss_efe = efe_loss(G_phi, G).mean()  # Ensure it's a scalar by taking the mean

            # Compute total loss
            total_loss = loss_policy + loss_efe

            # Zero gradients before backpropagation
            optimizer_policy.zero_grad()
            optimizer_efe.zero_grad()

            # Backpropagation with anomaly detection
            with torch.autograd.set_detect_anomaly(True):
                total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(efe_network.parameters(), max_norm=1.0)

            # # Backpropagation
            # total_loss.backward()

            # Perform a step of the optimizer
            optimizer_policy.step()
            optimizer_efe.step()


            # Print training loss every iteration for monitoring
            print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {total_loss.item()}")
        
        
        print("Training test completed.")

        # validation loop, complete the validation code here
        # During validation, set models to evaluation mode
        feature_extractor.eval()
        belief_update.eval()
        policy_network.eval()
        efe_network.eval()

        with torch.no_grad():  # Turn off gradients for validation
            # validation loop code based on the previous training code
            # After each training epoch, run a validation pass
            validation_loss = 0.0
            num_validation_batches = len(val_loader)  # Assuming you have a val_loader

            # Initialize lists to store true labels and predicted labels for evaluation metrics
            true_labels = []
            predicted_labels = []

            for batch_idx, (image, mask) in enumerate(val_loader):
                image = image.to(device)
                mask = mask.to(device)
                mask = mask.unsqueeze(1)  # Add a channel dimension if not already present

                # Forward pass through the network
                features = feature_extractor(image, mask)
                pi_prev = belief_update(features)
                q = policy_network(pi_prev)
                action = torch.argmax(q, dim=1)
                action_one_hot = torch.nn.functional.one_hot(action, num_classes=num_actions)

                observation = get_observation(action, image, mask, feature_extractor)
                pi_current = update_belief(pi_prev, observation)
                input_to_efe = torch.cat((pi_current, action_one_hot), dim=1)

                G_phi = efe_network(input_to_efe)
                target_belief = get_target_belief(action, mask)
                G = compute_actual_efe(pi_current, target_belief)

                Q_batch = Q.unsqueeze(0).repeat(image.size(0), 1)
                Q_actions = torch.full((num_actions,), fill_value=1.0/num_actions)
                Q_actions_batch = Q_actions.expand(q.size(0), -1)
                F = compute_free_energy(q, Q_actions_batch)

                # Calculate validation loss (without backpropagation and optimization)
                loss_policy = policy_loss(F).mean()
                loss_efe = efe_loss(G_phi, G).mean()
                total_loss = loss_policy + loss_efe

                validation_loss += total_loss.item()  # Sum up the loss for averaging

                # Optionally, print validation loss every iteration for monitoring
                print(f"Validation - Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {total_loss.item()}")
                # Convert predictions and target beliefs to class labels
                predicted_class = pi_current.argmax(dim=1)
                true_class = target_belief.argmax(dim=1)

                # Append to lists for later evaluation
                predicted_labels.extend(predicted_class.cpu().numpy())
                true_labels.extend(true_class.cpu().numpy())

            # Compute the average validation loss
            average_validation_loss = validation_loss / num_validation_batches
            print(f"Average Validation Loss for Epoch {epoch+1}: {average_validation_loss}")

            print("true_labels:", true_labels)
            print("predicted_labels", predicted_labels)
            # After the loop, calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')
            recall = recall_score(true_labels, predicted_labels, average='macro')
            f1 = f1_score(true_labels, predicted_labels, average='macro')

            print(f"Validation Metrics - Epoch {epoch+1}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

