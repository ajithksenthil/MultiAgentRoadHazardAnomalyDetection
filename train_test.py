# contains the training loop for the DAI algorithm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Add more imports as necessary, for example, for your dataset, models, etc.
from data_loader import StreetHazardsDataset, transform
from data_loader import val_loader # data_loader should be imported and taken out of the training script 
from models import CNNEncoder, PolicyNetwork, BootstrappedEFENetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import logging

logging.basicConfig(level=logging.INFO)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.image_feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_feature_extractor = nn.Sequential(*list(self.image_feature_extractor.children())[:-1])  # Remove the last FC layer

    def forward(self, image):
        image_features = self.image_feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        return image_features



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

def get_observation(image, feature_extractor):
    # Process the whole image with the feature extractor
    observation = feature_extractor(image)
    return observation



def update_belief(current_belief, observation):
    updated_belief = belief_update(observation)
    # Combine the current belief and the new belief (you can adjust the weights)
    combined_belief = 0.7 * current_belief + 0.3 * updated_belief
    return combined_belief

# def compute_actual_efe(current_belief, target_belief):
#     # If target_belief is 1D (binary labels), reshape it to 2D for KL divergence calculation
#     if target_belief.dim() == 1:
#         target_belief = target_belief.unsqueeze(1)  # Shape: [batch_size, 1]
#         target_belief = torch.cat([1 - target_belief, target_belief], dim=1)  # Convert to two-class format

#     # Use log_softmax for current_belief
#     log_current_belief = torch.log_softmax(current_belief, dim=1) + 1e-10

#     # Compute KL divergence
#     kl_div = nn.KLDivLoss(reduction='batchmean')
#     efe = kl_div(log_current_belief, target_belief)

#     return efe

def compute_actual_efe(current_belief, target_belief):
    # Ensure target_belief is 2D for KL divergence calculation
    if target_belief.dim() == 1:
        target_belief = target_belief.unsqueeze(1)
        target_belief = torch.cat([1 - target_belief, target_belief], dim=1)

    # Convert target_belief to float type if it's not already
    target_belief = target_belief.type_as(current_belief)

    # Use log_softmax for current_belief
    log_current_belief = torch.log_softmax(current_belief, dim=1)

    # Compute KL divergence
    kl_div = nn.KLDivLoss(reduction='batchmean')
    efe = kl_div(log_current_belief, target_belief.float())  # Ensure target_belief is float

    return efe



def get_target_belief(anomaly_mask):
    """
    Given masks, compute the target beliefs indicating the presence of an anomaly.
    
    Parameters:
    - masks (torch.Tensor): The mask (annotation) images for the batch.
        
    Returns:
    - target_beliefs (torch.Tensor): The computed beliefs indicating the presence (1) or absence (0) of anomalies for each instance in the batch.
   """
    # Check if there is any anomaly in the mask (any pixel with value > 0)
    target_beliefs = (anomaly_mask.sum(dim=[1, 2, 3]) > 0).float()  # Convert to binary labels
    return target_beliefs


if __name__ == "__main__":
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    hidden_dim = 356 # originally 256
    learning_rate = 0.005
    num_epochs = 10
    cnn_output_dim = 512  # For ResNet-18 without the final FC layer
    input_dim = cnn_output_dim  # Adjusted for binary classification
    output_dim_belief = 2  # Binary classification: presence or absence of anomaly

    if torch.cuda.is_available():
        torch.cuda.empty_cache()   

    # Initialize networks with the defined dimensions
    feature_extractor = FeatureExtractor().to(device)
    belief_update = BeliefUpdateNetwork(input_dim, hidden_dim, output_dim=output_dim_belief).to(device)
    # belief_update = BeliefUpdateNetwork(input_dim+2, hidden_dim, output_dim=output_dim_belief).to(device)
 
    # Correct initialization of the policy network
    policy_network = PolicyNetwork(input_dim=output_dim_belief, hidden_dim=hidden_dim, output_dim=2).to(device)

    # Correct initialization of the efe network
    # efe_network = BootstrappedEFENetwork(input_dim=cnn_output_dim + 2, hidden_dim=hidden_dim).to(device)
    efe_network = BootstrappedEFENetwork(input_dim=4, hidden_dim=hidden_dim).to(device)


    # optimizer_policy = optim.Adam(policy_network.parameters(), lr=learning_rate)
    # optimizer_efe = optim.Adam(efe_network.parameters(), lr=learning_rate)
    # accuracy: epoch: 

    # optimizer_policy = optim.SGD(policy_network.parameters(), lr=0.01, momentum=0.9)
    # optimizer_efe = optim.SGD(efe_network.parameters(), lr=0.01, momentum=0.9)

    # optimizer_policy = optim.RMSprop(policy_network.parameters(), lr=0.001, alpha=0.99)
    # optimizer_efe = optim.RMSprop(efe_network.parameters(), lr=0.001, alpha=0.99)

    # optimizer_policy = optim.Adagrad(policy_network.parameters(), lr=0.01)
    # optimizer_efe = optim.Adagrad(efe_network.parameters(), lr=0.01)

    optimizer_policy = optim.AdamW(policy_network.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_efe = optim.AdamW(efe_network.parameters(), lr=0.001, weight_decay=0.01)

    # from adabelief_pytorch import AdaBelief
    # optimizer_policy = AdaBelief(policy_network.parameters(), lr=0.001, eps=1e-16, betas=(0.9, 0.999), weight_decay=0.01)
    # optimizer_efe = AdaBelief(efe_network.parameters(), lr=0.001, eps=1e-16, betas=(0.9, 0.999), weight_decay=0.01)




    
    # Create dataset
    image_dir = 'train/images/training/t1-3'
    annotation_dir = 'train/annotations/training/t1-3'
    dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

    # Create data loader
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    class_counts = torch.zeros(2)  # Two classes: 0 for normal, 1 for anomaly
    for _, _, anomaly_mask in data_loader:
        anomaly_mask_flat = anomaly_mask.view(-1)  # Flatten the mask
        class_counts += torch.bincount(anomaly_mask_flat.long(), minlength=2)
    Q = class_counts / class_counts.sum()
    


    print("Q:", Q)  # Q should be a tensor of shape [2]



    print("Q shape:", Q.shape)  # Should be (num_classes,)

   
    # Enable anomaly detection to find the operation that produces nan
    torch.autograd.set_detect_anomaly(True)



    belief_update = BeliefUpdateNetwork(input_dim, hidden_dim, output_dim_belief).to(device)
    belief_update.eval()  # Set to evaluation mode to disable dropout, if any

    for i, (image, _, mask) in enumerate(data_loader):
        if i >= 5:  # Check first 5 batches
            break
        image, mask = image.to(device), mask.to(device)
        features = feature_extractor(image)
        pi_current = belief_update(features)
        print(f"Batch {i}, pi_current: {pi_current}")


    test_distributions = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.9, 0.1]], device=device)
    for dist in test_distributions:
        entropy = compute_entropy(dist.unsqueeze(0))
        print(f"Distribution: {dist}, Entropy: {entropy}")


    test_policies = torch.tensor([[0.7, 0.3], [0.5, 0.5]], device=device)
    Q_actions = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device)
    for policy in test_policies:
        free_energy = compute_free_energy(policy.unsqueeze(0), Q_actions)
        print(f"Policy: {policy}, Free Energy: {free_energy}")

    dummy_F = torch.tensor([0.5, -0.2], device=device)  # Sample free energy values
    dummy_G_phi = torch.tensor([[0.1], [0.3]], device=device)  # Sample G_phi values
    dummy_G = torch.tensor(0.2, device=device)  # Sample G value

    loss_policy = policy_loss(dummy_F).mean()
    loss_efe = efe_loss(dummy_G_phi, dummy_G).mean()

    print(f"Dummy Policy Loss: {loss_policy}")
    print(f"Dummy EFE Loss: {loss_efe}")

    image_batch, _, _ = next(iter(data_loader))
    image_batch = image_batch.to(device)
    features = feature_extractor(image_batch)
    pi_current = belief_update(features)

    print(f"Extracted Features: {features}")
    print(f"Updated Belief (pi_current): {pi_current}")

    for i in range(5):
        _, _, anomaly_mask = dataset[i]
        target_belief = get_target_belief(anomaly_mask.unsqueeze(0).to(device))
        print(f"Anomaly Mask {i}: {anomaly_mask.squeeze()}")
        print(f"Target Belief {i}: {target_belief}")


    dummy_current_belief = torch.tensor([[0.7, 0.3], [0.5, 0.5]], device=device)
    dummy_target_belief = torch.tensor([[0, 1], [1, 0]], device=device)

    for i in range(2):
        efe = compute_actual_efe(dummy_current_belief[i].unsqueeze(0), dummy_target_belief[i].unsqueeze(0))
        print(f"Current Belief: {dummy_current_belief[i]}, Target Belief: {dummy_target_belief[i]}, EFE: {efe}")


    # # validation and training loop
    # # Hyperparameters for updating Q_actions
    # learning_adjustment = 0.01 
    # for epoch in range(num_epochs):
    #     # Set models to training mode
    #     feature_extractor.train()
    #     belief_update.train()
    #     policy_network.train()
    #     efe_network.train()

    #     # Initialize Q_actions with a uniform distribution at the start of each epoch
    #     Q_actions_epoch = torch.full((2,), fill_value=0.5, device=device)

    #     for batch_idx, (image, _, mask) in enumerate(data_loader):
    #         image, mask = image.to(device), mask.to(device)

    #         # Forward pass through the feature extractor
    #         features = feature_extractor(image)

    #         # Update belief based on features
    #         pi_current = belief_update(features) # input shape:512 output shape: 2

    #         # Compute the target belief for binary classification
    #         target_belief = get_target_belief(mask)  # Assuming the presence of an anomaly is 1, absence is 0

    #         # Calculate policy network output
    #         q = policy_network(pi_current)

    #         # Concatenate belief state and action for EFE input
    #         input_to_efe = torch.cat([pi_current, q], dim=1)

    #         # Calculate the expected free energy
    #         G_phi = efe_network(input_to_efe)  # Adjust the efe_network as necessary
    #         G = compute_actual_efe(pi_current, target_belief)

    #         # Update Q_actions based on the actions chosen
    #         action_chosen = torch.argmax(q, dim=1)
    #         for action in range(2):  # Assuming 2 actions
    #             Q_actions_epoch[action] += learning_adjustment * (action_chosen == action).float().mean()

    #         # Normalize Q_actions to maintain a valid probability distribution
    #         Q_actions_batch = torch.nn.functional.normalize(Q_actions_epoch.unsqueeze(0).repeat(image.size(0), 1), p=1, dim=1)

    #         # Calculate free energy for the policy network
    #         F = compute_free_energy(q, Q_actions_batch)

    #         # Compute the total loss
    #         loss_policy = policy_loss(F).mean()
    #         loss_efe = efe_loss(G_phi, G).mean()
    #         total_loss = loss_policy + loss_efe

    #         # Backpropagation
    #         optimizer_policy.zero_grad()
    #         optimizer_efe.zero_grad()
    #         total_loss.backward()

    #         # Gradient clipping
    #         torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
    #         torch.nn.utils.clip_grad_norm_(efe_network.parameters(), max_norm=1.0)

    #         # Optimizer step
    #         optimizer_policy.step()
    #         optimizer_efe.step()

    #         # Print training loss for monitoring
    #         print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {total_loss.item()}")

    #     # Reset Q_actions for the next epoch
    #     Q_actions_epoch.fill_(0.5)

    #     print("Training completed.")    

    #     # validation loop, complete the validation code here
    #     # During validation, set models to evaluation mode
    #     feature_extractor.eval()
    #     belief_update.eval()
    #     policy_network.eval()
    #     efe_network.eval()

    #     with torch.no_grad():  # Turn off gradients for validation
    #         validation_loss = 0.0
    #         num_validation_batches = len(val_loader)

    #         # Initialize Q_actions with a uniform distribution for validation
    #         Q_actions_val = torch.full((2,), fill_value=0.5, device=device)

    #         true_labels = []
    #         predicted_labels = []

    #         for batch_idx, (image, _, mask) in enumerate(val_loader):
    #             image, mask = image.to(device), mask.to(device)

    #             # Forward pass through the network
    #             features = feature_extractor(image)
    #             pi_current = belief_update(features)

    #             # Compute the target belief for binary classification
    #             target_belief = get_target_belief(mask)

    #             # Predicted class for binary classification
    #             predicted_class = (pi_current[:, 1] > pi_current[:, 0]).long()
    #             true_class = mask.any(dim=1).any(dim=1).any(dim=1).long()

    #             # Append to lists for evaluation
    #             predicted_labels.extend(predicted_class.cpu().numpy())
    #             true_labels.extend(true_class.cpu().numpy())

    #             # Expected Free Energy calculation
    #             q = policy_network(pi_current)
    #             Q_actions_batch = torch.nn.functional.normalize(Q_actions_val.unsqueeze(0).repeat(image.size(0), 1), p=1, dim=1)
    #             G_phi = efe_network(torch.cat([pi_current, q], dim=1))
    #             G = compute_actual_efe(pi_current, target_belief)

    #             # Loss calculation
    #             loss_policy = policy_loss(compute_free_energy(q, Q_actions_batch)).mean()
    #             loss_efe = efe_loss(G_phi, G).mean()
    #             total_loss = loss_policy + loss_efe

    #             validation_loss += total_loss.item()

    #             # Optionally, print validation loss
    #             print(f"Validation - Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {total_loss.item()}")

    #         average_validation_loss = validation_loss / num_validation_batches
    #         print(f"Average Validation Loss for Epoch {epoch+1}: {average_validation_loss}")

    #         # Calculate metrics
    #         accuracy = accuracy_score(true_labels, predicted_labels)
    #         precision = precision_score(true_labels, predicted_labels, average='macro')
    #         recall = recall_score(true_labels, predicted_labels, average='macro')
    #         f1 = f1_score(true_labels, predicted_labels, average='macro')

    #         print(f"Validation Metrics - Epoch {epoch+1}:")
    #         print(f"Accuracy: {accuracy:.4f}")
    #         print(f"Precision: {precision:.4f}")
    #         print(f"Recall: {recall:.4f}")
    #         print(f"F1-Score: {f1:.4f}")



