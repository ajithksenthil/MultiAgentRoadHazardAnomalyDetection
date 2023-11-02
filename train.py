# contains the training loop for the DAI algorithm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Add more imports as necessary, for example, for your dataset, models, etc.
from models import PolicyNetwork, EFENetwork
from data_loader import StreetHazardsDataset, transform
from models import CNNEncoder, PolicyNetwork, BootstrappedEFENetwork

import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.image_feature_extractor = models.resnet18(pretrained=True)
        self.image_feature_extractor = nn.Sequential(*list(self.image_feature_extractor.children())[:-1])  # Remove the last FC layer

    def forward(self, image, mask):
        image_features = self.image_feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        
        mask_hist = []
        for m in mask:
            hist = torch.histc(m.float(), bins=13, min=0, max=12)  # 13 classes
            mask_hist.append(hist)
        mask_features = torch.stack(mask_hist)
        
        combined_features = torch.cat([image_features, mask_features], dim=1)
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

def compute_free_energy(q, Q):
    entropy_q = compute_entropy(q)
    kl_term = torch.sum(q * torch.log((q + 1e-10) / (Q + 1e-10)), dim=1)
    return -entropy_q - kl_term

def policy_loss(F):
    return F

def efe_loss(G_phi, G):
    return torch.norm(G_phi - G, p=2)

def get_observation(action, image, mask):
    h, w = image.shape[2], image.shape[3]
    if action == 0:  # top-left quadrant
        roi_image = image[:, :, :h//2, :w//2]
        roi_mask = mask[:, :, :h//2, :w//2]
    elif action == 1:  # top-right quadrant
        roi_image = image[:, :, :h//2, w//2:]
        roi_mask = mask[:, :, :h//2, w//2:]
    # ... (similarly for other quadrants)
    
    observation = feature_extractor(roi_image, roi_mask)
    return observation


def update_belief(current_belief, observation):
    updated_belief = belief_update(observation)
    # Combine the current belief and the new belief (you can adjust the weights)
    combined_belief = 0.7 * current_belief + 0.3 * updated_belief
    return combined_belief


def compute_actual_efe(current_belief, target_belief):
    # Assuming EFE is the KL divergence between current and target beliefs
    # This is a simple example; in practice, EFE can be more complex.
    kl_div = nn.KLDivLoss()
    efe = kl_div(torch.log(current_belief + 1e-10), target_belief)
    return efe


def get_target_belief(action, mask):
    """
    Given an action and a mask, compute the target belief for that action.
    
    Parameters:
    - action (int): The action taken, determining the region of the image.
    - mask (torch.Tensor): The mask (annotation) image.
    
    Returns:
    - target_belief (torch.Tensor): The computed belief (distribution of classes) for the action's region.
    """
    h, w = mask.shape[2], mask.shape[3]
    
    if action == 0:  # top-left quadrant
        roi_mask = mask[:, :, :h//2, :w//2]
    elif action == 1:  # top-right quadrant
        roi_mask = mask[:, :, :h//2, w//2:]
    elif action == 2:  # bottom-left quadrant
        roi_mask = mask[:, :, h//2:, :w//2]
    elif action == 3:  # bottom-right quadrant
        roi_mask = mask[:, :, h//2:, w//2:]
    else:
        raise ValueError("Invalid action!")
    
    # Count the number of pixels for each class in the region of interest
    class_counts = torch.bincount(roi_mask.view(-1), minlength=13)  # Assuming 13 classes
    target_belief = class_counts.float() / class_counts.sum()
    
    return target_belief


if __name__ == "__main__":
    
    # Hyperparameters
    input_dim = ...  # Define based on the dataset
    hidden_dim = 256
    output_dim = ...  # Define based on the dataset
    learning_rate = 0.001
    num_epochs = 100
    # ... add more hyperparameters as needed
    
    # Initialize networks and optimizers
    cnn_encoder = CNNEncoder()
    policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)
    efe_network = EFENetwork(input_dim, hidden_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_network = policy_network.to(device)
    efe_network = efe_network.to(device)



    optimizer_policy = optim.Adam(policy_network.parameters(), lr=learning_rate)
    optimizer_efe = optim.Adam(efe_network.parameters(), lr=learning_rate)
    
    # Create dataset
    image_dir = 'train/images/training/t1-3'
    annotation_dir = 'train/annotations/training/t1-3'
    dataset = StreetHazardsDataset(image_dir, annotation_dir, transform=transform)

    # Create data loader
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    feature_extractor = FeatureExtractor()
    belief_update = BeliefUpdateNetwork(input_dim, hidden_dim, output_dim)  # Define appropriate dimensions

    # Compute Q from the training data
    class_counts = torch.zeros(13)  # Assuming 13 classes
    for _, mask in data_loader:
        class_counts += torch.bincount(mask.view(-1), minlength=13)
    Q = class_counts / class_counts.sum()


    for epoch in range(num_epochs):
        for image, mask in data_loader:  # assuming data contains the required inputs
            image, mask = image.to(device), mask.to(device)
            
            # 1. Extract features
            features = feature_extractor(image, mask)

            # 1. Get the posterior belief
            # pi_prev = cnn_encoder(image)  # Pass image through CNN encoder
            pi_prev = belief_update(features)

            # 2. Determine action using policy network
            q = policy_network(pi_prev)

            # Sample an action based on q
            action = torch.argmax(q, dim=1)  

            # 3. Get the observation
            observation = get_observation(action, image, mask)

            # 4. Update belief based on observation
            pi_current = update_belief(pi_prev, observation)

            # 5. Predict EFE using EFE network
            G_phi = efe_network(pi_current, action)

            # 6. Compute actual EFE
            target_belief = get_target_belief(action, mask)
            G = compute_actual_efe(pi_current, target_belief)

            # 7. Compute losses
            F = compute_free_energy(q, Q)  # Q needs to be defined based on your data/model
            loss_policy = policy_loss(F)
            loss_efe = efe_loss(G_phi, G)

            # 8. Backpropagation and optimization
            optimizer_policy.zero_grad()
            optimizer_efe.zero_grad()
            
            loss_policy.backward()
            loss_efe.backward()
            
            optimizer_policy.step()
            optimizer_efe.step()
        # TODO: Add logging, validation, and other necessary components