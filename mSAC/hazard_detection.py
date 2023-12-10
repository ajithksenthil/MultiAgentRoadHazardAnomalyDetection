import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

# Now import the required classes from ood_models
from ood_models import CNNEncoder, PolicyNetwork, BootstrappedEFENetwork, BeliefUpdateNetwork, FeatureExtractor

class HazardDetectionSystem:
    def __init__(self):
        # Initialize the models
        self.feature_extractor = FeatureExtractor().to('cuda:1')
        self.belief_update = BeliefUpdateNetwork().to('cuda:1')
        self.policy_network = PolicyNetwork().to('cuda:1')
        self.efe_network = BootstrappedEFENetwork().to('cuda:1')

        # Path to the model in the parent directory TODO
        model_path = '../trained_model.pth'

        # Load the model state dictionaries
        model_states = torch.load(model_path, map_location='cuda:1')
        self.feature_extractor.load_state_dict(model_states['feature_extractor'])
        self.belief_update.load_state_dict(model_states['belief_update'])
        self.policy_network.load_state_dict(model_states['policy_network'])
        self.efe_network.load_state_dict(model_states['efe_network'])

        # Wrap models in DataParallel and move to the correct device
        self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=[1, 0]).to('cuda:1')
        self.belief_update = nn.DataParallel(self.belief_update, device_ids=[1, 0]).to('cuda:1')
        self.policy_network = nn.DataParallel(self.policy_network, device_ids=[1, 0]).to('cuda:1')
        self.efe_network = nn.DataParallel(self.efe_network, device_ids=[1, 0]).to('cuda:1')

        # Set models to evaluation mode
        self.feature_extractor.eval()
        self.belief_update.eval()
        self.policy_network.eval()
        self.efe_network.eval()

        # ... Additional initialization for hazard detection, if necessary ...

# Rest of your script...

