class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # Define network layers
        # ...

    def forward(self, state):
        # Forward pass
        # ...

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Define network layers
        # ...

    def forward(self, state, action):
        # Forward pass
        # ...
