# replay_buffer.py
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffers = [[] for _ in range(num_agents)]
        self.positions = [0 for _ in range(num_agents)]

    def push(self, states, actions, rewards, next_states, dones):
        # states, actions, rewards, next_states, and dones are expected to be lists of lists
        # where each inner list corresponds to an agent's experiences
        for i in range(self.num_agents):
            if len(self.buffers[i]) < self.capacity:
                self.buffers[i].append(None)
            self.buffers[i][self.positions[i]] = (states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.positions[i] = (self.positions[i] + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of transitions for each agent from their respective buffers
        batches = []
        for i in range(self.num_agents):
            batch = random.sample(self.buffers[i], batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            batches.append((state, action, reward, next_state, done))
        return batches

    def __len__(self):
        return min(len(buffer) for buffer in self.buffers)

"""
Enhancements:
- Multi-Agent Compatibility: The buffer now maintains separate lists of experiences for each agent.
- Push Method: It now accepts lists of experiences for each agent and updates their respective buffers.
- Sample Method: It samples a batch of transitions for each agent, enabling parallel updates of multiple agents.
- Length Method: Returns the length of the shortest buffer to ensure consistent sampling.

Usage Considerations:
- When storing transitions, provide lists of states, actions, rewards, next_states, and dones for all agents.
- During training, sample from the buffer and update each agent's parameters using their respective batch of experiences.
- Ensure that the environment's step function returns experiences in a format compatible with this buffer structure.

This replay buffer is now ready to be integrated with the rest of your multi-agent SAC setup in CARLA.
"""

"""
Explanation:
Initialization: The ReplayBuffer is initialized with a specific capacity.
Push Method: Adds a new transition to the buffer. If the buffer is full, it starts overwriting the oldest transitions.
Sample Method: Randomly samples a batch of transitions from the buffer. This method is crucial for breaking the correlation between consecutive learning samples.
Length Method: Returns the current number of elements in the buffer.
Usage:
Storing Transitions: After each step in the environment, you store the transition (state, action, reward, next_state, done) in the replay buffer using the push method.
Training: When updating the agent's parameters, you sample a batch of transitions from the buffer.
Considerations:
Efficiency: For large-scale applications, consider optimizing the buffer for memory efficiency (e.g., using a more compact data format).
Balanced Sampling: In some scenarios, you might want to implement mechanisms to ensure a balanced sampling of different types of experiences (e.g., prioritized replay buffer).
This replay buffer implementation should integrate well with the rest of your mSAC setup.
"""