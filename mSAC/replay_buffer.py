import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


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