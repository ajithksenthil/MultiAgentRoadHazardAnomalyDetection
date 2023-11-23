
import torch
import torch.nn.functional as F



# def compute_entropy(pi):
#     return -torch.sum(pi * torch.log(pi + 1e-10), dim=1)
def compute_entropy(pi):
    return -torch.sum(pi * torch.log(pi + 1e-10), dim=0)  # Use dim=0 for 1D tensor


def compute_free_energy(q, Q_actions_batch):
    entropy_q = compute_entropy(q)
    kl_term = torch.sum(q * torch.log((q + 1e-10) / (Q_actions_batch + 1e-10)), dim=1)
    return -entropy_q - kl_term


def policy_loss(F):
    return F

def efe_loss(G_phi, G):
    return torch.norm(G_phi - G, p=2)


def test_compute_entropy():
    # Create a mock probability distribution
    pi = torch.tensor([0.25, 0.25, 0.25, 0.25])
    calculated_entropy = compute_entropy(pi)

    # Manual calculation
    expected_entropy = -torch.sum(pi * torch.log(pi + 1e-10))
    assert torch.isclose(calculated_entropy, expected_entropy), "Entropy computation failed"

def test_compute_free_energy():
    # Create mock distributions
    q = torch.tensor([0.4, 0.6])
    Q_actions_batch = torch.tensor([0.5, 0.5])

    calculated_free_energy = compute_free_energy(q, Q_actions_batch)

    # Manual calculation
    entropy_q = -torch.sum(q * torch.log(q + 1e-10))
    kl_term = torch.sum(q * torch.log((q + 1e-10) / (Q_actions_batch + 1e-10)))
    expected_free_energy = -entropy_q - kl_term

    assert torch.isclose(calculated_free_energy, expected_free_energy), "Free energy computation failed"

def test_policy_loss():
    # Mock free energy
    F = torch.tensor(2.0)
    loss = policy_loss(F)
    assert torch.isclose(loss, F), "Policy loss computation failed"

def test_efe_loss():
    # Create mock EFE values
    G_phi = torch.tensor([1.0, 2.0, 3.0])
    G = torch.tensor([1.5, 2.5, 3.5])

    calculated_loss = efe_loss(G_phi, G)
    expected_loss = torch.norm(G_phi - G, p=2)

    assert torch.isclose(calculated_loss, expected_loss), "EFE loss computation failed"

# Run the tests
test_compute_entropy()
test_compute_free_energy()
test_policy_loss()
test_efe_loss()

print("All tests passed!")








# # import torch

# # # Example mask (manually create a mask with known values)
# # test_mask = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# # # Compute histogram
# # hist = torch.histc(test_mask.float(), bins=13, min=0, max=12)

# # # Print the histogram
# # print(f"Test Histogram: {hist}")
# import torch

# # Create a test mask
# # For simplicity, we'll create a 1D tensor with values from 0 to 12
# test_mask = torch.arange(0, 13)

# # Compute the histogram
# hist = torch.histc(test_mask.float(), bins=13, min=0, max=12)

# # Print the test mask and its histogram
# print(f"Test Mask: {test_mask}")
# print(f"Test Histogram: {hist}")
