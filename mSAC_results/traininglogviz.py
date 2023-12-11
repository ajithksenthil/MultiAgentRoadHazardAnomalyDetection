import json
import matplotlib.pyplot as plt
from collections import Counter

def visualize_training_data(json_file_path):
    # Read data from JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extracting average rewards and hazards
    episodes = [entry['episode'] for entry in data]
    average_rewards = [entry['average_reward'] for entry in data]
    hazards = [hazard for entry in data for hazard in entry['hazard_list']]

    # Plotting average rewards
    plt.figure()
    plt.plot(episodes, average_rewards, marker='o', linestyle='-')
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig('average_rewards_plot.png')

    # Plotting hazard distribution
    hazard_count = Counter(hazards)
    plt.figure()
    plt.bar(hazard_count.keys(), hazard_count.values())
    plt.title('Hazard Encounters Distribution')
    plt.xlabel('Hazard Type')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.savefig('hazard_distribution_plot.png')

    print("Plots saved successfully.")

# Path to your JSON file
json_file_path = 'training_log-2.json'  # Update with your JSON file path

# Generate and save the plots
visualize_training_data(json_file_path)
