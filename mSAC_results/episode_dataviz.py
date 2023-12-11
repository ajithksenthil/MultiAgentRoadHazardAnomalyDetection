import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_and_save(data_path, output_folder):
    # Read data from CSV
    df = pd.read_csv(data_path)

    # List of metrics to plot
    metrics = ["episode","hazard_type","rewards","collisions","hazard_encounters"]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Plot each metric
    for metric in metrics:
        plt.figure()
        plt.plot(df['episode'], df[metric], label=metric)
        plt.xlabel('Episode')
        plt.ylabel(metric.capitalize())
        plt.title(f'Episode vs. {metric.capitalize()}')
        plt.legend()
        plt.grid(True)

        # Save plot as PNG
        plt.savefig(f'{output_folder}/{metric}_plot.png')

    print("Plots saved successfully.")

# Path to the CSV file and output folder
data_path = 'episode_data-3 copy.csv'
output_folder = 'episode_plots2'

# Call the function
visualize_and_save(data_path, output_folder)
