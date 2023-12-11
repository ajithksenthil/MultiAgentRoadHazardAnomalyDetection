import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_and_save_metrics(csv_file_path, output_folder):
    # Read the CSV data
    df = pd.read_csv(csv_file_path)

    # Define the metrics to plot
    metrics = ['Val Loss', 'Pixel Accuracy', 'mIoU', 'AUPR', 'FPR95', 'AUROC']

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Plot each metric
    for metric in metrics:
        plt.figure()
        plt.plot(df['Epoch'], df[metric], marker='o', linestyle='-')
        plt.title(f'{metric} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)

        # Save the plot
        plt.savefig(os.path.join(output_folder, f'{metric.replace(" ", "_")}_plot.png'))

    print("All plots saved successfully.")

# Path to your CSV file and the output directory
csv_file_path = 'validation_metrics-2.csv'  # Update with your CSV file path
output_folder = 'visualization_output'

# Generate and save the plots
visualize_and_save_metrics(csv_file_path, output_folder)
