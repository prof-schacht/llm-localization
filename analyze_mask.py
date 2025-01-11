import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def analyze_mask(mask_path, model_name, percentage, pooling=None, foundation=None):
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    # Load the mask
    mask = np.load(mask_path)
    num_layers, hidden_dim = mask.shape
    
    # Calculate percentage of active units per layer
    active_units_per_layer = mask.sum(axis=1)
    percentage_per_layer = (active_units_per_layer / hidden_dim) * 100
    
    # Create the visualization
    plt.figure(figsize=(6, 10))
    
    # Create heatmap-style visualization
    data = percentage_per_layer.reshape(-1, 1)
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Percentage of Active Units'})
    
    # Build title components
    title_parts = [f'Distribution of Active Units\nModel: {model_name}']
    if percentage is not None:
        title_parts.append(f'Percentage: {percentage}%')
    if pooling is not None:
        title_parts.append(f'Pooling: {pooling}')
    if foundation is not None:
        title_parts.append(f'Foundation: {foundation}')
    
    # Customize the plot
    plt.title('\n'.join(title_parts))
    plt.xlabel('Percentage')
    plt.ylabel('Layer')
    
    # Add text annotations
    for i in range(len(percentage_per_layer)):
        plt.text(0.5, i + 0.5, f'{percentage_per_layer[i]:.1f}%', 
                ha='center', va='center', color='white')
    
    # Build filename components
    filename_parts = [f'layer_distribution_{model_name}']
    if percentage is not None:
        filename_parts.append(f'perc={percentage}')
    if pooling is not None:
        filename_parts.append(f'pooling={pooling}')
    if foundation is not None:
        filename_parts.append(f'foundation={foundation}')
    
    # Save the plot
    plt.tight_layout()
    plot_filename = '_'.join(filename_parts) + '.png'
    plot_path = os.path.join('./plots', plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {plot_path}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Model: {model_name}")
    print(f"Total number of layers: {num_layers}")
    print(f"Hidden dimension size: {hidden_dim}")
    print(f"Total active units: {int(mask.sum())}")
    print("\nPer-layer breakdown:")
    for layer in range(num_layers):
        print(f"Layer {layer}: {int(active_units_per_layer[layer])} units ({percentage_per_layer[layer]:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze unit distribution across layers')
    parser.add_argument('--mask-path', type=str, required=True, help='Path to the .npy mask file')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model')
    parser.add_argument('--percentage', type=float, help='Percentage of active units')
    parser.add_argument('--pooling', type=str, help='Pooling method')
    parser.add_argument('--foundation', type=str, help='Foundation')
    args = parser.parse_args()
    
    analyze_mask(args.mask_path, args.model_name, args.percentage, args.pooling, args.foundation) 