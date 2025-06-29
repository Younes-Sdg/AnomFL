#!/usr/bin/env python3
"""
Script to fix the all data notebook by:
1. Fixing the threshold calculation to use normal training data
2. Adding missing performance analysis cells
"""

import json
import re

def fix_all_data_notebook():
    """Fix the all data notebook threshold calculation and add missing cells."""
    
    # Read the notebook
    with open('autoencoder_federated_vs_centralized_analysis_all_data.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find the plotting cell and fix the threshold calculation
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix the threshold calculation
            if 'cen_threshold = np.mean(cen_errors)' in source:
                print("Found threshold calculation cell, fixing...")
                
                # Replace the adaptive threshold with normal training data threshold
                old_pattern = r'cen_threshold = np\.mean\(cen_errors\) \+ 3 \* np\.std\(cen_errors\)'
                new_pattern = r'# FIXED: Use normal training data for threshold calculation instead of adaptive threshold\n    normal_training_errors = get_errors(centralized_model, train_tensor)\n    cen_threshold = np.mean(normal_training_errors) + 3 * np.std(normal_training_errors)'
                
                cell['source'] = [re.sub(old_pattern, new_pattern, source)]
                print("✓ Fixed threshold calculation")
    
    # Add missing performance analysis cells
    print("Adding missing performance analysis cells...")
    
    # Performance statistics cell
    performance_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"\\n--- Model Performance Statistics ---\")\n\n",
            "centralized_stats = []\n",
            "federated_stats = []\n\n",
            "# Ensure models are in evaluation mode\n",
            "centralized_model.eval()\n",
            "federated_model.eval()\n\n",
            "# Use the same globally-fitted scaler for a fair comparison\n",
            "global_scaler = scaler\n\n",
            "with torch.no_grad():\n",
            "    for i, file_path in enumerate(file_paths):\n",
            "        client_id = i + 1\n",
            "        df = pd.read_csv(file_path)\n",
            "        sensor_data = df[DATA_CONFIG['sensor_names']]\n",
            "        n_points = len(sensor_data)\n",
            "        \n",
            "        # Scale data using fixed ranges (same as training)\n",
            "        scaler = get_aircraft_scaler(include_anomalies=True)\n",
            "        scaled_data = scaler.transform(sensor_data)\n",
            "        data_tensor = torch.tensor(scaled_data, dtype=torch.float32)\n",
            "        \n",
            "        # --- Centralized Model Evaluation ---\n",
            "        reconstructed_centralized = centralized_model(data_tensor)\n",
            "        # Calculate per-point MSE loss\n",
            "        mse_centralized = torch.mean((data_tensor - reconstructed_centralized)**2, dim=1)\n",
            "        centralized_stats.append({\n",
            "            'client_id': client_id,\n",
            "            'status': 'Anomalous' if \"anomalous\" in file_path else 'Normal',\n",
            "            'mean_mse': mse_centralized.mean().item(),\n",
            "            'max_mse': mse_centralized.max().item(),\n",
            "            'std_mse': mse_centralized.std().item(),\n",
            "            'n_points': n_points\n",
            "        })\n",
            "        \n",
            "        # --- Federated Model Evaluation ---\n",
            "        reconstructed_federated = federated_model(data_tensor)\n",
            "        # Calculate per-point MSE loss\n",
            "        mse_federated = torch.mean((data_tensor - reconstructed_federated)**2, dim=1)\n",
            "        federated_stats.append({\n",
            "            'client_id': client_id,\n",
            "            'status': 'Anomalous' if \"anomalous\" in file_path else 'Normal',\n",
            "            'mean_mse': mse_federated.mean().item(),\n",
            "            'max_mse': mse_federated.max().item(),\n",
            "            'std_mse': mse_federated.std().item(),\n",
            "            'n_points': n_points\n",
            "        })\n\n",
            "# Create DataFrames for a clean display\n",
            "centralized_df = pd.DataFrame(centralized_stats).set_index('client_id')\n",
            "federated_df = pd.DataFrame(federated_stats).set_index('client_id')\n\n",
            "# Set pandas display options for better readability\n",
            "pd.set_option('display.float_format', '{:.6f}'.format)\n\n",
            "print(\"\\n--- Centralized Model Performance ---\")\n",
            "display(centralized_df[['status', 'mean_mse', 'max_mse', 'std_mse', 'n_points']])\n\n",
            "print(\"\\n--- Federated Model Performance ---\")\n",
            "display(federated_df[['status', 'mean_mse', 'max_mse', 'std_mse', 'n_points']])\n"
        ]
    }
    
    # Visualization cell
    visualization_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"\\n--- Visualizing Performance Statistics ---\")\n\n",
            "# Create a figure for the plots\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))\n",
            "fig.suptitle('Model Performance Comparison Across Clients (ALL DATA)', fontsize=18, y=1.03)\n\n",
            "# --- Bar Chart for Mean MSE ---\n",
            "bar_width = 0.35\n",
            "client_ids = centralized_df.index\n",
            "x = np.arange(len(client_ids))\n\n",
            "rects1 = ax1.bar(x - bar_width/2, centralized_df['mean_mse'], bar_width, label='Centralized', color='royalblue')\n",
            "rects2 = ax1.bar(x + bar_width/2, federated_df['mean_mse'], bar_width, label='Federated', color='seagreen')\n\n",
            "ax1.set_ylabel('Mean Squared Error (MSE)')\n",
            "ax1.set_title('Mean MSE per Client')\n",
            "ax1.set_xticks(x)\n",
            "ax1.set_xticklabels([f'Client {id}' for id in client_ids])\n",
            "ax1.legend()\n",
            "ax1.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "ax1.set_yscale('log') # Use log scale for better visibility\n\n",
            "# --- Bar Chart for Max MSE ---\n",
            "rects3 = ax2.bar(x - bar_width/2, centralized_df['max_mse'], bar_width, label='Centralized', color='royalblue')\n",
            "rects4 = ax2.bar(x + bar_width/2, federated_df['max_mse'], bar_width, label='Federated', color='seagreen')\n\n",
            "ax2.set_ylabel('Max Squared Error (MSE)')\n",
            "ax2.set_title('Max (Peak) MSE per Client')\n",
            "ax2.set_xticks(x)\n",
            "ax2.set_xticklabels([f'Client {id}' for id in client_ids])\n",
            "ax2.legend()\n",
            "ax2.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "ax2.set_yscale('log') # Log scale is useful for comparing large spikes\n\n",
            "# Add a visual indicator for the anomalous client\n",
            "for ax in [ax1, ax2]:\n",
            "    for i, client_id in enumerate(client_ids):\n",
            "        # Find which client is anomalous from the dataframe status\n",
            "        if centralized_df.loc[client_id, 'status'] == 'Anomalous':\n",
            "            ax.get_xticklabels()[i].set_color('red')\n",
            "            ax.get_xticklabels()[i].set_weight('bold')\n\n",
            "fig.tight_layout()\n",
            "plt.show()\n"
        ]
    }
    
    # Insert the new cells before the epoch comparison cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Simple Epoch Comparison' in source:
                # Insert before this cell
                notebook['cells'].insert(i, performance_cell)
                notebook['cells'].insert(i + 1, visualization_cell)
                print("✓ Added performance analysis cells")
                break
    
    # Write the fixed notebook
    with open('autoencoder_federated_vs_centralized_analysis_all_data_fixed.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✓ Fixed notebook saved as 'autoencoder_federated_vs_centralized_analysis_all_data_fixed.ipynb'")

if __name__ == "__main__":
    fix_all_data_notebook() 