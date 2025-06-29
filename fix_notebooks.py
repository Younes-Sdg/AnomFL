import json
import os

def add_reproducibility_cell(notebook):
    """Add reproducibility cell after imports"""
    # Find the imports cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'import' in source and 'torch' in source:
                # Insert reproducibility cell after imports
                reproducibility_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# --- Reproducibility Cell ---\n",
                        "# Add this cell right after the imports in both notebooks\n",
                        "\n",
                        "import random\n",
                        "import numpy as np\n",
                        "import torch\n",
                        "\n",
                        "# Set random seeds for reproducibility\n",
                        "SEED = 42\n",
                        "random.seed(SEED)\n",
                        "np.random.seed(SEED)\n",
                        "torch.manual_seed(SEED)\n",
                        "torch.cuda.manual_seed(SEED)\n",
                        "torch.cuda.manual_seed_all(SEED)  # if using multi-GPU\n",
                        "torch.backends.cudnn.deterministic = True\n",
                        "torch.backends.cudnn.benchmark = False\n",
                        "\n",
                        "# Set Python hash seed for additional reproducibility\n",
                        "import os\n",
                        "os.environ['PYTHONHASHSEED'] = str(SEED)\n",
                        "\n",
                        "print(f\"✅ Reproducibility set with seed: {SEED}\")\n",
                        "print(\"All random operations will now be deterministic.\")\n"
                    ]
                }
                notebook['cells'].insert(i + 1, reproducibility_cell)
                print(f"✅ Added reproducibility cell after imports")
                break

def fix_threshold_calculation(notebook):
    """Fix the threshold calculation in the plotting cell"""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Generating Comparison Plots' in source:
                # Replace the plotting cell with corrected version
                cell['source'] = [
                    'print("\\n--- Generating Comparison Plots ---")\n',
                    '\n',
                    '# Helper function to get prediction errors\n',
                    'def get_errors(model, input_tensor, target_tensor):\n',
                    '    with torch.no_grad():\n',
                    '        model.eval()\n',
                    '        predictions = model(input_tensor)\n',
                    '        return torch.mean((target_tensor - predictions)**2, dim=1).numpy()\n',
                    '\n',
                    '# Calculate thresholds using normal data only for better anomaly detection\n',
                    '# Get normal file paths\n',
                    'normal_file_paths = [p for p in file_paths if "anomalous" not in p]\n',
                    '\n',
                    '# Collect normal data for threshold calculation\n',
                    'normal_dfs = [pd.read_csv(p) for p in normal_file_paths]\n',
                    'normal_data_df = pd.concat(normal_dfs, ignore_index=True)\n',
                    'normal_input_data = normal_data_df[input_sensors]\n',
                    'normal_target_data = normal_data_df[[MODEL_CONFIG[\'target_sensor\']]]\n',
                    '\n',
                    'normal_scaled_input = input_scaler.transform(normal_input_data)\n',
                    'normal_scaled_target = target_scaler.transform(normal_target_data)\n',
                    '\n',
                    'normal_input_tensor = torch.tensor(normal_scaled_input, dtype=torch.float32)\n',
                    'normal_target_tensor = torch.tensor(normal_scaled_target, dtype=torch.float32)\n',
                    '\n',
                    '# Calculate thresholds using normal data only\n',
                    'fed_normal_errors = get_errors(federated_model, normal_input_tensor, normal_target_tensor)\n',
                    'cen_normal_errors = get_errors(centralized_model, normal_input_tensor, normal_target_tensor)\n',
                    '\n',
                    'fed_threshold = np.mean(fed_normal_errors) + 3 * np.std(fed_normal_errors)\n',
                    'cen_threshold = np.mean(cen_normal_errors) + 3 * np.std(cen_normal_errors)\n',
                    '\n',
                    'print(f"Federated threshold (normal data): {fed_threshold:.6f}")\n',
                    'print(f"Centralized threshold (normal data): {cen_threshold:.6f}")\n',
                    '\n',
                    '# Generate a plot for each aircraft\n',
                    'for i, file_path in enumerate(file_paths):\n',
                    '    client_id = i + 1\n',
                    '    \n',
                    '    # Load the aircraft\'s full dataset\n',
                    '    client_df = pd.read_csv(file_path)\n',
                    '    timestamps = pd.to_datetime(client_df[\'timestamp\'])\n',
                    '    \n',
                    '    # Prepare input and target data for this client\n',
                    '    client_input_data = client_df[input_sensors]\n',
                    '    client_target_data = client_df[[MODEL_CONFIG[\'target_sensor\']]]\n',
                    '    \n',
                    '    # Scale data using the GLOBAL scalers\n',
                    '    client_input_tensor = torch.tensor(input_scaler.transform(client_input_data), dtype=torch.float32)\n',
                    '    client_target_tensor = torch.tensor(target_scaler.transform(client_target_data), dtype=torch.float32)\n',
                    '\n',
                    '    # Get errors for both models\n',
                    '    fed_errors = get_errors(federated_model, client_input_tensor, client_target_tensor)\n',
                    '    cen_errors = get_errors(centralized_model, client_input_tensor, client_target_tensor)\n',
                    '    \n',
                    '    # Identify anomalies using the thresholds calculated from normal data\n',
                    '    fed_anomalies = fed_errors > fed_threshold\n',
                    '    cen_anomalies = cen_errors > cen_threshold\n',
                    '\n',
                    '    # Create the plot\n',
                    '    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)\n',
                    '    fig.suptitle(f\'Aircraft {client_id}: Federated vs. Centralized Anomaly Detection\', fontsize=18)\n',
                    '\n',
                    '    # Federated Plot\n',
                    '    ax1.plot(timestamps, fed_errors, label=\'MSE\', color=\'crimson\')\n',
                    '    ax1.axhline(fed_threshold, color=\'navy\', linestyle=\'--\', label=f\'Threshold ({fed_threshold:.4f})\')\n',
                    '    ax1.scatter(timestamps[fed_anomalies], fed_errors[fed_anomalies], s=60, color=\'gold\', edgecolor=\'black\', label=\'Anomalous Points\', zorder=5)\n',
                    '    ax1.set_title(\'Federated Model\')\n',
                    '    ax1.set_ylabel("Prediction Error (MSE)"); ax1.set_xlabel("Time")\n',
                    '    ax1.grid(True, which=\'both\', linestyle=\':\'); ax1.legend()\n',
                    '    ax1.xaxis.set_major_formatter(mdates.DateFormatter(\'%H:%M\'))\n',
                    '\n',
                    '    # Centralized Plot\n',
                    '    ax2.plot(timestamps, cen_errors, label=\'MSE\', color=\'crimson\')\n',
                    '    ax2.axhline(cen_threshold, color=\'navy\', linestyle=\'--\', label=f\'Threshold ({cen_threshold:.4f})\')\n',
                    '    ax2.scatter(timestamps[cen_anomalies], cen_errors[cen_anomalies], s=60, color=\'gold\', edgecolor=\'black\', label=\'Anomalous Points\', zorder=5)\n',
                    '    ax2.set_title(\'Centralized Model\')\n',
                    '    ax2.set_xlabel("Time")\n',
                    '    ax2.grid(True, which=\'both\', linestyle=\':\'); ax2.legend()\n',
                    '    ax2.xaxis.set_major_formatter(mdates.DateFormatter(\'%H:%M\'))\n',
                    '    \n',
                    '    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()\n'
                ]
                print("✅ Fixed threshold calculation in plotting cell")
                break

def fix_notebook(notebook_path):
    """Fix a single notebook"""
    print(f"\n🔧 Fixing {notebook_path}...")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return False
    
    # Add reproducibility cell
    add_reproducibility_cell(notebook)
    
    # Fix threshold calculation
    fix_threshold_calculation(notebook)
    
    # Write the fixed notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Fixed notebook saved: {notebook_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing notebook: {e}")
        return False

def main():
    """Fix both linear regression notebooks"""
    notebooks = [
        'examples/linear_regression_federated_vs_centralized_analysis.ipynb',
        'examples/linear_regression_federated_vs_centralized_analysis_all_data.ipynb'
    ]
    
    print("🚀 Starting notebook fixes...")
    print("📝 Changes to be applied:")
    print("   1. Add reproducibility cell with seed=42")
    print("   2. Fix threshold calculation to use normal data only")
    print("   3. This should restore anomaly detection spikes in federated learning")
    
    success_count = 0
    for notebook_path in notebooks:
        if os.path.exists(notebook_path):
            if fix_notebook(notebook_path):
                success_count += 1
        else:
            print(f"❌ Notebook not found: {notebook_path}")
    
    print(f"\n✅ Successfully fixed {success_count}/{len(notebooks)} notebooks")
    print("🎯 The federated learning models should now properly detect anomalies!")
    print("   Run the notebooks again to see the corrected results.")

if __name__ == "__main__":
    main() 