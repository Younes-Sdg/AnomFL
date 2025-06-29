import json
import os

def create_fixed_notebook():
    # Read the original notebook
    notebook_path = os.path.join('examples', 'federated_vs_centralized_analysis.ipynb')
    
    if not os.path.exists(notebook_path):
        print(f"❌ Original notebook not found: {notebook_path}")
        return
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return
    
    # Find and fix Cell 3 (centralized training)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training Centralized Model' in source:
                cell['source'] = [
                    'print("\\n--- Training Centralized Model ---")\n',
                    '\n',
                    '# Load data from ALL aircraft (both normal and anomalous)\n',
                    'all_file_paths = file_paths\n',
                    'print(f"Loading data from {len(all_file_paths)} aircraft files:")\n',
                    'for i, path in enumerate(all_file_paths):\n',
                    '    print(f"  Aircraft {i+1}: {os.path.basename(path)}")\n',
                    '\n',
                    '# Load all data files\n',
                    'all_dfs = []\n',
                    'for path in all_file_paths:\n',
                    '    df = pd.read_csv(path)\n',
                    '    all_dfs.append(df)\n',
                    '    print(f"✅ Loaded {len(df)} records from {os.path.basename(path)}")\n',
                    '\n',
                    '# Combine all data\n',
                    'all_data_df = pd.concat(all_dfs, ignore_index=True)\n',
                    'print(f"📊 Total combined data: {len(all_data_df)} records")\n',
                    '\n',
                    '# Extract sensor data\n',
                    'sensor_data = all_data_df[DATA_CONFIG[\'sensor_names\']]\n',
                    'print(f"📊 Sensor data shape: {sensor_data.shape}")\n',
                    '\n',
                    '# Fit centralized scaler on all the data\n',
                    'centralized_scaler = MinMaxScaler()\n',
                    'scaled_data = centralized_scaler.fit_transform(sensor_data)\n',
                    'centralized_train_tensor = torch.tensor(scaled_data, dtype=torch.float32)\n',
                    'print(f"📊 Centralized training tensor shape: {centralized_train_tensor.shape}")\n',
                    '\n',
                    '# Instantiate and train the centralized model\n',
                    'centralized_model = Autoencoder(\n',
                    '    input_dim=MODEL_CONFIG[\'input_dim\'],\n',
                    '    hidden_dim=MODEL_CONFIG[\'hidden_dim\'],\n',
                    '    latent_dim=MODEL_CONFIG[\'latent_dim\']\n',
                    ')\n',
                    '\n',
                    'print("🚀 Starting centralized training...")\n',
                    'centralized_model.train_on_tensor(\n',
                    '    centralized_train_tensor,\n',
                    '    num_epochs=TRAINING_CONFIG[\'centralized_epochs\'],\n',
                    '    lr=TRAINING_CONFIG[\'learning_rate\']\n',
                    ')\n',
                    '\n',
                    '# Calculate and print final MSE statistic\n',
                    'centralized_model.eval()\n',
                    'with torch.no_grad():\n',
                    '    reconstructed = centralized_model(centralized_train_tensor)\n',
                    '    final_loss = nn.MSELoss()(reconstructed, centralized_train_tensor)\n',
                    'print(f"\\n📊 Final MSE on All Training Data (Centralized): {final_loss.item():.8f}")\n'
                ]
                print("✅ Fixed Cell 3 - Centralized model with its own scaler")
                break
    
    # Find and fix Cell 4 (federated training) - add federated scaler
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training Federated Model' in source:
                # Find where federated training ends and add federated scaler setup
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if 'federated_model = server.global_model' in line:
                        # Add federated scaler setup after federated training
                        new_source.extend([
                            '\n',
                            '# Setup federated scaler using all federated training data\n',
                            'print("\\n📊 Setting up federated scaler...")\n',
                            'federated_scaler = MinMaxScaler()\n',
                            '\n',
                            '# Collect all data used in federated training\n',
                            'federated_dfs = []\n',
                            'for path in file_paths:\n',
                            '    df = pd.read_csv(path)\n',
                            '    federated_dfs.append(df)\n',
                            '\n',
                            '# Fit federated scaler on all federated data\n',
                            'federated_all_data = pd.concat(federated_dfs, ignore_index=True)\n',
                            'federated_sensor_data = federated_all_data[DATA_CONFIG[\'sensor_names\']]\n',
                            'federated_scaler.fit(federated_sensor_data)\n',
                            'print(f"✅ Federated scaler fitted on {len(federated_all_data)} records")\n'
                        ])
                        break
                
                cell['source'] = new_source
                print("✅ Fixed Cell 4 - Added federated scaler setup")
                break
    
    # Find and fix Cell 5 (plotting) - use separate scalers and proper thresholds
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Generating Comparison Plots' in source:
                cell['source'] = [
                    'print("\\n--- Generating Comparison Plots ---")\n',
                    '\n',
                    '# Helper function to get reconstruction errors\n',
                    'def get_errors(model, data_tensor):\n',
                    '    with torch.no_grad():\n',
                    '        model.eval()\n',
                    '        reconstructed = model(data_tensor)\n',
                    '        return torch.mean((data_tensor - reconstructed)**2, dim=1).numpy()\n',
                    '\n',
                    '# Generate a plot for each aircraft\n',
                    'for i, file_path in enumerate(file_paths):\n',
                    '    client_id = i + 1\n',
                    '    \n',
                    '    # Load the aircraft\'s full dataset\n',
                    '    client_df = pd.read_csv(file_path)\n',
                    '    timestamps = pd.to_datetime(client_df[\'timestamp\'])\n',
                    '    \n',
                    '    # Scale data using the appropriate scaler for each model\n',
                    '    client_sensor_data = client_df[DATA_CONFIG[\'sensor_names\']]\n',
                    '    \n',
                    '    # For federated model evaluation\n',
                    '    fed_scaled_data = federated_scaler.transform(client_sensor_data)\n',
                    '    fed_client_tensor = torch.tensor(fed_scaled_data, dtype=torch.float32)\n',
                    '    \n',
                    '    # For centralized model evaluation\n',
                    '    cen_scaled_data = centralized_scaler.transform(client_sensor_data)\n',
                    '    cen_client_tensor = torch.tensor(cen_scaled_data, dtype=torch.float32)\n',
                    '    \n',
                    '    # Get errors for Federated Model\n',
                    '    fed_errors = get_errors(federated_model, fed_client_tensor)\n',
                    '    \n',
                    '    # Get errors for Centralized Model\n',
                    '    cen_errors = get_errors(centralized_model, cen_client_tensor)\n',
                    '    \n',
                    '    # Calculate thresholds using each model\'s own training data\n',
                    '    # Federated threshold: use federated training data\n',
                    '    fed_training_errors = get_errors(federated_model, torch.tensor(federated_scaler.transform(federated_sensor_data), dtype=torch.float32))\n',
                    '    fed_threshold = np.mean(fed_training_errors) + 3 * np.std(fed_training_errors)\n',
                    '    \n',
                    '    # Centralized threshold: use centralized training data\n',
                    '    cen_training_errors = get_errors(centralized_model, centralized_train_tensor)\n',
                    '    cen_threshold = np.mean(cen_training_errors) + 3 * np.std(cen_training_errors)\n',
                    '    \n',
                    '    # Identify anomalies\n',
                    '    fed_anomalies = fed_errors > fed_threshold\n',
                    '    cen_anomalies = cen_errors > cen_threshold\n',
                    '    \n',
                    '    # Create the plot\n',
                    '    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)\n',
                    '    fig.suptitle(f\'Aircraft {client_id}: Federated vs. Centralized Anomaly Detection\', fontsize=18)\n',
                    '    \n',
                    '    # Federated Plot\n',
                    '    ax1.plot(timestamps, fed_errors, label=\'MSE\', color=\'crimson\')\n',
                    '    ax1.axhline(fed_threshold, color=\'navy\', linestyle=\'--\', label=f\'Threshold ({fed_threshold:.4f})\')\n',
                    '    ax1.scatter(timestamps[fed_anomalies], fed_errors[fed_anomalies], s=60, color=\'gold\', edgecolor=\'black\', label=\'Anomalous Points\', zorder=5)\n',
                    '    ax1.set_title(\'Federated Model\')\n',
                    '    ax1.set_ylabel("Reconstruction Error (MSE)"); ax1.set_xlabel("Time")\n',
                    '    ax1.grid(True, which=\'both\', linestyle=\':\'); ax1.legend()\n',
                    '    ax1.xaxis.set_major_formatter(mdates.DateFormatter(\'%H:%M\'))\n',
                    '    \n',
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
                print("✅ Fixed Cell 5 - Separate scalers and proper thresholds")
                break
    
    # Find and fix Cell 6 (statistics) - use separate scalers
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Model Performance Statistics' in source:
                cell['source'] = [
                    'print("\\n--- Model Performance Statistics ---")\n',
                    '\n',
                    'centralized_stats = []\n',
                    'federated_stats = []\n',
                    '\n',
                    '# Ensure models are in evaluation mode\n',
                    'centralized_model.eval()\n',
                    'federated_model.eval()\n',
                    '\n',
                    'with torch.no_grad():\n',
                    '    for i, file_path in enumerate(file_paths):\n',
                    '        client_id = i + 1\n',
                    '        df = pd.read_csv(file_path)\n',
                    '        sensor_data = df[DATA_CONFIG[\'sensor_names\']]\n',
                    '        n_points = len(sensor_data)\n',
                    '        \n',
                    '        # --- Centralized Model Evaluation ---\n',
                    '        cen_scaled_data = centralized_scaler.transform(sensor_data)\n',
                    '        cen_data_tensor = torch.tensor(cen_scaled_data, dtype=torch.float32)\n',
                    '        reconstructed_centralized = centralized_model(cen_data_tensor)\n',
                    '        mse_centralized = torch.mean((cen_data_tensor - reconstructed_centralized)**2, dim=1)\n',
                    '        centralized_stats.append({\n',
                    '            \'client_id\': client_id,\n',
                    '            \'status\': \'Anomalous\' if "anomalous" in file_path else \'Normal\',\n',
                    '            \'mean_mse\': mse_centralized.mean().item(),\n',
                    '            \'max_mse\': mse_centralized.max().item(),\n',
                    '            \'std_mse\': mse_centralized.std().item(),\n',
                    '            \'n_points\': n_points\n',
                    '        })\n',
                    '        \n',
                    '        # --- Federated Model Evaluation ---\n',
                    '        fed_scaled_data = federated_scaler.transform(sensor_data)\n',
                    '        fed_data_tensor = torch.tensor(fed_scaled_data, dtype=torch.float32)\n',
                    '        reconstructed_federated = federated_model(fed_data_tensor)\n',
                    '        mse_federated = torch.mean((fed_data_tensor - reconstructed_federated)**2, dim=1)\n',
                    '        federated_stats.append({\n',
                    '            \'client_id\': client_id,\n',
                    '            \'status\': \'Anomalous\' if "anomalous" in file_path else \'Normal\',\n',
                    '            \'mean_mse\': mse_federated.mean().item(),\n',
                    '            \'max_mse\': mse_federated.max().item(),\n',
                    '            \'std_mse\': mse_federated.std().item(),\n',
                    '            \'n_points\': n_points\n',
                    '        })\n',
                    '\n',
                    '# Create DataFrames for a clean display\n',
                    'centralized_df = pd.DataFrame(centralized_stats).set_index(\'client_id\')\n',
                    'federated_df = pd.DataFrame(federated_stats).set_index(\'client_id\')\n',
                    '\n',
                    '# Set pandas display options for better readability\n',
                    'pd.set_option(\'display.float_format\', \'{:.6f}\'.format)\n',
                    '\n',
                    'print("\\n--- Centralized Model Performance ---")\n',
                    'display(centralized_df[[\'status\', \'mean_mse\', \'max_mse\', \'std_mse\', \'n_points\']])\n',
                    '\n',
                    'print("\\n--- Federated Model Performance ---")\n',
                    'display(federated_df[[\'status\', \'mean_mse\', \'max_mse\', \'std_mse\', \'n_points\']])\n'
                ]
                print("✅ Fixed Cell 6 - Separate scalers for statistics")
                break
    
    # Write the fixed notebook
    output_path = os.path.join('examples', 'federated_vs_centralized_analysis_proper_scaling.ipynb')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Fixed notebook saved as '{output_path}'")
        print("📝 Key improvements:")
        print("   - Each model has its own scaler")
        print("   - Proper threshold calculation using each model's training data")
        print("   - Anomaly detection should work correctly for both models")
    except Exception as e:
        print(f"❌ Error writing fixed notebook: {e}")

if __name__ == "__main__":
    create_fixed_notebook() 