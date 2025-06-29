import json
import os

def fix_linear_regression():
    # Read the linear regression notebook
    notebook_path = os.path.join('examples', 'linear_regression_federated_vs_centralized_analysis.ipynb')
    
    if not os.path.exists(notebook_path):
        print(f"❌ Linear regression notebook not found: {notebook_path}")
        return
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return
    
    # Find and fix Cell 4 (centralized training) - train on all data
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training Centralized Model' in source:
                cell['source'] = [
                    'print("\\n--- Training Centralized Model ---")\n',
                    '\n',
                    '# Load data from ALL aircraft (both normal and anomalous)\n',
                    'all_paths = file_paths\n',
                    'print(f"Loading data from {len(all_paths)} aircraft files:")\n',
                    'for i, path in enumerate(all_paths):\n',
                    '    print(f"  Aircraft {i+1}: {os.path.basename(path)}")\n',
                    '\n',
                    'all_dfs = [pd.read_csv(p) for p in all_paths]\n',
                    'all_data_df = pd.concat(all_dfs, ignore_index=True)\n',
                    '\n',
                    '# Prepare input and target data for linear regression\n',
                    'input_sensors = [s for s in DATA_CONFIG[\'sensor_names\'] if s != MODEL_CONFIG[\'target_sensor\']]\n',
                    'input_data = all_data_df[input_sensors]\n',
                    'target_data = all_data_df[[MODEL_CONFIG[\'target_sensor\']]]\n',
                    '\n',
                    '# Scale the data\n',
                    'input_scaler = MinMaxScaler()\n',
                    'target_scaler = MinMaxScaler()\n',
                    'scaled_input = input_scaler.fit_transform(input_data)\n',
                    'scaled_target = target_scaler.fit_transform(target_data)\n',
                    '\n',
                    '# Convert to tensors\n',
                    'input_tensor = torch.tensor(scaled_input, dtype=torch.float32)\n',
                    'target_tensor = torch.tensor(scaled_target, dtype=torch.float32)\n',
                    '\n',
                    'print(f"Input sensors: {input_sensors}")\n',
                    'print(f"Target sensor: {MODEL_CONFIG[\'target_sensor\']}")\n',
                    'print(f"Data shape - Input: {input_tensor.shape}, Target: {target_tensor.shape}")\n',
                    '\n',
                    '# Instantiate and train the model\n',
                    'centralized_model = LinearRegression(\n',
                    '    input_dim=MODEL_CONFIG[\'input_dim\'],\n',
                    '    output_dim=MODEL_CONFIG[\'output_dim\']\n',
                    ')\n',
                    'centralized_model.train_on_tensor(\n',
                    '    input_tensor,\n',
                    '    target_tensor,\n',
                    '    num_epochs=TRAINING_CONFIG[\'centralized_epochs\'],\n',
                    '    lr=TRAINING_CONFIG[\'learning_rate\']\n',
                    ')\n',
                    '\n',
                    '# Calculate and print final MSE statistic\n',
                    'centralized_model.eval()\n',
                    'with torch.no_grad():\n',
                    '    predictions = centralized_model(input_tensor)\n',
                    '    final_loss = nn.MSELoss()(predictions, target_tensor)\n',
                    'print(f"\\n📊 Final MSE on All Training Data (Centralized): {final_loss.item():.8f}")\n'
                ]
                print("✅ Fixed Cell 4 - Centralized model trains on all data")
                break
    
    # Find and fix Cell 6 (plotting) - use proper threshold calculation
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Generating Comparison Plots' in source:
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
                    '# Generate a plot for each aircraft\n',
                    'for i, file_path in enumerate(file_paths):\n',
                    '    client_id = i + 1\n',
                    '    \n',
                    '    # Load the aircraft\'s full dataset\n',
                    '    client_df = pd.read_csv(file_path)\n',
                    '    timestamps = pd.to_datetime(client_df[\'timestamp\'])\n',
                    '    \n',
                    '    # Prepare input and target data for this client\n',
                    '    input_sensors = [s for s in DATA_CONFIG[\'sensor_names\'] if s != MODEL_CONFIG[\'target_sensor\']]\n',
                    '    client_input_data = client_df[input_sensors]\n',
                    '    client_target_data = client_df[[MODEL_CONFIG[\'target_sensor\']]]\n',
                    '    \n',
                    '    # Scale data using the global scalers\n',
                    '    client_scaled_input = input_scaler.transform(client_input_data)\n',
                    '    client_scaled_target = target_scaler.transform(client_target_data)\n',
                    '    \n',
                    '    client_input_tensor = torch.tensor(client_scaled_input, dtype=torch.float32)\n',
                    '    client_target_tensor = torch.tensor(client_scaled_target, dtype=torch.float32)\n',
                    '\n',
                    '    # Get errors for both models\n',
                    '    fed_errors = get_errors(federated_model, client_input_tensor, client_target_tensor)\n',
                    '    cen_errors = get_errors(centralized_model, client_input_tensor, client_target_tensor)\n',
                    '    \n',
                    '    # Calculate thresholds using normal data only for better anomaly detection\n',
                    '    # Get normal file paths\n',
                    '    normal_file_paths = [p for p in file_paths if "anomalous" not in p]\n',
                    '    \n',
                    '    # Collect normal data for threshold calculation\n',
                    '    normal_dfs = [pd.read_csv(p) for p in normal_file_paths]\n',
                    '    normal_data_df = pd.concat(normal_dfs, ignore_index=True)\n',
                    '    normal_input_data = normal_data_df[input_sensors]\n',
                    '    normal_target_data = normal_data_df[[MODEL_CONFIG[\'target_sensor\']]]\n',
                    '    \n',
                    '    normal_scaled_input = input_scaler.transform(normal_input_data)\n',
                    '    normal_scaled_target = target_scaler.transform(normal_target_data)\n',
                    '    \n',
                    '    normal_input_tensor = torch.tensor(normal_scaled_input, dtype=torch.float32)\n',
                    '    normal_target_tensor = torch.tensor(normal_scaled_target, dtype=torch.float32)\n',
                    '    \n',
                    '    # Calculate thresholds using normal data only\n',
                    '    fed_normal_errors = get_errors(federated_model, normal_input_tensor, normal_target_tensor)\n',
                    '    cen_normal_errors = get_errors(centralized_model, normal_input_tensor, normal_target_tensor)\n',
                    '    \n',
                    '    fed_threshold = np.mean(fed_normal_errors) + 3 * np.std(fed_normal_errors)\n',
                    '    cen_threshold = np.mean(cen_normal_errors) + 3 * np.std(cen_normal_errors)\n',
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
                    '    ax1.set_ylabel("Prediction Error (MSE)"); ax1.set_xlabel("Time")\n',
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
                print("✅ Fixed Cell 6 - Proper threshold calculation using normal data")
                break
    
    # Write the fixed notebook
    output_path = os.path.join('examples', 'linear_regression_federated_vs_centralized_analysis_all_data.ipynb')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Fixed linear regression notebook saved as '{output_path}'")
        print("📝 Key changes:")
        print("   - Both models train on all data (normal + anomalous)")
        print("   - Thresholds calculated using normal data only")
        print("   - Linear regression should be better at detecting anomalies")
        print("   - Should show peaks for anomalous clients")
    except Exception as e:
        print(f"❌ Error writing fixed notebook: {e}")

if __name__ == "__main__":
    fix_linear_regression() 