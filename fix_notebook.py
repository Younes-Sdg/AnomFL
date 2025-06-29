import json
import os

def fix_notebook():
    # Read the original notebook
    notebook_path = os.path.join('examples', 'federated_vs_centralized_analysis.ipynb')
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return
    
    # Find Cell 3 (the centralized training cell)
    cell_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training Centralized Model' in source and 'normal_paths = file_paths' in source:
                # This is Cell 3 - replace it with the fixed version
                cell['source'] = [
                    'print("\\n--- Training Centralized Model ---")\n',
                    '\n',
                    '# Load data from ALL aircraft (both normal and anomalous)\n',
                    'all_file_paths = file_paths\n',
                    'print(f"Loading data from {len(all_file_paths)} aircraft files:")\n',
                    'for i, path in enumerate(all_file_paths):\n',
                    '    print(f"  Aircraft {i+1}: {path}")\n',
                    '\n',
                    '# Verify all files exist before loading\n',
                    'missing_files = []\n',
                    'for path in all_file_paths:\n',
                    '    if not os.path.exists(path):\n',
                    '        missing_files.append(path)\n',
                    '\n',
                    'if missing_files:\n',
                    '    print(f"❌ Missing files: {missing_files}")\n',
                    '    raise FileNotFoundError(f"Some data files are missing: {missing_files}")\n',
                    '\n',
                    '# Load all data files\n',
                    'all_dfs = []\n',
                    'for path in all_file_paths:\n',
                    '    try:\n',
                    '        df = pd.read_csv(path)\n',
                    '        all_dfs.append(df)\n',
                    '        print(f"✅ Loaded {len(df)} records from {os.path.basename(path)}")\n',
                    '    except Exception as e:\n',
                    '        print(f"❌ Error loading {path}: {e}")\n',
                    '        raise\n',
                    '\n',
                    '# Combine all data\n',
                    'all_data_df = pd.concat(all_dfs, ignore_index=True)\n',
                    'print(f"📊 Total combined data: {len(all_data_df)} records")\n',
                    '\n',
                    '# Extract sensor data\n',
                    'sensor_data = all_data_df[DATA_CONFIG[\'sensor_names\']]\n',
                    'print(f"📊 Sensor data shape: {sensor_data.shape}")\n',
                    '\n',
                    '# Fit a single scaler on all the data\n',
                    'scaler = MinMaxScaler()\n',
                    'scaled_data = scaler.fit_transform(sensor_data)\n',
                    'train_tensor = torch.tensor(scaled_data, dtype=torch.float32)\n',
                    'print(f"📊 Training tensor shape: {train_tensor.shape}")\n',
                    '\n',
                    '# Instantiate and train the model\n',
                    'centralized_model = Autoencoder(\n',
                    '    input_dim=MODEL_CONFIG[\'input_dim\'],\n',
                    '    hidden_dim=MODEL_CONFIG[\'hidden_dim\'],\n',
                    '    latent_dim=MODEL_CONFIG[\'latent_dim\']\n',
                    ')\n',
                    '\n',
                    'print("🚀 Starting centralized training...")\n',
                    'centralized_model.train_on_tensor(\n',
                    '    train_tensor,\n',
                    '    num_epochs=TRAINING_CONFIG[\'centralized_epochs\'],\n',
                    '    lr=TRAINING_CONFIG[\'learning_rate\']\n',
                    ')\n',
                    '\n',
                    '# Calculate and print final MSE statistic\n',
                    'centralized_model.eval()\n',
                    'with torch.no_grad():\n',
                    '    reconstructed = centralized_model(train_tensor)\n',
                    '    final_loss = nn.MSELoss()(reconstructed, train_tensor)\n',
                    'print(f"\\n📊 Final MSE on All Training Data (Centralized): {final_loss.item():.8f}")\n'
                ]
                cell_found = True
                print("✅ Fixed Cell 3 in the notebook")
                break
    
    if not cell_found:
        print("❌ Cell 3 not found in the notebook")
        return
    
    # Write the fixed notebook
    output_path = os.path.join('examples', 'federated_vs_centralized_analysis_fixed.ipynb')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Fixed notebook saved as '{output_path}'")
    except Exception as e:
        print(f"❌ Error writing fixed notebook: {e}")

if __name__ == "__main__":
    fix_notebook() 