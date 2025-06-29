import json
import os

def restore_original():
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
    
    # Find Cell 3 (the centralized training cell) and restore it to train only on normal data
    cell_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training Centralized Model' in source:
                # Restore to version that loads only NORMAL data (excluding anomalous)
                cell['source'] = [
                    'print("\\n--- Training Centralized Model ---")\n',
                    '\n',
                    '# Load data from NORMAL aircraft only (exclude anomalous)\n',
                    'normal_paths = [p for p in file_paths if "anomalous" not in p]\n',
                    'print(f"Loading data from {len(normal_paths)} normal aircraft files:")\n',
                    'for i, path in enumerate(normal_paths):\n',
                    '    print(f"  Normal Aircraft {i+1}: {os.path.basename(path)}")\n',
                    '\n',
                    'normal_dfs = [pd.read_csv(p) for p in normal_paths]\n',
                    'all_normal_data_df = pd.concat(normal_dfs, ignore_index=True)\n',
                    'sensor_data = all_normal_data_df[DATA_CONFIG[\'sensor_names\']]\n',
                    '\n',
                    '# Fit a single scaler on normal data only\n',
                    'scaler = MinMaxScaler()\n',
                    'scaled_data = scaler.fit_transform(sensor_data)\n',
                    'train_tensor = torch.tensor(scaled_data, dtype=torch.float32)\n',
                    '\n',
                    '# Instantiate and train the model on normal data only\n',
                    'centralized_model = Autoencoder(\n',
                    '    input_dim=MODEL_CONFIG[\'input_dim\'],\n',
                    '    hidden_dim=MODEL_CONFIG[\'hidden_dim\'],\n',
                    '    latent_dim=MODEL_CONFIG[\'latent_dim\']\n',
                    ')\n',
                    'centralized_model.train_on_tensor(\n',
                    '    train_tensor,\n',
                    '    num_epochs=TRAINING_CONFIG[\'centralized_epochs\'],\n',
                    '    lr=TRAINING_CONFIG[\'learning_rate\']\n',
                    ')\n',
                    '\n',
                    '# Calculate and print final MSE statistic on normal data\n',
                    'centralized_model.eval()\n',
                    'with torch.no_grad():\n',
                    '    reconstructed = centralized_model(train_tensor)\n',
                    '    final_loss = nn.MSELoss()(reconstructed, train_tensor)\n',
                    'print(f"\\n📊 Final MSE on Normal Training Data (Centralized): {final_loss.item():.8f}")\n'
                ]
                cell_found = True
                print("✅ Restored Cell 3 to train centralized model on NORMAL data only")
                break
    
    if not cell_found:
        print("❌ Cell 3 not found in the notebook")
        return
    
    # Write the restored notebook
    output_path = os.path.join('examples', 'federated_vs_centralized_analysis_restored.ipynb')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"✅ Restored notebook saved as '{output_path}'")
        print("📝 Note: This version trains centralized model on NORMAL data only")
        print("📝 Both centralized and federated models should show peaks for anomalous clients")
    except Exception as e:
        print(f"❌ Error writing restored notebook: {e}")

if __name__ == "__main__":
    restore_original() 