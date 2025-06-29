# --- Reproducibility Cell ---
# Add this cell right after the imports in both notebooks

import random
import numpy as np
import torch

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set Python hash seed for additional reproducibility
import os
os.environ['PYTHONHASHSEED'] = str(SEED)

print(f"✅ Reproducibility set with seed: {SEED}")
print("All random operations will now be deterministic.") 