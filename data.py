import torch
from torch.utils.data import Dataset
import glob
import os
import pandas as pd

class PatchFeaturesDataset(Dataset):
    def __init__(self, patch_features_dir, csv_label_file):
        label_df = pd.read_csv(csv_label_file)
        self.patch_features_files = [os.path.join(patch_features_dir, filename) for filename in label_df.iloc[:, 0].values]
        self.labels = label_df.iloc[:, 1].values
        

    def __len__(self):
        return len(self.patch_features_files)
    
    def __getitem__(self, idx):
        patch_features = torch.load(self.patch_features_files[idx]).to(torch.float32)
        # Handle different data formats
        if patch_features.ndim == 3:
            patch_features = patch_features.squeeze(0)
        
        return {
            'patch_features': patch_features,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
