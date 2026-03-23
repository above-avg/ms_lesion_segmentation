import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import BasicUNetPlusPlus
from monai.losses import DiceCELoss
from monai.transforms import Compose, RandGaussianNoise, RandBiasField
import glob
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import KFold # Fixed import

# --- 1. SOTA Dataset ---
class MSSOTADataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        self.aug = Compose([
            RandGaussianNoise(prob=0.2),
            RandBiasField(prob=0.3) 
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = torch.from_numpy(np.load(self.files[i]))
        mask = torch.from_numpy(np.load(self.files[i].replace("images", "masks"))).unsqueeze(0)
        return self.aug(img), mask

# --- 2. Training Logic per Fold ---
def train_fold(fold_idx, train_files, val_files):
    device = torch.device("cuda")
    
    # FIX: BasicUNetPlusPlus uses 'features' instead of 'channels'
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=9, 
        out_channels=1,
        deep_supervision=True, 
        features=(16, 32, 64, 128, 256, 16), # Final int is the bottleneck size
    ).to(device)

    loss_function = DiceCELoss(sigmoid=True, squared_pred=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') 

    train_ds = MSSOTADataset(train_files)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

    print(f"\n--- Training Fold {fold_idx+1}/5 ---")
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}", leave=False)
        
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(x)
                # For Deep Supervision, calculate loss for each auxiliary output
                # and average them for a robust gradient
                loss = sum([loss_function(out, y) for out in outputs]) / len(outputs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Summary Fold {fold_idx+1} | Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"sota_fold_{fold_idx+1}.pth")

# --- 3. K-Fold Master Execution ---
if __name__ == "__main__":
    PROCESSED_DIR = "sota_data/images"
    # Get unique patient IDs from filenames
    all_files = os.listdir(PROCESSED_DIR)
    all_patients = sorted(list(set([f.split("_")[0] for f in all_files])))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_patients)):
        train_pids = [all_patients[i] for i in train_idx]
        train_files = []
        for pid in train_pids:
            train_files.extend(glob.glob(f"{PROCESSED_DIR}/{pid}_s*.npy"))
        
        # We define val_files here so you can pass them to an eval function later
        val_pids = [all_patients[i] for i in val_idx]
        val_files = []
        for pid in val_pids:
            val_files.extend(glob.glob(f"{PROCESSED_DIR}/{pid}_s*.npy"))
            
        train_fold(fold, train_files, val_files)