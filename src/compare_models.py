import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from monai.networks.nets import UNet, AttentionUnet
from monai.losses import TverskyLoss
from monai.transforms import Compose, Resize, EnsureChannelFirst, EnsureType
from monai.metrics import DiceMetric

# --- PHASE 2: SCIENTIFIC DATASET (Patient-Wise Split) ---
class MSComparisonDataset(Dataset):
    def __init__(self, root, patient_ids):
        self.imgs, self.masks = [], []
        for pid in patient_ids:
            p_imgs = sorted(glob.glob(f"{root}/images/p{pid}_s*.npy"))
            p_masks = sorted(glob.glob(f"{root}/masks/p{pid}_s*.npy"))
            for i, m in zip(p_imgs, p_masks):
                # We filter for lesions to speed up learning for all models
                if np.load(m).sum() > 0:
                    self.imgs.append(i)
                    self.masks.append(m)

        self.trans = Compose([EnsureChannelFirst(channel_dim="no_channel"), Resize((256, 256)), EnsureType()])

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.trans(np.load(self.imgs[i])), self.trans(np.load(self.masks[i]))

# --- PHASE 1: MODEL FACTORY ---
def get_model(name="unet"):
    base_params = dict(spatial_dims=2, in_channels=1, out_channels=1, 
                       channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    
    if name == "unet":
        return UNet(**base_params, num_res_units=0).to("cuda")
    elif name == "resunet":
        # U-Net with Residual units (often better convergence)
        return UNet(**base_params, num_res_units=2).to("cuda")
    elif name == "attention_unet":
        return AttentionUnet(**base_params).to("cuda")
    else:
        raise ValueError("Unknown model name")

# --- TRAINING & EVALUATION ENGINE ---
def run_experiment(model_name, epochs=30):
    device = torch.device("cuda")
    model = get_model(model_name)
    
    # Split 60 patients: 1-48 Train, 49-60 Val
    all_pids = [str(i) for i in range(1, 61)]
    train_ds = MSComparisonDataset("processed_2d", all_pids[:48])
    val_ds = MSComparisonDataset("processed_2d", all_pids[48:])
    
    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Hybrid Loss for small lesion stability
    criterion = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print(f"\n--- Starting Experiment: {model_name.upper()} ---")
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(x)
                loss = bce(output, y) + criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # PHASE 3: QUANTITATIVE EVALUATION
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_out = (torch.sigmoid(model(val_x)) > 0.5).float()
            dice_metric(y_pred=val_out, y=val_y)
    
    final_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"FINISHED {model_name}. Final Validation Dice: {final_dice:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), f"ms_{model_name}.pth")
    return final_dice

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    results = {}
    
    # Compare all 3 major architectures
    for m_name in ["unet", "resunet", "attention_unet"]:
        try:
            dice_score = run_experiment(m_name, epochs=30)
            results[m_name] = dice_score
        except Exception as e:
            print(f"Error training {m_name}: {e}")

    # Display Comparison Table
    print("\n" + "="*30)
    print("FINAL RESEARCH COMPARISON")
    print("="*30)
    df = pd.DataFrame(list(results.items()), columns=['Model', 'Mean Dice Score'])
    print(df.to_string(index=False))
    df.to_csv("research_comparison_results.csv")