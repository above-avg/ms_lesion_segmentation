import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet, AttentionUnet
from monai.losses import TverskyLoss
from monai.transforms import Compose, Resize, EnsureChannelFirst, EnsureType
import glob
import numpy as np
import os
import sys
from tqdm import tqdm

# --- PHASE 2: SCIENTIFIC DATA SPLITTING ---
class MSResearchDataset(Dataset):
    def __init__(self, root, patient_ids):
        self.imgs, self.masks = [], []
        # Filter: Only use slices from these specific patients that HAVE lesions
        for pid in patient_ids:
            p_imgs = sorted(glob.glob(f"{root}/images/p{pid}_s*.npy"))
            p_masks = sorted(glob.glob(f"{root}/masks/p{pid}_s*.npy"))
            for i, m in zip(p_imgs, p_masks):
                if np.load(m).sum() > 0:
                    self.imgs.append(i)
                    self.masks.append(m)

        self.img_trans = Compose([EnsureChannelFirst(channel_dim="no_channel"), Resize((256, 256), mode="bilinear"), EnsureType()])
        self.mask_trans = Compose([EnsureChannelFirst(channel_dim="no_channel"), Resize((256, 256), mode="nearest"), EnsureType()])

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.img_trans(np.load(self.imgs[i])), self.mask_trans(np.load(self.masks[i]))

# --- PHASE 1: MODEL DEFINITIONS ---
def get_model(model_type="baseline"):
    configs = dict(spatial_dims=2, in_channels=1, out_channels=1, 
                   channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    if model_type == "attention":
        print("Creating Attention U-Net (Research Extension)...")
        return AttentionUnet(**configs).to("cuda")
    else:
        print("Creating Standard U-Net (Baseline)...")
        return UNet(**configs, num_res_units=2).to("cuda")

# --- TRAINING ENGINE ---
def run_train(model_type):
    device = torch.device("cuda")
    model = get_model(model_type)
    
    # Split 60 patients: 1-48 Train (80%), 49-60 Val (20%)
    all_pids = [str(i) for i in range(1, 61)]
    train_ds = MSResearchDataset("processed_2d", all_pids[:48])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc=f"{model_type.upper()} Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(x)
                loss = bce(output, y) + criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Avg Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"ms_{model_type}.pth")
    print(f"Model saved as ms_{model_type}.pth")

# --- PHASE 3: EVALUATION ---
def run_eval():
    print("Evaluating models on unseen patients...")
    # Add evaluation logic here to compare Dice scores (as discussed previously)
    pass

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_research.py [baseline|attention|eval]")
    else:
        mode = sys.argv[1].lower()
        if mode == "baseline":
            run_train("baseline")
        elif mode == "attention":
            run_train("attention")
        elif mode == "eval":
            run_eval()
        else:
            print("Invalid mode. Use 'baseline', 'attention', or 'eval'.")