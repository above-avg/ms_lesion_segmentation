import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet
from monai.losses import TverskyLoss
from monai.transforms import Compose, Resize, EnsureChannelFirst, EnsureType
import glob
import numpy as np
import os
from tqdm import tqdm

# --- 1. Optimized Dataset: Only learn from slices with lesions ---
class MS2DDataset(Dataset):
    def __init__(self, root, filter_empty=True):
        all_imgs = sorted(glob.glob(f"{root}/images/*.npy"))
        all_masks = sorted(glob.glob(f"{root}/masks/*.npy"))
        
        if filter_empty:
            self.imgs, self.masks = [], []
            for i, m in zip(all_imgs, all_masks):
                if np.load(m).sum() > 0: # Only keep slices with actual lesions
                    self.imgs.append(i)
                    self.masks.append(m)
        else:
            self.imgs, self.masks = all_imgs, all_masks

        self.trans = Compose([EnsureChannelFirst(channel_dim="no_channel"), Resize((256, 256)), EnsureType()])

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.trans(np.load(self.imgs[i])), self.trans(np.load(self.masks[i]))

# --- Setup ---
device = torch.device("cuda")
model = AttentionUnet(
    spatial_dims=2, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
).to(device)

dataset = MS2DDataset("processed_2d", filter_empty=True)
loader = DataLoader(dataset, batch_size=8, shuffle=True) # Smaller batch for more updates

# --- 2. Hybrid Loss: BCE forces the model to wake up, Tversky refines ---
tversky_loss = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
bce_loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Higher starting LR
scaler = torch.amp.GradScaler('cuda')

print(f"Training on {len(dataset)} lesion-positive slices...")

for epoch in range(50): # 50 epochs is enough to see a massive drop now
    model.train()
    epoch_loss = 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output = model(x)
            # Combine losses: BCE helps with initial discovery
            loss = bce_loss(output, y) + tversky_loss(output, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.5f}")

torch.save(model.state_dict(), "ms_attention_unet_v2.pth")