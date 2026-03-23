import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet
from monai.losses import TverskyLoss
from monai.transforms import Compose, Resize, EnsureChannelFirst, EnsureType
import glob
import numpy as np
import os
from tqdm import tqdm

# Suggested by the error log to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class MS2DDataset(Dataset):
    def __init__(self, root):
        self.imgs = sorted(glob.glob(f"{root}/images/*.npy"))
        self.masks = sorted(glob.glob(f"{root}/masks/*.npy"))
        
        self.img_trans = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=(256, 256), mode="bilinear"),
            EnsureType()
        ])
        
        self.mask_trans = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=(256, 256), mode="nearest"),
            EnsureType()
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = self.img_trans(np.load(self.imgs[i]))
        mask = self.mask_trans(np.load(self.masks[i]))
        return img, mask

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache() # Clear any leftover memory before starting

model = AttentionUnet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256), # Reduced filter counts slightly to save VRAM
    strides=(2, 2, 2, 2),
).to(device)

dataset = MS2DDataset("processed_2d")
# REDUCED BATCH SIZE TO 16
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

criterion = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Scaler for Mixed Precision
scaler = torch.cuda.amp.GradScaler()

print(f"Starting training on {device}...")

for epoch in range(30):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/30")
    
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Runs the forward pass with Mixed Precision
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
        
        # Scales the loss and performs backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    print(f"Epoch {epoch+1} Summary: Avg Loss = {epoch_loss/len(loader):.5f}")

torch.save(model.state_dict(), "ms_attention_unet_optimized.pth")

import matplotlib.pyplot as plt

def debug_prediction(model, dataset, device, index=50):
    model.eval()
    with torch.no_grad():
        img, mask = dataset[index]
        img_in = img.unsqueeze(0).to(device)
        # Get prediction and apply sigmoid
        pred = torch.sigmoid(model(img_in)).cpu().numpy()[0, 0]
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(img[0], cmap='gray'); plt.title("FLAIR Input")
        plt.subplot(1, 3, 2); plt.imshow(mask[0], cmap='hot'); plt.title("Ground Truth (Lesion)")
        plt.subplot(1, 3, 3); plt.imshow(pred, cmap='jet'); plt.title("Model Probability Map")
        plt.colorbar()
        plt.savefig("debug_result.png")
        print("Debug image saved as debug_result.png")
