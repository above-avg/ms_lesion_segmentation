import torch
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from monai.networks.nets import BasicUNetPlusPlus
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. Load Ensemble Models ---
device = torch.device("cuda")
models = []
for i in range(1, 6):
    model = BasicUNetPlusPlus(
        spatial_dims=2, in_channels=9, out_channels=1,
        deep_supervision=True, features=(16, 32, 64, 128, 256, 16)
    ).to(device)
    model.load_state_dict(torch.load(f"sota_fold_{i}.pth", weights_only=True))
    model.eval()
    models.append(model)

# --- 2. Data Preparation ---
PROCESSED_DIR = "sota_data"
eval_pids = [str(i) for i in range(49, 61)]
eval_files = [f for f in glob.glob(f"{PROCESSED_DIR}/images/*.npy") if any(f"p{p}_" in os.path.basename(f) for p in eval_pids)]

# --- 3. Threshold Sweep & TLL Collection ---
thresholds = np.arange(0.1, 1.0, 0.1)
sweep_results = []
gt_volumes = []
pred_volumes_at_opt = [] # We'll store volumes for the 0.3 threshold (usually optimal)

# We'll also hunt for a "False Negative" sample
false_negative_sample = None

print("Running Threshold Sweep and TLL Correlation...")
with torch.no_grad():
    # Pre-calculate all probabilities to save time
    all_probs = []
    all_masks = []
    
    for f in tqdm(eval_files):
        x = torch.from_numpy(np.load(f)).unsqueeze(0).to(device)
        y = torch.from_numpy(np.load(f.replace("images", "masks"))).unsqueeze(0).to(device)
        
        # Ensemble Average
        probs = torch.mean(torch.stack([torch.sigmoid(m(x)[0]) for m in models]), dim=0)
        all_probs.append(probs.cpu())
        all_masks.append(y.cpu())

    # --- STEP 1: THRESHOLD SWEEP ---
    for t in thresholds:
        tp, fp, fn, tn = 0, 0, 0, 0
        for p, m in zip(all_probs, all_masks):
            pred = (p > t).float()
            tp += (pred * m).sum().item()
            fp += (pred * (1 - m)).sum().item()
            fn += ((1 - pred) * m).sum().item()
            tn += ((1 - pred) * (1 - m)).sum().item()
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        sweep_results.append({"Threshold": round(t, 1), "Sensitivity": sensitivity, "Dice": dice})

    # --- STEP 2: TLL CORRELATION (at t=0.3) ---
    opt_t = 0.3 
    for p, m in zip(all_probs, all_masks):
        gt_vol = m.sum().item()
        pred_vol = (p > opt_t).float().sum().item()
        gt_volumes.append(gt_vol)
        pred_volumes_at_opt.append(pred_vol)
        
        # --- STEP 3: FALSE NEGATIVE SEARCH ---
        # Find a slice where GT has a lesion but prediction is nearly empty
        if gt_vol > 20 and pred_vol < 2 and false_negative_sample is None:
            false_negative_sample = (p, m)

# --- RESULTS OUTPUT ---
sweep_df = pd.DataFrame(sweep_results)
print("\n--- Threshold Sweep Results ---")
print(sweep_df)

corr, _ = pearsonr(gt_volumes, pred_volumes_at_opt)
print(f"\nTotal Lesion Load (TLL) Correlation (r): {corr:.4f}")

# --- VISUALIZE FALSE NEGATIVE ---
if false_negative_sample:
    p, m = false_negative_sample
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(m[0, 0], cmap='hot'); plt.title("Ground Truth (The Missed Lesion)")
    plt.subplot(1, 2, 2); plt.imshow(p[0, 0], cmap='jet'); plt.title("Model Confidence Map")
    plt.savefig("false_negative_analysis.png")
    print("\nFalse Negative analysis saved as 'false_negative_analysis.png'")