import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from monai.networks.nets import BasicUNetPlusPlus
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. CONFIGURATION & MODEL LOADING ---
device = torch.device("cuda")
PROCESSED_DIR = "sota_data"
# Validation on Patients 49-60 (Unseen during training of their respective folds)
eval_pids = [str(i) for i in range(49, 61)]
eval_files = [f for f in glob.glob(f"{PROCESSED_DIR}/images/*.npy") 
              if any(f"p{p}_" in os.path.basename(f) for p in eval_pids)]

print(f"Loading Ensemble... Found {len(eval_files)} validation slices.")
models = []
for i in range(1, 6):
    model = BasicUNetPlusPlus(
        spatial_dims=2, 
        in_channels=9, 
        out_channels=1,
        deep_supervision=True, 
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)
    
    model_path = f"sota_fold_{i}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        models.append(model)
    else:
        print(f"Warning: {model_path} not found. Skipping.")

# --- 2. INFERENCE LOOP (PROBABILITY COLLECTION) ---
all_probs = []
all_masks = []

print("Running Ensemble Inference...")
with torch.no_grad():
    for f in tqdm(eval_files):
        # Load 9-channel image and 1-channel mask
        x = torch.from_numpy(np.load(f)).unsqueeze(0).to(device)
        y = torch.from_numpy(np.load(f.replace("images", "masks"))).unsqueeze(0).cpu()
        
        # Average probabilities from all available folds
        fold_probs = []
        for m in models:
            outputs = m(x)
            # Take the first output (final resolution) from the Deep Supervision list
            final_out = outputs[0] if isinstance(outputs, list) else outputs
            fold_probs.append(torch.sigmoid(final_out))
        
        ensemble_prob = torch.mean(torch.stack(fold_probs), dim=0).cpu()
        all_probs.append(ensemble_prob)
        all_masks.append(y)

# --- 3. STEP 1: THRESHOLD SWEEP (0.1 to 0.9) ---
thresholds = np.arange(0.1, 1.0, 0.1)
sweep_results = []

print("\nCalculating Threshold Sweep...")
for t in thresholds:
    tp, fp, fn, tn = 0, 0, 0, 0
    for p, m in zip(all_probs, all_masks):
        pred = (p > t).float()
        tp += (pred * m).sum().item()
        fp += (pred * (1 - m)).sum().item()
        fn += ((1 - pred) * m).sum().item()
        tn += ((1 - pred) * (1 - m)).sum().item()
    
    sensitivity = tp / (tp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    sweep_results.append({"Threshold": round(t, 1), "Sensitivity": sensitivity, "Dice": dice})

sweep_df = pd.DataFrame(sweep_results)
print("\n--- THRESHOLD SWEEP RESULTS ---")
print(sweep_df.to_string(index=False))

# --- 4. STEP 2: TLL CORRELATION (r) ---
# We use t=0.3 as the standard clinical 'optimal' from the sweep
opt_t = 0.3
gt_volumes = [m.sum().item() for m in all_masks]
pred_volumes = [(p > opt_t).float().sum().item() for p in all_probs]

corr, p_value = pearsonr(gt_volumes, pred_volumes)
print(f"\nTotal Lesion Load (TLL) Correlation (r): {corr:.4f}")
print(f"P-value: {p_value:.4e}")

# --- 5. STEP 3: GREEDY ERROR SEARCH (FALSE NEGATIVE ANALYSIS) ---
max_fn_error = -1
error_sample = None

print("\nPerforming Greedy Error Search for clinical report...")
for p, m in zip(all_probs, all_masks):
    # Masking where the model missed (GT is 1, Pred is 0)
    fn_map = (m == 1) & (p < opt_t)
    fn_count = fn_map.sum().item()
    
    if fn_count > max_fn_error:
        max_fn_error = fn_count
        error_sample = (p, m)

# --- UPDATED VISUALIZATION BLOCK ---
# --- FINAL PRO-VERSION VISUALIZATION ---
if error_sample:
    p, m = error_sample
    p_plot = p.squeeze().numpy()
    m_plot = m.squeeze().numpy()

    # Increased figure height to prevent cropping
    plt.figure(figsize=(18, 7)) 
    
    # 1. Ground Truth
    plt.subplot(1, 3, 1)
    plt.imshow(m_plot, cmap='gray')
    plt.title("Expert Ground Truth (Gold Standard)", fontsize=14, pad=20)
    plt.axis('off')
    
    # 2. AI Confidence
    plt.subplot(1, 3, 2)
    plt.imshow(p_plot, cmap='jet')
    plt.title("AI Lesion Confidence Map (Probabilities)", fontsize=14, pad=20)
    plt.axis('off')
    
    # 3. Clinical Error Analysis
    plt.subplot(1, 3, 3)
    plt.imshow(m_plot, cmap='gray', alpha=0.5)
    miss_mask = (m_plot == 1) & (p_plot < opt_t)
    if np.any(miss_mask):
        plt.imshow(np.ma.masked_where(~miss_mask, miss_mask), cmap='autumn', alpha=0.9)
    
    plt.title(f"Clinical Error: {int(max_fn_error)} Missed Pixels", fontsize=14, pad=20)
    plt.axis('off')
    
    # rect=[left, bottom, right, top] - this ensures titles are NOT cropped
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    # Save as high-res PNG for the proposal
    plt.savefig("false_negative_analysis2.png", dpi=300, bbox_inches='tight')
    print(f"\nSuccess: High-resolution analysis saved as 'false_negative_analysis.png'")

# Save sweep results to CSV for the proposal
sweep_df.to_csv("threshold_sweep_report.csv", index=False)