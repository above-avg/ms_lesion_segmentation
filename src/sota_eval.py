import torch
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNetPlusPlus
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import Compose, EnsureType
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. Dataset for Unseen Patients ---
class MSEvalDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        self.trans = EnsureType()

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = torch.from_numpy(np.load(self.files[i]))
        mask = torch.from_numpy(np.load(self.files[i].replace("images", "masks"))).unsqueeze(0)
        return self.trans(img), self.trans(mask)

# --- 2. Load Ensemble with Deep Supervision Handling ---
device = torch.device("cuda")
models = []

for i in range(1, 6):
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=9,
        out_channels=1,
        deep_supervision=True, # Must match training architecture
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)
    
    # Setting weights_only=True is recommended for security in clinical environments
    model.load_state_dict(torch.load(f"sota_fold_{i}.pth", weights_only=True))
    model.eval()
    models.append(model)

# --- 3. Clinical Metrics Setup ---
dice_metric = DiceMetric(include_background=False, reduction="mean")
conf_metric = ConfusionMatrixMetric(include_background=False, metric_name=["sensitivity", "specificity"])

PROCESSED_DIR = "sota_data/images"
all_files = sorted(glob.glob(f"{PROCESSED_DIR}/*.npy"))
eval_pids = [str(i) for i in range(49, 61)]
eval_files = [f for f in all_files if any(f"p{p}_" in os.path.basename(f) for p in eval_pids)]

eval_loader = DataLoader(MSEvalDataset(eval_files), batch_size=1)

# --- 4. Running Ensemble Inference ---
print(f"Running Ensemble Inference on {len(eval_files)} unseen slices...")

with torch.no_grad():
    for i, (x, y) in enumerate(tqdm(eval_loader)):
        x, y = x.to(device), y.to(device)
        
        ensemble_preds = []
        for model in models:
            outputs = model(x)
            
            # FIX: Select the first output (final resolution) from the Deep Supervision list
            final_output = outputs[0] if isinstance(outputs, list) else outputs
            pred = torch.sigmoid(final_output)
            ensemble_preds.append(pred)
        
        # Average results across all 5 models
        final_prob = torch.mean(torch.stack(ensemble_preds), dim=0)
        final_pred = (final_prob > 0.5).float()
        
        dice_metric(y_pred=final_pred, y=y)
        conf_metric(y_pred=final_pred, y=y)

        # Periodic Visualization for Proposal
        if i % 50 == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(x[0, 1].cpu(), cmap='gray'); plt.title("FLAIR Input")
            plt.subplot(1, 3, 2); plt.imshow(y[0, 0].cpu(), cmap='hot'); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(final_pred[0, 0].cpu(), cmap='jet'); plt.title("Ensemble Prediction")
            plt.savefig(f"clinical_sample_{i}.png")
            plt.close()

# --- 5. Generate Final Clinical Performance Report ---
mean_dice = dice_metric.aggregate().item()
conf_results = conf_metric.aggregate()
sensitivity = conf_results[0].item()
specificity = conf_results[1].item()

report_data = {
    "Metric": ["Dice Similarity Coefficient (DSC)", "Sensitivity (Lesion Recall)", "Specificity (Background Accuracy)"],
    "Value": [f"{mean_dice:.4f}", f"{sensitivity:.4f}", f"{specificity:.4f}"],
    "Clinical Interpretation": [
        "Overlap between prediction and gold standard",
        "Ability to detect all present lesions",
        "Ability to avoid false positives (healthy tissue)"
    ]
}

df = pd.DataFrame(report_data)
print("\n" + "="*60)
print("CLINICAL VALIDATION REPORT: MS LESION SEGMENTATION")
print("="*60)
print(df.to_string(index=False))
df.to_csv("clinical_validation_report.csv")