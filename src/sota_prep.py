import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

DATA_DIR = "dataset"
PROCESSED_DIR = "sota_data"
TARGET_SIZE = (256, 256) # Standardizing everything to this matrix size

os.makedirs(f"{PROCESSED_DIR}/images", exist_ok=True)
os.makedirs(f"{PROCESSED_DIR}/masks", exist_ok=True)

def normalize(data):
    mask = data > 0
    if np.any(mask):
        mu, sigma = data[mask].mean(), data[mask].std()
        data[mask] = (data[mask] - mu) / (sigma + 1e-8)
    return data

def get_resized_slice(volume, slice_idx):
    """Safely extracts and resizes a single 2D slice."""
    s = volume[:, :, slice_idx]
    return resize(s, TARGET_SIZE, order=1, preserve_range=True, anti_aliasing=True)

patient_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("Patient-")]

for folder in tqdm(patient_folders, desc="Processing Multi-Modal Slices"):
    p_id = folder.split("-")[-1]
    p_path = os.path.join(DATA_DIR, folder)
    
    try:
        # Load the 3 primary modalities + Mask
        flair = normalize(nib.load(os.path.join(p_path, f"{p_id}-Flair.nii")).get_fdata())
        t1 = normalize(nib.load(os.path.join(p_path, f"{p_id}-T1.nii")).get_fdata())
        t2 = normalize(nib.load(os.path.join(p_path, f"{p_id}-T2.nii")).get_fdata())
        mask = nib.load(os.path.join(p_path, f"{p_id}-LesionSeg-Flair.nii")).get_fdata()

        # Find the minimum number of slices across all volumes to avoid IndexError
        min_slices = min(flair.shape[2], t1.shape[2], t2.shape[2], mask.shape[2])

        # 2.5D Slicing: current slice (s) plus neighbors (s-1, s+1)
        for s in range(1, min_slices - 1):
            # RESIZE EACH INDIVIDUALLY BEFORE STACKING
            # This is the fix for your ValueError
            slices = [
                get_resized_slice(flair, s-1), get_resized_slice(flair, s), get_resized_slice(flair, s+1),
                get_resized_slice(t1, s-1),    get_resized_slice(t1, s),    get_resized_slice(t1, s+1),
                get_resized_slice(t2, s-1),    get_resized_slice(t2, s),    get_resized_slice(t2, s+1)
            ]
            
            multi_slice = np.stack(slices, axis=0) # Shape: [9, 256, 256]
            
            # Mask only needs the current slice 's'
            mask_slice = resize(mask[:,:,s], TARGET_SIZE, order=0, preserve_range=True, anti_aliasing=False)
            
            # SOTA strategy: We only train on slices that actually contain lesions
            if mask_slice.sum() > 0:
                out_name = f"p{p_id}_s{s}.npy"
                np.save(os.path.join(PROCESSED_DIR, "images", out_name), multi_slice.astype(np.float32))
                np.save(os.path.join(PROCESSED_DIR, "masks", out_name), mask_slice.astype(np.float32))
                
    except Exception as e:
        print(f"Skipping {folder} due to error: {e}")