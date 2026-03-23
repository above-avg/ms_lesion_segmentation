import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize # New import for resizing

DATA_DIR = "dataset"
PROCESSED_DIR = "processed_2d"
TARGET_SIZE = (256, 256) # Standardizing all images to this size

os.makedirs(f"{PROCESSED_DIR}/images", exist_ok=True)
os.makedirs(f"{PROCESSED_DIR}/masks", exist_ok=True)

def normalize(data):
    mask = data > 0
    if np.any(mask):
        mu = data[mask].mean()
        sigma = data[mask].std()
        data[mask] = (data[mask] - mu) / (sigma + 1e-8)
    return data

patient_folders = [f for f in os.listdir(DATA_DIR) if f.startswith("Patient-")]

for folder in tqdm(patient_folders, desc="Standardizing & Extracting Slices"):
    p_id = folder.split("-")[-1]
    p_path = os.path.join(DATA_DIR, folder)
    
    flair_path = os.path.join(p_path, f"{p_id}-Flair.nii")
    mask_path = os.path.join(p_path, f"{p_id}-LesionSeg-Flair.nii")
    
    if not os.path.exists(flair_path): flair_path += ".gz"
    if not os.path.exists(mask_path): mask_path += ".gz"

    try:
        flair_vol = nib.load(flair_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()
        flair_vol = normalize(flair_vol)
        
        for s in range(flair_vol.shape[2]):
            img_slice = flair_vol[:, :, s]
            mask_slice = mask_vol[:, :, s]
            
            if np.mean(img_slice) > 0.01:
                # --- NEW RESIZE STEP ---
                # 'anti_aliasing' helps preserve small lesion details during downsampling
                img_res = resize(img_slice, TARGET_SIZE, order=1, preserve_range=True, anti_aliasing=True)
                # 'order=0' is critical for masks to keep them binary (0 or 1)
                mask_res = resize(mask_slice, TARGET_SIZE, order=0, preserve_range=True, anti_aliasing=False)
                
                slice_name = f"p{p_id}_s{s}.npy"
                np.save(os.path.join(PROCESSED_DIR, "images", slice_name), img_res.astype(np.float32))
                np.save(os.path.join(PROCESSED_DIR, "masks", slice_name), mask_res.astype(np.float32))
                
    except Exception as e:
        print(f"Error in {folder}: {e}")