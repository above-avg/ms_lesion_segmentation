import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning.callbacks as pl_callbacks

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord, MapTransform, ResampleToMatchd,
    ConcatItemsd, DeleteItemsd, SpatialPadd, CropForegroundd,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd
)
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism

torch.set_float32_matmul_precision('high')


class ConvertToMultiChannel25Dd(MapTransform):
    """
    Converts a multi-modal 3D volume crop into a 2.5D multi-channel tensor.
    - Image: [C, H, W, Z] -> [C*Z, H, W] (e.g., 3 modalities * 3 slices = 9 channels)
    - Label: [1, H, W, Z] -> [1, H, W] (Extracts the center slice for prediction)
    """
    def __call__(self, data):
        d = dict(data)

        if "image" not in d:
            raise ValueError(f"Missing 'image'. Keys found: {list(d.keys())}")

        img = d["image"]
        c, h, w, z = img.shape
        d["image"] = img.permute(0, 3, 1, 2).reshape(c * z, h, w)

        
        lbl = d["label"]
        center_z = z // 2
        d["label"] = lbl[..., center_z]  

        return d



class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class UNetPlusPlus_25D(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.depth = len(features)
        self.nodes = nn.ModuleDict()

        
        curr_ch = in_channels
        for i, feat in enumerate(features):
            self.nodes[f"x_{i}_0"] = ResBlock(curr_ch, feat)
            curr_ch = feat

        
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                in_ch = features[i] * j + features[i + 1]
                self.nodes[f"x_{i}_{j}"] = ResBlock(in_ch, features[i])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        out = {}
        out["x_0_0"] = self.nodes["x_0_0"](x)
        out["x_1_0"] = self.nodes["x_1_0"](nn.MaxPool2d(2)(out["x_0_0"]))
        out["x_2_0"] = self.nodes["x_2_0"](nn.MaxPool2d(2)(out["x_1_0"]))
        out["x_3_0"] = self.nodes["x_3_0"](nn.MaxPool2d(2)(out["x_2_0"]))

        out["x_0_1"] = self.nodes["x_0_1"](torch.cat([out["x_0_0"], self.up(out["x_1_0"])], 1))
        out["x_1_1"] = self.nodes["x_1_1"](torch.cat([out["x_1_0"], self.up(out["x_2_0"])], 1))
        out["x_2_1"] = self.nodes["x_2_1"](torch.cat([out["x_2_0"], self.up(out["x_3_0"])], 1))

        out["x_0_2"] = self.nodes["x_0_2"](torch.cat([out["x_0_0"], out["x_0_1"], self.up(out["x_1_1"])], 1))
        out["x_1_2"] = self.nodes["x_1_2"](torch.cat([out["x_1_0"], out["x_1_1"], self.up(out["x_2_1"])], 1))

        out["x_0_3"] = self.nodes["x_0_3"](torch.cat([out["x_0_0"], out["x_0_1"], out["x_0_2"], self.up(out["x_1_2"])], 1))

        return self.final_head(out["x_0_3"])



class MSLesionModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNetPlusPlus_25D(in_channels=9, out_channels=1, features=[32, 64, 128, 256])

        
        self.loss_function = DiceFocalLoss(
            sigmoid=True,
            gamma=3.0,
            lambda_focal=5.0,   
        )

        
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        loss = self.loss_function(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        loss = self.loss_function(logits, labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        self.dice_metric(y_pred=preds, y=labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dice", mean_dice, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



def main():
    set_determinism(seed=42)

    data_dir = "" #path to the data

    if not os.path.exists(data_dir):
        raise RuntimeError(f"Data directory not found: {data_dir}")

    patient_folders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir() and "Patient" in f.name])

    data_dicts = []
    for folder in patient_folders:
        folder_name = os.path.basename(folder)
        patient_id = folder_name.split('-')[-1]

        flair_path = os.path.join(folder, f"{patient_id}-Flair.nii")
        t1_path    = os.path.join(folder, f"{patient_id}-T1.nii")
        t2_path    = os.path.join(folder, f"{patient_id}-T2.nii")
        label_path = os.path.join(folder, f"{patient_id}-LesionSeg-Flair.nii")

        if all(os.path.exists(p) for p in [flair_path, t1_path, t2_path, label_path]):
            data_dicts.append({
                "flair": flair_path,
                "t1": t1_path,
                "t2": t2_path,
                "label": label_path
            })
        else:
            print(f"Skipping {folder_name}: Could not find all required .nii files.")

    if len(data_dicts) == 0:
        raise RuntimeError("No data found! Please double-check the directory path and filenames.")

    print(f"Successfully loaded {len(data_dicts)} patients.")

    split = int(len(data_dicts) * 0.8)
    train_files, val_files = data_dicts[:split], data_dicts[split:]

    
    load_and_align = [
        LoadImaged(keys=["flair", "t1", "t2", "label"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t2", "label"]),
        ResampleToMatchd(keys=["t1", "t2", "label"], key_dst="flair", mode=("bilinear", "bilinear", "nearest")),
        
        ScaleIntensityRangePercentilesd(keys=["flair", "t1", "t2"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=["flair", "t1", "t2"], name="image", dim=0),
        DeleteItemsd(keys=["flair", "t1", "t2"]),
        
        SpatialPadd(keys=["image", "label"], spatial_size=(224, 224, 3)),
    ]

    train_transforms = Compose(load_and_align + [
        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(224, 224, 3),
            pos=4,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        ConvertToMultiChannel25Dd(keys=["image", "label"]),
        
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        ToTensord(keys=["image", "label"]),
    ])

    
    val_transforms = Compose(load_and_align + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(224, 224, 3),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        ConvertToMultiChannel25Dd(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = MSLesionModel(learning_rate=1e-4)

    checkpoint_cb = pl_callbacks.ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_dice:.3f}",
        verbose=True,
    )
    early_stop_cb = pl_callbacks.EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=30,  
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=5,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
