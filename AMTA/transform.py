from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    HistogramNormalized,
    RandAffined, Resized, RandSpatialCropd,
    RandCropByLabelClassesd,
    Spacing, SpatialCropd, SpatialCropd
)
import torch
import os
import numpy as np
import nibabel as nib

def get_transform(size, resolution, image_path, mask_path, target_label, is_train=False, avg_center=None):
    
    post_augmentation = [
        HistogramNormalized(keys=["t2w"]),
        ScaleIntensityd(keys=["t2w"]),
        ToTensord(keys=["t2w", "seg"])
    ]
    
    center = 0
    
    if is_train:
        pre_augmentation = [
            LoadImaged(keys=["t2w", "seg"]),
            AddChanneld(keys=["t2w", "seg"]),
            Spacingd(
                keys=["t2w", "seg"],
                pixdim=resolution,
                mode=("bilinear", "nearest"),
            ),
        ]
        
        # Crop with the center of target label
        pa = Compose(pre_augmentation)
        po = pa({"t2w": image_path, "seg": mask_path})
        label = np.array(po['seg'][0])
        ms = ()
        for t in target_label:
            ms += (np.vstack(np.where(label==t)),)
        ms = np.hstack(ms)
        center = np.mean(ms, axis=1).astype(np.int32)
        # The center will be returned to compute average center
        
        middle_transform = [
            # CenterSpatialCropd(keys=["t2w", "seg"], roi_size=size),
            SpatialCropd(keys=["t2w", "seg"], roi_size=size, roi_center=center),
            SpatialPadd(
                keys=["t2w", "seg"],
                spatial_size=size,
                method='symmetric',
                mode='constant',
                allow_missing_keys=False
            )
        ]
    else:
        # Remove all-zero layer
        mask = np.array(nib.load(mask_path).dataobj)
        width, height, queue = mask.shape
        start = 0
        while start < queue and np.all(mask[:, :, start] == 0):
            start += 1
        end = queue - 1
        while end > start and np.all(mask[:, :, end] == 0):
            end -= 1

        pre_augmentation = [
            LoadImaged(keys=["t2w", "seg"]),
            AddChanneld(keys=["t2w", "seg"]),
            SpatialCropd(keys=["t2w", "seg"], roi_start=[0,0,start], roi_end=[1000,1000,end+1]),
            Spacingd(
                keys=["t2w", "seg"],
                pixdim=resolution,
                mode=("bilinear", "nearest"),
            ),
        ]
        
        #print(avg_center)
        
        middle_transform = [
            CenterSpatialCropd(keys=["t2w", "seg"], roi_size=size),
            # Crop with average center could not be used with "Removing all-zero layer" at the same time
            # SpatialCropd(keys=["t2w", "seg"], roi_size=size, roi_center=avg_center),
            SpatialPadd(
                keys=["t2w", "seg"],
                spatial_size=size,
                method='symmetric',
                mode='constant',
                allow_missing_keys=False
            )
        ]

    return Compose(pre_augmentation + middle_transform + post_augmentation), center


if __name__ == "__main__":
    transform = get_transform(
            size=[ 256, 256, 40 ],
            resolution=[ 0.75, 0.75, 2.5 ]
        )
    codes = []
    for f in os.listdir('data'):
        if 'img' in f:
            codes.append(f[:6])
    for c in codes:
        out = transform({"t2w":"data/%s_img.nii" % c, "seg":"data/%s_mask.nii" % c})
        t2w = out['t2w']
        seg = out['seg']
        torch.save(t2w, "tensor/%s_img.pt" % c)
        torch.save(seg, "tensor/%s_mask.pt" % c)