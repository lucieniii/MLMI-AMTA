import os
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transform import get_transform


class NPyDataset(Dataset):
    def __init__(self, folder_name, target_label, is_train=True, slice_dataset=False, center=None):
        self.folder_name = folder_name
        self.is_train = is_train
        self.slice_dataset = slice_dataset
        self.target_label = target_label
        self.center = center

        # get data list
        self.file_list = sorted([f for f in os.listdir(folder_name) if f.startswith('00') and f.endswith('_img.nii')])
        # self.file_list = sorted([f for f in os.listdir(folder_name) if f.endswith('_img.nii')])
        print('total files: %d' % len(self.file_list))

        # split data
        train_files, test_files = train_test_split(np.array(self.file_list), test_size=0.2, random_state=42)
        # choose file
        self.files = train_files if is_train else test_files
        print('total files used: %d' % len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_name, self.files[idx])
        mask_path = image_path.replace('_img.nii', '_mask.nii')
        
        self.transform, center = get_transform(
            size =[256,256,32],
            resolution=[0.75,0.75,2.5],
            image_path=image_path,
            mask_path=mask_path,
            target_label=self.target_label,
            is_train=self.is_train,
            avg_center=self.center
        )

        out = self.transform({"t2w": image_path, "seg": mask_path})
        image = out["t2w"]
        label = out["seg"]

        return image, label, self.files[idx], center

def create_data_loader(folder_name, target_label, slice_dataset = False):
    # training data loader
    train_set = NPyDataset(folder_name, target_label, is_train=True, slice_dataset=slice_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=1)
    
    # Compute average center
    avg_center = np.array([0,0,0])
    for ii, (images, labels, _, center) in enumerate(train_loader):
        avg_center += np.array(center).flatten()
    avg_center = avg_center / len(train_loader)
    avg_center = avg_center.astype(np.uint32)

    # test/validation data loader
    test_set = NPyDataset(folder_name, target_label, is_train=False, slice_dataset=slice_dataset, center=avg_center)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,  # change to False for predefined test data
        num_workers=1)

    return train_loader, test_loader        