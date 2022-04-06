import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class WaterBirdsDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        print(len(metadata_df))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print(len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return img, y, g, p


def get_transform_cub(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(data, train, reweight_groups, reweight_classes, reweight_places, **kwargs):
    if not train: # Validation or testing
        assert reweight_groups is None
        assert reweight_classes is None
        assert reweight_places is None
        shuffle = False
        sampler = None
    elif not (reweight_groups or reweight_classes or reweight_places): # Training but not reweighting
        shuffle = True
        sampler = None
    elif reweight_groups:
        # Training and reweighting groups
        # reweighting changes the loss function from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight)
        group_weights = len(data) / data.group_counts
        weights = group_weights[data.group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    elif reweight_classes:  # Training and reweighting classes
        class_weights = len(data) / data.y_counts
        weights = class_weights[data.y_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    else: # Training and reweighting places
        place_weights = len(data) / data.p_counts
        weights = place_weights[data.p_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False

    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader


def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    # group_id = y_id * n_places + place_id
    # y_id = group_id // n_places
    # place_id = group_id % n_places
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')
    if val_data is not None:
        logger.write(f'Validation Data (total {len(val_data)})\n')
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')
