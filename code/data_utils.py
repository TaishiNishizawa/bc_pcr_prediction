import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import json
from typing import List, Optional
import random

def balance_data(data_list, label_key='label'):
    positive_samples = [data for data in data_list if data[label_key] == 1]
    negative_samples = [data for data in data_list if data[label_key] == 0]

    num_negatives = len(negative_samples)
    num_positives = len(positive_samples)
    upsample_factor = num_negatives // num_positives

    balanced_data_list = negative_samples + positive_samples * upsample_factor

    random.shuffle(balanced_data_list)
    return balanced_data_list


class NumpyDataset(Dataset):
    def __init__(self, data_list : List[dict], transform : Optional[tio.transforms.Transform] = None, num_transforms : int=1):
        self.data_list : List[dict] = data_list
        self.transform : Optional[tio.transforms.Transform] = transform
        self.num_transforms : int = num_transforms

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Load images and convert them directly to PyTorch tensors with expanded dimensions
        t0_first = torch.tensor(np.load(self.data_list[idx]['image_t0_first']).astype(np.float32)).unsqueeze(0)
        t1_first = torch.tensor(np.load(self.data_list[idx]['image_t1_first']).astype(np.float32)).unsqueeze(0)
        t2_first = torch.tensor(np.load(self.data_list[idx]['image_t2_first']).astype(np.float32)).unsqueeze(0)
        t3_first = torch.tensor(np.load(self.data_list[idx]['image_t3_first']).astype(np.float32)).unsqueeze(0)

        patient_id = self.data_list[idx]['patient_id']

        non_mri_features = self.data_list[idx]['non_mri_features']
        label = self.data_list[idx]['label']

        transformed_images = []
        for _ in range(self.num_transforms):
            t0_first_transformed = self.transform(t0_first)
            t1_first_transformed = self.transform(t1_first)
            t2_first_transformed = self.transform(t2_first)
            t3_first_transformed = self.transform(t3_first)
            
            transformed_images.append((t0_first_transformed, t1_first_transformed, t2_first_transformed, t3_first_transformed))
            

        return transformed_images, [non_mri_features] * self.num_transforms, [label] * self.num_transforms, [patient_id] * self.num_transforms

    
def collate_fn(batch):
    # Unpack the batch
    images, features, labels, pids = zip(*batch)

    t0_first_images = []
    t1_first_images = []
    t2_first_images = []
    t3_first_images = []
    non_mri_features = []
    flattened_labels = []
    patient_ids = []

    for image_set, feature_set, label_set, pid_set in zip(images, features, labels, pids):
        for (t0_first, t1_first, t2_first, t3_first) in image_set:
            t0_first_images.append(t0_first)
            t1_first_images.append(t1_first)
            t2_first_images.append(t2_first)
            t3_first_images.append(t3_first)

        non_mri_features.extend(feature_set)
        flattened_labels.extend(label_set)
        patient_ids.extend(pid_set)

    # Stack the images and labels into tensors
    t0_first_images = torch.stack(t0_first_images)
    t1_first_images = torch.stack(t1_first_images)
    t2_first_images = torch.stack(t2_first_images)
    t3_first_images = torch.stack(t3_first_images)

    non_mri_features = non_mri_features
    flattened_labels = torch.tensor(flattened_labels, dtype=torch.long)
    patient_ids = torch.tensor(patient_ids, dtype=torch.int)

    return (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri_features, flattened_labels, patient_ids    
            

def get_data_loader(type : str, indices, data_list, batch_size=2, num_workers = 1):
    num_transforms = 1

    if type == "train":
        is_horizontal_flip = random.choice([0.0, 1.0]) 

        # Apply transformations using TorchIO
        transform = tio.Compose([
            tio.transforms.CropOrPad((80, 256, 256)), 
            tio.transforms.RandomFlip(axes=(2), flip_probability=is_horizontal_flip), 
            tio.RescaleIntensity(out_min_max=(0, 1)) 
        ])
        # num_transforms = 3
    else:
        transform = tio.Compose([
            tio.transforms.CropOrPad((80, 256, 256)),
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])

    # Apply the balancing function only to the training data
    if type == "train":
        data = [data_list[i] for i in indices]
        balanced_data_list = balance_data(data)
    else:
        balanced_data_list = [data_list[i] for i in indices]

    dataset = NumpyDataset(data_list=balanced_data_list, transform=transform, num_transforms = num_transforms)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=(type == "train"), num_workers = num_workers, collate_fn = collate_fn)

    return data_loader