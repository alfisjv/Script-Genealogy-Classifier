# dataset.py

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import compute_directional_features
from config import resize_x, resize_y, batchsize

# -----------------------------------
# Transforms
# -----------------------------------
transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=2, translate=(0.05, 0.05)),
    transforms.RandomAutocontrast(p=0.3),
    transforms.Lambda(lambda img: compute_directional_features(img)),
])

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((resize_x, resize_y)),
    transforms.Lambda(lambda img: compute_directional_features(img)),
])

# -----------------------------------
# Custom Dataset Loader
# -----------------------------------
class CustomDataset:
    def __init__(self, train_dir, test_dir, val_split=0.2):
        from torch.utils.data import random_split
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler

        self.train_path = train_dir
        self.test_path = test_dir

        # Full train dataset
        full_train_dataset = datasets.ImageFolder(self.train_path, transform=transform_train)

        val_size = int(val_split * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Create class balancing
        train_targets = [full_train_dataset.targets[i] for i in self.train_dataset.indices]
        class_counts = Counter(train_targets)
        total_samples = sum(class_counts.values())
        class_weights = {cls: total_samples / (count + 1e-6) for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_targets]
        self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        self.test_dataset = datasets.ImageFolder(self.test_path, transform=transform_test)

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=batchsize, sampler=self.sampler)
        val_loader = DataLoader(self.val_dataset, batch_size=batchsize, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batchsize, shuffle=False)
        return train_loader, val_loader, test_loader

# -----------------------------------
# Convenience Dataloader Creator
# -----------------------------------
def get_dataloader(train_dir, test_dir):
    dataset = CustomDataset(train_dir, test_dir)
    return dataset.get_loaders()
