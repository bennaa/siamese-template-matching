import random

import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(1)
random.seed(1)


#  https://github.com/fangpin/siamese-pytorch/blob/master/mydataset.py
class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = list(self.dataset.keys())
        self.tot_images = np.sum([len(images) for class_id, images in self.dataset.items()])

    def __getitem__(self, idx):
        idx1 = random.choice(self.classes)  # sample one index (i.e one class)
        idx2 = idx1
        img1 = random.choice(self.dataset[idx1])  # sample one image from that class

        # we need to have 50% of positive pairs on each batch
        if idx % 2 == 1:  # generate a pair of the same class
            label = 0.0  # Positive
            img2 = random.choice(self.dataset[idx2])  # sample another image from the same class
        else:  # generate a pair of the different classes
            label = 1.0  # Negative
            idx2 = random.choice(self.classes)

            while idx1 == idx2:  # loop until the classes are different
                idx2 = random.choice(self.classes)

            img2 = random.choice(self.dataset[idx2])

        return (img1, idx1), (img2, idx2), torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return self.tot_images


class TripletSiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = list(self.dataset.keys())
        self.tot_images = np.sum([len(images) for class_id, images in self.dataset.items()])

    def __getitem__(self, idx):
        # sample two different classes
        class1, class2 = np.random.choice(self.classes, 2, replace=False)

        # first sample 2 indexes in the same class and then use these indices to get the 2 images of the same class
        anchor_idx, positive_idx = np.random.choice(len(self.dataset[class1]), 2, replace=False)
        anchor, positive = self.dataset[class1][anchor_idx], self.dataset[class1][positive_idx]

        # sample one image of the remaining class (different from the previous one)
        negative_idx = np.random.choice(len(self.dataset[class2]))
        negative = self.dataset[class2][negative_idx]

        return (anchor, class1), (positive, class1), (negative, class2)

    def __len__(self):
        return self.tot_images


def get_loader(filename, shuffle=False, batch_size=32, triplet=False):
    data = torch.load(filename)
    if triplet:
        dataset = TripletSiameseDataset(data)
    else:
        dataset = SiameseDataset(data)
    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
