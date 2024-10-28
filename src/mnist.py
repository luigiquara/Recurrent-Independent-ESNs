import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_mnist_data(root, image_size, batch_size, subset_size=-1, val_percentage=10):
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.1307,), (0.3081,)) # mean and std of mnist training set
    ])

    train_dataset = datasets.MNIST(root, train=True, transform=preprocessing)
    test_dataset = datasets.MNIST(root, train=False, transform=preprocessing)

    # use a smaller subset, for debugging purposes
    if subset_size > 0:
        train_idxs = np.random.choice(len(train_dataset), subset_size, replace=False)
        test_idxs = np.random.choice(len(test_dataset), subset_size, replace=False)

        train_dataset = Subset(train_dataset, train_idxs)
        test_dataset = Subset(test_dataset, test_idxs)

    # extract a validation set from the training set
    val_size = int(len(train_dataset) * val_percentage / 100.0)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader