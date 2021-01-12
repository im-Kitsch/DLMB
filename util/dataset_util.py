import torch
import torch.utils.data
import torchvision
import numpy as np


def split_dataset(dataset, num_sup):
    sup_ind = []
    unsup_ind = []

    targets = dataset.targets.numpy()
    tar_unique = np.unique(targets)
    lab_per_class = np.int(num_sup/len(tar_unique))
    for i in  tar_unique:
        ind = np.argwhere(targets == i)
        ind = ind.reshape(-1)
        rng = np.random.default_rng()
        rng.shuffle(ind)
        sup_ind = sup_ind + ind[:lab_per_class].tolist()
        unsup_ind = unsup_ind + ind[lab_per_class:].tolist()

    sup_dataset = torch.utils.data.Subset(dataset, sup_ind)
    unsup_ind = torch.utils.data.Subset(dataset, unsup_ind)
    return sup_dataset, unsup_ind