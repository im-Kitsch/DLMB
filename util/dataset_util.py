import torch
import torch.utils.data
import torchvision
import numpy as np
import os
import util.ham10000_dataset


def load_dataset(dataset_name, root, transform=None, csv_file=None, percentage=1.):
    if transform is None:
        transform = torchvision.transforms.ToTensor()

    root = os.path.join(root, dataset_name)
    if dataset_name == 'MNIST':
        train_data = torchvision.datasets.MNIST(root=root, transform=transform, train=True, download=True)
        # if percentage < 1.0:
        #     train_data = torch.utils.data.Subset(train_data, range(int(len(train_data)*percentage)))
        test_data = torchvision.datasets.MNIST(root=root, transform=transform, train=False)
        # img_info = (1, 28, 28)
    elif dataset_name == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        # if percentage < 1.0:
        #     train_data = torch.utils.data.Subset(train_data, int(len(train_data) * percentage))
        test_data = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform)
        # img_info = (3, 32, 32)
    elif dataset_name == 'HAM10000':
        if csv_file is None:
            raise Exception['csv_file is required']
        train_data = util.ham10000_dataset.HAM10000Dataset(img_root=root, csv_file=csv_file,
                                                           transform=transform, report=True)
        # if percentage < 1.0:
        #     train_data = torch.utils.data.Subset(train_data, int(len(train_data) * percentage))
        test_data = None
    else:
        raise Exception['Dataset not given']
    if percentage > 1.0:
        percentage = 1.0
    sample_list = range(int(len(train_data) * percentage))
    train_data = torch.utils.data.Subset(train_data, sample_list)

    img_info = tuple(train_data[0][0].shape)
    return train_data, test_data, img_info


def split_dataset(dataset, num_sup):
    sup_ind = []
    unsup_ind = []

    targets = dataset.targets.numpy()
    tar_unique = np.unique(targets)
    lab_per_class = np.int(num_sup/len(tar_unique))
    for i in tar_unique:
        ind = np.argwhere(targets == i)
        ind = ind.reshape(-1)
        rng = np.random.default_rng()
        rng.shuffle(ind)
        sup_ind = sup_ind + ind[:lab_per_class].tolist()
        unsup_ind = unsup_ind + ind[lab_per_class:].tolist()

    sup_dataset = torch.utils.data.Subset(dataset, sup_ind)
    unsup_ind = torch.utils.data.Subset(dataset, unsup_ind)
    return sup_dataset, unsup_ind