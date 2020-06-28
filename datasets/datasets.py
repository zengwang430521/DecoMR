"""
This file contains the definition of different heterogeneous datasets used for training
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
import torch
import numpy as np

from .base_dataset import BaseDataset


class ITWDataset(torch.utils.data.Dataset):
    """Mixed dataset with data only from "in-the-wild" datasets (no data from H36M)."""
    
    def __init__(self, options, use_IUV=False):
        super(ITWDataset, self).__init__()
        self.lsp_dataset = BaseDataset(options, 'lsp-orig', use_IUV=use_IUV)
        self.coco_dataset = BaseDataset(options, 'coco', use_IUV=use_IUV)
        self.mpii_dataset = BaseDataset(options, 'mpii', use_IUV=use_IUV)
        self.up3d_dataset = BaseDataset(options, 'up-3d', use_IUV=use_IUV)

        self.length = max(len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        self.partition = np.array([.1, .3, .3, .3]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        elif p <= self.partition[1]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        elif p <= self.partition[2]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        elif p <= self.partition[3]:
            return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length


class MeshDataset(torch.utils.data.Dataset):
    """Mixed dataset with data from all available datasets."""

    def __init__(self, options, use_IUV=False):
        super(MeshDataset, self).__init__()
        self.h36m_dataset = BaseDataset(options, 'h36m-train', use_IUV=use_IUV)
        self.up3d_dataset = BaseDataset(options, 'up-3d', use_IUV=use_IUV)
        self.length = max(len(self.h36m_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        self.partition = np.array([.7, .3]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.h36m_dataset[i % len(self.h36m_dataset)]
        else:
            return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length


class FullDataset(torch.utils.data.Dataset):
    """Mixed dataset with data from all available datasets."""

    def __init__(self, options, use_IUV=False):
        super(FullDataset, self).__init__()
        self.h36m_dataset = BaseDataset(options, 'h36m-train', use_IUV=use_IUV)
        self.lsp_dataset = BaseDataset(options, 'lsp-orig', use_IUV=use_IUV)
        self.coco_dataset = BaseDataset(options, 'coco', use_IUV=use_IUV)
        self.mpii_dataset = BaseDataset(options, 'mpii', use_IUV=use_IUV)
        self.up3d_dataset = BaseDataset(options, 'up-3d', use_IUV=use_IUV)

        self.length = max(len(self.h36m_dataset),
                          len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        self.partition = np.array([.3, .1, .2, .2, .2]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.h36m_dataset[i % len(self.h36m_dataset)]
        elif p <= self.partition[1]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        elif p <= self.partition[2]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        elif p <= self.partition[3]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        elif p <= self.partition[4]:
            return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length


def create_dataset(dataset, options, use_IUV=False):
    if dataset == 'all':
        return FullDataset(options, use_IUV=use_IUV)
    elif dataset == 'itw':
        return ITWDataset(options, use_IUV=use_IUV)
    elif dataset == 'h36m':
        return BaseDataset(options, 'h36m-train', use_IUV=use_IUV)
    elif dataset == 'up-3d':
        return BaseDataset(options, 'up-3d', use_IUV=use_IUV)
    elif dataset == 'mesh':
        return MeshDataset(options, use_IUV=use_IUV)
    else:
        raise ValueError('Undefined dataset')
