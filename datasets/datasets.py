"""
This file contains the definition of different heterogeneous datasets used for training
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
import torch
import numpy as np

from .base_dataset import BaseDataset
from .surreal_dataset import SurrealDataset


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


class SPINDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        self.dataset_list = ['h36m-train', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        self.dataset_dict = {'h36m-train': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = [.3,
                          .6*len(self.datasets[1])/length_itw,
                          .6*len(self.datasets[2])/length_itw,
                          .6*len(self.datasets[3])/length_itw,
                          .6*len(self.datasets[4])/length_itw,
                          0.1]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(6):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

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
    elif dataset == 'spin':
        return SPINDataset(options, use_IUV=use_IUV)
    elif dataset == 'surreal':
        return SurrealDataset(options, use_IUV=use_IUV)
    else:
        raise ValueError('Undefined dataset')
