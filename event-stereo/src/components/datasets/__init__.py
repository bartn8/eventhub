import torch
from utils.dataloader import MultiEpochsDataLoader

from . import dsec
from . import m3ed
from . import nsd
from . import scannet
from . import mvsec

def get_dataset(name, args, dataset_cfg):
    if name == 'scannet':
        return scannet.get_dataset(args, dataset_cfg)
    elif name == 'nsd':
        return nsd.get_dataset(args, dataset_cfg)
    elif name == 'dsec':
        return dsec.get_dataset(args, dataset_cfg)
    elif name == 'm3ed':
        return m3ed.get_dataset(args, dataset_cfg)
    elif name == 'mvsec':
        return mvsec.get_dataset(args, dataset_cfg)
    else:
        raise ValueError(f"Dataset {name} not recognized.")

def get_dataloader(name, args, dataset, dataloader_cfg):
    if name == 'get_simple_dataloader':
        return get_simple_dataloader(dataset, dataloader_cfg, args.num_workers)
    elif name == 'get_multi_epochs_dataloader':
        return get_multi_epochs_dataloader(dataset, dataloader_cfg, args.num_workers)
    elif name == 'get_sequence_dataloader':
        return get_sequence_dataloader(dataset, dataloader_cfg, args.num_workers)
    else:
        raise ValueError(f"Dataloader {name} not recognized.")
    
    
def get_simple_dataloader(dataset, dataloader_cfg, num_workers):
    return torch.utils.data.DataLoader(dataset=dataset,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        **dataloader_cfg.PARAMS)

def get_multi_epochs_dataloader(dataset, dataloader_cfg, num_workers):
    if len(dataset) == 0:
        return torch.utils.data.DataLoader(dataset)

    multi_epochs_dataloader = MultiEpochsDataLoader(dataset=dataset,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    **dataloader_cfg.PARAMS)

    return multi_epochs_dataloader


def get_sequence_dataloader(dataset, dataloader_cfg, num_workers):
    if len(dataset) == 0:
        return torch.utils.data.DataLoader(dataset)

    sequence_dataloader = [torch.utils.data.DataLoader(dataset=sequence_dataset,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        **dataloader_cfg.PARAMS)
                            for sequence_dataset in dataset.sequence_data_list]

    return sequence_dataloader