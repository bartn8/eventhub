import os
import csv
import copy

import numpy as np

import torch.utils.data

from . import disparity
from . import event
from ..utils import transforms
from .constant import MVSEC_HEIGHT, MVSEC_WIDTH


class SequenceDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'event': 'events',
        'disparity': 'disparity',
    }
    HEIGHT = MVSEC_HEIGHT
    WIDTH = MVSEC_WIDTH

    def __init__(self, root, split, split_filter, sampling_ratio, event_cfg, disparity_cfg, 
                 crop_height, crop_width, num_workers=0, args=None):
        self.root = root
        self.split = split
        self.split_filter = split_filter
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers
        self.seq_size = args.seq_size if args is not None and hasattr(args, 'seq_size') else 0

        self.basedir = os.path.dirname(root)
        self.sequence_name = os.path.basename(root).replace('_data.hdf5', '')

        # Event Dataset
        event_module = getattr(event, event_cfg.NAME)
        self.event_dataset = event_module.EventDataset(root=self.basedir, seq_name=self.sequence_name, **event_cfg.PARAMS)

        # Disparity Dataset
        disparity_module = getattr(disparity, disparity_cfg.NAME)
        self.disparity_dataset = disparity_module.DisparityDataset(root=self.basedir, seq_name=self.sequence_name, seq_size=self.seq_size, **disparity_cfg.PARAMS)

        # Timestamps
        if split in ['train', 'validation', 'val_indoor', 'val_outdoor_day', 'val_outdoor_night', 'proxy']:
            self.timestamps = copy.copy(self.disparity_dataset.timestamps)
            if event_cfg.NAME != 'none':
                minimum_timestamp = max(self.event_dataset.event_slicer['left'].min_time,
                                        self.event_dataset.event_slicer['right'].min_time,)
                maximum_timestamp = min(self.event_dataset.event_slicer['left'].max_time,
                                        self.event_dataset.event_slicer['right'].max_time)
            
                self.timestamps = self.timestamps[self.timestamps >= minimum_timestamp]
                self.timestamps = self.timestamps[self.timestamps <= maximum_timestamp]
        else:
            raise NotImplementedError

        self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]
        
        if split_filter is not None and len(split_filter) == 2:
            _idx_min = max(0, split_filter[0])
            _idx_max = min(len(self.timestamps), split_filter[1]) if split_filter[1] >= 0 else len(self.timestamps)
            self.timestamps = self.timestamps[_idx_min:_idx_max]

        # Transforms
        if split in ['train', 'proxy']:
            self.transforms = transforms.Compose([
                # Need to resize first before cropping cause the original size is too small for cropping
                transforms.Resize(event_module=event_module,
                                   disparity_module=disparity_module,
                                   img_height=MVSEC_HEIGHT*2, img_width=MVSEC_WIDTH*2),
                transforms.RandomCrop(event_module=event_module,
                                      disparity_module=disparity_module,
                                      crop_height=crop_height, crop_width=crop_width),
                # transforms.RandomVerticalFlip(event_module=event_module,
                #                               disparity_module=disparity_module, ),
                transforms.ToTensor(event_module=event_module,
                                    disparity_module=disparity_module, ),
            ])
        elif split in ['validation', 'val_indoor', 'val_outdoor_day', 'val_outdoor_night', 'test']:
            self.transforms = transforms.Compose([
                transforms.Padding(event_module=event_module,
                                   disparity_module=disparity_module,
                                   img_height=crop_height, img_width=crop_width,
                                   no_event_value=self.event_dataset.NO_VALUE,
                                   no_disparity_value=self.disparity_dataset.NO_VALUE),
                transforms.ToTensor(event_module=event_module,
                                    disparity_module=disparity_module, ),
            ])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        data = self.load_data(idx)

        data = self.transforms(data)

        return data

    def collate_fn(self, batch):
        output = {}
        # Event
        domain = 'event'
        if domain in batch[0].keys():
            output[domain] = self.event_dataset.collate_fn([sample[domain] for sample in batch])

        # Disparity
        for domain in ['disp', 'alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
            if domain in batch[0].keys():
                output[domain] = self.disparity_dataset.collate_fn([sample[domain] for sample in batch])

        # Others
        for key in batch[0].keys():
            if key not in output:
                output[key] = torch.utils.data._utils.collate.default_collate([sample[key] for sample in batch]) # type: ignore

        return output

    def load_data(self, idx):
        timestamp = self.timestamps[idx]
        data = {}
        
        disparity_data = self.disparity_dataset[timestamp]

        if disparity_data is not None:
            # Copy the dictionary into data
            for key in disparity_data.keys():
                data[key] = disparity_data[key]

        event_data = self.event_dataset[{'timestamp':timestamp}]

        if event_data is not None:
            data['event'] = event_data

        return data

