import os
import csv
import copy

import numpy as np

import torch.utils.data

from . import disparity
from . import event
from ..utils import transforms
from .constant import M3ED_HEIGHT, M3ED_WIDTH


class SequenceDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'event': 'events',
        'disparity': 'disparity',
    }
    HEIGHT = M3ED_HEIGHT
    WIDTH = M3ED_WIDTH

    def __init__(self, root, split, sampling_ratio, event_cfg, disparity_cfg, 
                 crop_height, crop_width, num_workers=0, args=None):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers
        self.seq_size = args.seq_size if args is not None and hasattr(args, 'seq_size') else 0

        self.sequence_name = root.split('/')[-1]

        # Event Dataset
        event_module = getattr(event, event_cfg.NAME)
        event_root = os.path.join(root, self._PATH_DICT['event'])
        self.event_dataset = event_module.EventDataset(root=event_root, **event_cfg.PARAMS)

        # Disparity Dataset
        disparity_module = getattr(disparity, disparity_cfg.NAME)
        disparity_root = os.path.join(root, self._PATH_DICT['disparity'])
        self.disparity_dataset = disparity_module.DisparityDataset(root=disparity_root, seq_size=self.seq_size,  **disparity_cfg.PARAMS)

        # Timestamps
        if split in ['val_indoor', 'val_outdoor_day', 'val_outdoor_night']:
            self.timestamps = copy.copy(self.disparity_dataset.timestamps)

            if event_cfg.NAME != 'none':
                minimum_timestamp = max(self.event_dataset.event_slicer['left'].t_offset,
                                        self.event_dataset.event_slicer['right'].t_offset)
                maximum_timestamp = None
                
                if event_cfg.NAME == 'sbn':
                    minimum_timestamp = max(self.event_dataset.event_slicer['left'].min_time,
                                            self.event_dataset.event_slicer['right'].min_time,)
                    maximum_timestamp = min(self.event_dataset.event_slicer['left'].max_time,
                                            self.event_dataset.event_slicer['right'].max_time)
                else:
                    raise NotImplementedError(f"Event dataset {event_cfg.NAME} not implemented for timestamp filtering.")
                    
                self.timestamps = self.timestamps[self.timestamps >= minimum_timestamp]
                if maximum_timestamp is not None:
                    self.timestamps = self.timestamps[self.timestamps <= maximum_timestamp]
        else:
            raise NotImplementedError

        self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]

        # Transforms
        # if split in ['train', 'trainval']:
        #     self.transforms = transforms.Compose([
        #         transforms.RandomCrop(event_module=event_module,
        #                               disparity_module=disparity_module,
        #                               crop_height=crop_height, crop_width=crop_width),
        #         transforms.RandomVerticalFlip(event_module=event_module,
        #                                       disparity_module=disparity_module, ),
        #         transforms.ToTensor(event_module=event_module,
        #                             disparity_module=disparity_module, ),
        #     ])
        if split in ['val_indoor', 'val_outdoor_day', 'val_outdoor_night']:
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
        for domain in ['disp']:
            if domain in batch[0].keys():
                output[domain] = self.disparity_dataset.collate_fn([sample[domain] for sample in batch])

        # Others
        for key in batch[0].keys():
            if key not in output:
                output[key] = torch.utils.data._utils.collate.default_collate([sample[key] for sample in batch])

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
