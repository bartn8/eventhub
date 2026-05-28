import os
import csv
import copy

import time

import numpy as np

import torch.utils.data

from . import disparity
from . import event
from ..utils import transforms
from .constant import SCANNET_HEIGHT, SCANNET_WIDTH

class SequenceDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'event': '',
        'disparity': '',
    }

    def __init__(self, root, baseline, split, sampling_ratio, event_cfg, disparity_cfg, 
                 crop_height, crop_width, width=None, height=None, args=None, **kwargs):
        self.root = root
        self.baseline = baseline
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.width = width if width is not None else SCANNET_WIDTH
        self.height = height if height is not None else SCANNET_HEIGHT
        self.seq_size = args.seq_size if args is not None and hasattr(args, 'seq_size') else 0

        self.sequence_name = root.split('/')[-1]

        # Event Dataset
        event_module = getattr(event, event_cfg.NAME)
        event_root = os.path.join(root, self._PATH_DICT['event'])
        
        # start_time = time.time()
        self.event_dataset = event_module.EventDataset(root=event_root, baseline=baseline, width=self.width, height=self.height, **event_cfg.PARAMS)
        # end_time = time.time()
        # print(f"Event dataset loaded in {end_time - start_time:.2f} seconds")

        # Disparity Dataset
        disparity_module = getattr(disparity, disparity_cfg.NAME)
        # start_time = time.time()
        disparity_root = os.path.join(root, self._PATH_DICT['disparity'])
        # end_time = time.time()
        # print(f"Disparity dataset loaded in {end_time - start_time:.2f} seconds")

        self.disparity_dataset = disparity_module.DisparityDataset(root=disparity_root, baseline=baseline, seq_size=self.seq_size, width=self.width, height=self.height, **disparity_cfg.PARAMS)

        # Timestamps
        if split in ['train', 'validation', 'curation']:
            self.timestamps = self.disparity_dataset.timestamps
            # self.timestamps = copy.copy(self.disparity_dataset.timestamps)
            # if event_cfg.NAME != 'none':
            #     self.timestamps = self.timestamps[self.timestamps >= 0]

            # self.timestamp_to_index = copy.copy(self.disparity_dataset.timestamp_to_index)
        else:
            raise NotImplementedError

        if sampling_ratio > 1:
            self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]

        # Transforms
        if split in ['train']:
            self.transforms = transforms.Compose([
                transforms.RandomResize(event_module=event_module,
                                  disparity_module=disparity_module,
                                  min_img_height=crop_height, min_img_width=crop_width,
                                  img_height=self.height, img_width=self.width,),
                transforms.RandomCrop(event_module=event_module,
                                      disparity_module=disparity_module,
                                      crop_height=crop_height, crop_width=crop_width),
                # transforms.RandomMotionBlur(event_module=event_module, motion_blur_prob=0.05),
                # transforms.RandomVerticalFlip(event_module=event_module,
                #                               disparity_module=disparity_module, v_flip_prob=0.05),
                transforms.ToTensor(event_module=event_module,
                                    disparity_module=disparity_module, ),
            ])
        elif split in ['validation', 'curation']:
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

        # data['file_index'] = self.timestamp_to_index[timestamp]

        return data
