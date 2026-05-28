import os
from PIL import Image
import yaml
from pathlib import Path
import numpy as np

import torch.utils.data

class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'event': 'event',
    }
    _DOMAIN = ['event']
    NO_VALUE = 0.0

    def __init__(self, root, seq_size):
        self.root = root
        self.seq_size = seq_size
        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))

        self.disparity_path_list = {}
        self.timestamp_to_disparity_path = {}
        for domain in self._DOMAIN:            
            self.disparity_path_list[domain] = get_path_list(os.path.join(root, self._PATH_DICT[domain]))

            self.timestamp_to_disparity_path[domain] = {timestamp: filepath for timestamp, filepath in
                                                        zip(self.timestamps, self.disparity_path_list[domain])}
        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_disparity_path['event'][timestamp]))[0])
            for timestamp in self.timestamp_to_disparity_path['event'].keys()
        }

        if self.seq_size > 0:
            mysize = min(len(self.timestamps), self.seq_size)
            self.timestamps = self.timestamps[:mysize]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        disp = load_tensor(self.timestamp_to_disparity_path['event'][timestamp])

        return {'disp': disp, 'alpha': np.ones_like(disp), 'myconfidence': np.ones_like(disp),
                'ao': np.ones_like(disp), 'sizeconf': np.ones_like(disp),
                'rgb_L': np.zeros_like(disp).repeat(3,axis=0), 'rgb_R': np.zeros_like(disp).repeat(3,axis=0), 'rgb_C': np.zeros_like(disp).repeat(3,axis=0)}

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch) # type: ignore
        return batch


def load_timestamp(root):
    return np.loadtxt(root, dtype='int64')


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_tensor(root, scale_factor=256.0):
    tensor = np.array(Image.open(root)).astype(np.float32) / scale_factor

    if tensor.ndim == 2:  # If it's a single channel image
        tensor = np.expand_dims(tensor, axis=-1)

    if tensor.ndim == 3:
        # H x W x C -> C x H x W
        tensor = np.transpose(tensor, (2, 0, 1))

    return tensor
