import os
from PIL import Image
import numpy as np
import yaml

import torch.utils.data


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'event': 'event',
        'disp_factor': 'disp_factor.yaml',
        'sample_mask': 'sample_mask.txt'
    }
    _DOMAIN = ['event']
    NO_VALUE = 0.0

    def __init__(self, root, seq_size):
        self.root = root
        self.seq_size = seq_size
        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))

        if os.path.exists(os.path.join(root, self._PATH_DICT['disp_factor'])):
            with open(os.path.join(root, self._PATH_DICT['disp_factor']), 'r') as f:
                self.disp_factor = yaml.safe_load(f)
        else:
            self.disp_factor = None

        if os.path.exists(os.path.join(root, self._PATH_DICT['sample_mask'])):
            self.sample_mask = np.loadtxt(os.path.join(root, self._PATH_DICT['sample_mask']), dtype=bool)
        else:
            self.sample_mask = None

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

        #Filter accordingly to mask
        if self.sample_mask is not None:
            assert len(self.timestamps) == len(self.sample_mask), "Timestamps and sample mask must have the same length"
            self.timestamps = self.timestamps[self.sample_mask]

        if self.seq_size > 0:
            mysize = min(len(self.timestamps), self.seq_size)
            self.timestamps = self.timestamps[:mysize]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        disp = load_disparity(self.timestamp_to_disparity_path['event'][timestamp],self.disp_factor)

        return {'disp': disp,}

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch) #type: ignore
        return batch


def load_timestamp(root):
    return np.loadtxt(root, dtype='int64')


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_disparity(root, factor_dict = None):
    if factor_dict is not None:
        dir = os.path.dirname(root).split("/")[-1]
        filename = os.path.basename(root)
        return load_tensor(root, scale_factor=factor_dict[dir][filename])
    else:
        return load_tensor(root, scale_factor=256.0)

def load_tensor(root, scale_factor=256.0):
    tensor = np.array(Image.open(root)).astype(np.float32) / scale_factor

    if tensor.ndim == 2:  # If it's a single channel image
        tensor = np.expand_dims(tensor, axis=-1)

    if tensor.ndim == 3:
        # H x W x C -> C x H x W
        tensor = np.transpose(tensor, (2, 0, 1))

    return tensor
