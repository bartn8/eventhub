import torch.utils.data
import numpy as np


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'event': 'event',
        'image': 'image'
    }
    _DOMAIN = ['event', 'image']
    NO_VALUE = 0.0

    def __init__(self, root, **kwargs):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, timestamp):
        return {'disp': np.zeros((1, 480, 640))}

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch
