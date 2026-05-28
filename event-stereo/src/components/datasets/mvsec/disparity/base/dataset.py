import os
from PIL import Image
import base64
import io
import h5py
import numpy as np

import torch.utils.data

MVSEC_CAR_MASK_B64 = "iVBORw0KGgoAAAANSUhEUgAAAVoAAAEECAAAAABMvqMtAAAACXBIWXMAAC4jAAAuIwF4pT92AAACV0lEQVR42u3Yu1IDMRBFQcu1///LQwAZAYVLV88+CZnxtqe0klq9lOmNAC1aoUWLVmjRohVatGiFFi1aoUWLVmjRohVatGiFFi1aoUUrtGjRCi1atEKLFq3QokUrtGjRCi1atEKLFq3QokUrtGjRCi1aoUWLVmjRohVatGiFFi1aoUWLVmjRohVatGiFFi1aoUWLVmjRCi1atEKLFq3QokUrtGjRCi1atEKLFq3QokUrtGjRCi1atEKLVmjRohVatGiFFi1aoUWLVmjRohVatGiFFi1aoUWLVmjRohVatEKLFq3QokUrtGjRCi1atEKLFq3QokUrtGjRCi1atEKLFq3QohVatGiFFi1aoUWLVmjRohVatGiFdqGeI5+qff+puV+izpH8q0Lbi3Iy7g60rd9H1c/n1Y20bdQ/qqNp29xfsQ6hba8Fq71plzQdYZumXRo2a5ulXR02apuk3QA2afu+XTb3NWNTuwlscG6f62FjPWA3WhD2g609ptbEpnYIZEMLwq6wtTrtxhNbS9PuvRTUurTbr7G1KO0JL6/utm+yqYfoMLXH7LdqMdqTNrJ9bR+wa57GDpNty0ytkU1N7YGyXR/p49fYqSNb06f22MWgTZ7ao1fZmkh7+uurptFesDGoKbR37Lh62L7Jph7zf1N70SGhxk7tTcevNnJqrzvX1iDaGy8MasiCcOVVTBswtdfecVWa9ubbw0rSXn4tWyFat92f2/6mpdnJFmQMF23MFm0MF20MF21MF21MF20MF21MF20MF21MF20MF+2EI7AkSZIkKdcXTZpQvSuRJ2sAAAAASUVORK5CYII="
MVSEC_CAR_MASK = np.array(Image.open(io.BytesIO(base64.b64decode(MVSEC_CAR_MASK_B64)))) > 0
assert MVSEC_CAR_MASK.shape == (260, 346)

class DisparityDataset(torch.utils.data.Dataset):
    NO_VALUE = 0.0
    
    FOCAL_LENGTH_X_BASELINE = {
            'indoor_flying1': 19.941772,
            'indoor_flying2': 19.941772,
            'indoor_flying3': 19.941772,
            'indoor_flying4': 19.941772,
            'outdoor_night1': 19.651191,
            'outdoor_day1': 19.635287,
            'outdoor_day2': 19.635287,
            'motorcycle1':  19.430372,
        }
    
    MIN_DEPTH_DICT ={
        'indoor_flying1': 0.125,
        'indoor_flying2': 0.125,
        'indoor_flying3': 0.125,
        'indoor_flying4': 0.125,
        # 'outdoor_day1': 5.0,
        # 'outdoor_day2': 5.0,
        # 'outdoor_night1': 5.0,
        # 'motorcycle1':  5.0,
        'outdoor_day1': 0.125,
        'outdoor_day2': 0.125,
        'outdoor_night1': 0.125,
        'motorcycle1':  0.125,
    }
    
    MASK_DICT = {
        'indoor_flying1': None,
        'indoor_flying2': None,
        'indoor_flying3': None,
        'indoor_flying4': None,
        'outdoor_night1': MVSEC_CAR_MASK,
        'outdoor_day1': MVSEC_CAR_MASK,
        'outdoor_day2': MVSEC_CAR_MASK,
        'motorcycle1':  None,
    }

    def __init__(self, root, seq_name, seq_size):
        self.root = root
        self.seq_name = seq_name
        self.seq_size = seq_size
        self.gt_path = os.path.join(root, f'{seq_name}_gt.hdf5')
        self.data_path = os.path.join(root, f'{seq_name}_data.hdf5')
        
        with h5py.File(self.gt_path, 'r') as h5f:
            self.timestamps = h5f['davis/left/depth_image_rect_ts'][:].astype('float64')
            self.timestamp_to_index = {ts: i for i, ts in enumerate(self.timestamps)}

        if self.seq_size > 0:
            mysize = min(len(self.timestamps), self.seq_size)
            self.timestamps = self.timestamps[:mysize]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        disp = self.load_disp(timestamp)

        return {'disp': disp, 'alpha': np.ones_like(disp), 'myconfidence': np.ones_like(disp),
                'ao': np.ones_like(disp), 'sizeconf': np.ones_like(disp),
                'rgb_L': np.zeros_like(disp).repeat(3,axis=0), 'rgb_R': np.zeros_like(disp).repeat(3,axis=0), 'rgb_C': np.zeros_like(disp).repeat(3,axis=0)}

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch) # type: ignore
        return batch

    def load_disp(self, timestamp):
        with h5py.File(self.gt_path, 'r') as h5f:
            index = self.timestamp_to_index[timestamp]
            tensor = h5f['davis/left/depth_image_rect'][index]

        if tensor.ndim == 2:  # If it's a single channel image
            tensor = np.expand_dims(tensor, axis=-1)

        if tensor.ndim == 3:
            # H x W x C -> C x H x W
            tensor = np.transpose(tensor, (2, 0, 1))
            
        #Remove nan and inf
        tensor = np.nan_to_num(tensor, nan=self.NO_VALUE, posinf=self.NO_VALUE, neginf=self.NO_VALUE)
        #clip negative values
        tensor[tensor < 0] = self.NO_VALUE
        # #clip large values
        tensor[tensor > 40] = self.NO_VALUE
        
        #use min depth dict
        min_depth = self.MIN_DEPTH_DICT[self.seq_name]
        tensor[tensor < min_depth] = self.NO_VALUE
        
        #use mask dict
        mask = self.MASK_DICT[self.seq_name]
        if mask is not None:
            assert mask.shape == tensor.shape[1:], f"Mask shape {mask.shape} does not match tensor shape {tensor.shape[1:]}"
            tensor[:, ~mask] = self.NO_VALUE
        
        #Convert depth to disparity
        disp = tensor.copy()
        mask = disp > 0
        disp[mask] = self.FOCAL_LENGTH_X_BASELINE[self.seq_name] / disp[mask]

        return disp
