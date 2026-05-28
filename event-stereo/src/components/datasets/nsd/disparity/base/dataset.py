import os
from PIL import Image
import numpy as np
import cv2
import json

import torch.utils.data

class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'intrinsics': 'intrinsics.txt',
        'curation_epe': 'curation_EPE',
        'depth': 'rendered_frames_depth_median',
        'ao': 'rendered_frames_ao',
        'alpha': 'rendered_frames_alpha',
        'sizeconf': 'rendered_frames_sizeconf',
        'rgb': 'rendered_frames',
    }

    _SCALE_DICT = {
        'alpha_C': 65535.0,
        'ao_C': 65535.0,
        'sizeconf_C': 65535.0,
        'rgb_C': 255.0,
        'rgb_L': 255.0,
        'rgb_R': 255.0,
    }

    _OUTPUT_DICT = {
        'disparity_C': 'disp',
        'depth_C': 'depth',
        'alpha_C': 'alpha',
        'ao_C': 'ao',
        'sizeconf_C': 'sizeconf',
        'rgb_C': 'rgb_C',
        'rgb_L': 'rgb_L',
        'rgb_R': 'rgb_R',
    }
    
    NO_VALUE = 0.0

    # Do it during loss calculation
    # MIN_DISPARITY = 0.5
    # MAX_DISPARITY = 256.0

    def __init__(self, root, seq_size, baseline, width, height, curation_epe_threshold = -1, **kwargs):
        self.root = root
        self.seq_size = seq_size
        self.baseline = baseline
        self.width = width 
        self.height = height
        self.intrinsics = np.loadtxt(os.path.join(root, self._PATH_DICT['intrinsics']), dtype='float32')
        
        curation_path = os.path.join(root, f"{self._PATH_DICT['curation_epe']}_{baseline:.2f}.txt")
        
        if curation_epe_threshold > 0 and os.path.exists(curation_path):
            self.curation_epe_list = np.loadtxt(os.path.join(root, curation_path), dtype='float32')
        else: 
            self.curation_epe_list = None
        
        # self.timestamps = self.load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))
        # self.timestamps_to_idx = {t: i for i, t in enumerate(self.timestamps)}
        # self.timestamps_dict = {j:self.timestamps for j in ["R", "C", "L"]}

        #Check all timestamps: final list will be the intersection of all timestamps
        self.timestamps = None
        self.timestamps_dict = {}
        
        for i,j in zip([-1,0,1], ["R", "C", "L"]):
            _baseline = abs(self.baseline) * i
            shift_sign_str = "LC" if _baseline > 0 else "CR"
            shift_str = f"_{shift_sign_str}{abs(_baseline):.2f}" if _baseline != 0 else ""
            _tmp_path = os.path.join(root, self._PATH_DICT['timestamp'].replace(".txt", f"{shift_str}.txt"))

            if os.path.exists(_tmp_path):
                _tmp_timestamps = self.load_timestamp(_tmp_path)
                self.timestamps_dict[j] = _tmp_timestamps

                # convert numpy array to list and update the set
                _tmp_timestamps = set(_tmp_timestamps.tolist())
                self.timestamps = self.timestamps.intersection(_tmp_timestamps) if self.timestamps is not None else _tmp_timestamps
            else: 
                self.onetime_warning(f"Timestamp file {_tmp_path} not found. Skipping.", key=f'1_timestamp_{shift_str}')

        # Ensure all timestamps are present for L, C, R
        for i in ["L", "R"]:
            if i not in self.timestamps_dict:
                self.onetime_warning(f"Timestamp for {i} not found. Using C timestamps instead.", key=f'2_timestamp_{i}')
                self.timestamps_dict[i] = self.timestamps_dict['C']

        if self.timestamps is None:
            raise ValueError("No valid timestamps found.")

        self.timestamps = sorted(self.timestamps)
        # Convert timestamps to np.ndarray
        self.timestamps = np.array(self.timestamps, dtype='int64')

        self.timestamp_to_path = {}

        # Load paths for depth, ao, alpha, and sizeconf
        for i,j in zip(['ao', 'alpha', 'sizeconf', 'depth'], ['ao_C', 'alpha_C', 'sizeconf_C', 'depth_C']):
            _tmp_dir = os.path.join(root, f"{self._PATH_DICT[i]}")
            if os.path.exists(_tmp_dir):
                _tmp_dict = {}
                _tmp_dict[j] = self.get_path_list(_tmp_dir)
                self.timestamp_to_path[j] = {timestamp: filepath for timestamp, filepath in
                                                        zip(self.timestamps_dict["C"], _tmp_dict[j])}   

        assert 'depth_C' in self.timestamp_to_path, "Depth paths not found. Check the dataset structure."

        # Load paths for RGB images
        for i,j in zip([-1,0,1], ["R", "C", "L"]):
            _baseline = abs(self.baseline) * i
            shift_sign_str = "LC" if _baseline > 0 else "CR"
            shift_str = f"_{shift_sign_str}{abs(_baseline):.2f}" if _baseline != 0 else ""
            _tmp_dir = os.path.join(root, f"{self._PATH_DICT['rgb']}{shift_str}")

            if os.path.exists(_tmp_dir):
                _tmp_dict = {}
                _tmp_dict[f'rgb_{j}'] = self.get_path_list(_tmp_dir)
                self.timestamp_to_path[f'rgb_{j}'] = {timestamp: filepath for timestamp, filepath in
                                                        zip(self.timestamps_dict[j], _tmp_dict[f'rgb_{j}'])}
                
        # Remove duplicates
        # self.timestamps = np.array(list(set(self.timestamps.tolist())), dtype='int64')

        # Skip first frame (no events)
        # if len(self.timestamps) > 0:
        #     self.timestamps = self.timestamps[1:]
        
        # Curation based on EPE if available
        if self.curation_epe_list is not None:
            if len(self.curation_epe_list) == len(self.timestamps):
                self.timestamps = self.timestamps[self.curation_epe_list <= curation_epe_threshold]
            else:
                self.onetime_warning(f"Curation EPE list {curation_path} length {len(self.curation_epe_list)} does not match timestamps length {len(self.timestamps)}. Skipping curation.", key='curation_length_mismatch')

        # Default way: select first seq_size timestamps
        if self.seq_size > 0:
            mysize = min(len(self.timestamps), self.seq_size)
            self.timestamps = self.timestamps[:mysize]

        # Alternative way: select random seq_size timestamps
        # if self.seq_size > 0:
        #     if not len(self.timestamps) < self.seq_size:    
        #         # Randomly select seq_size timestamps
        #         indices = np.random.choice(len(self.timestamps), self.seq_size, replace=False)
        #         # Sort indices to maintain order
        #         indices.sort()
        #         self.timestamps = self.timestamps[indices]

    def __len__(self):
        return len(self.timestamps) # type:ignore

    def __getitem__(self, timestamp):

        output_dict = {}
        assert self._OUTPUT_DICT['disparity_C'] == 'disp', "Output dictionary key mismatch for disparity_C"

        # Load Depth assume depth_C is always present
        _disp = self.load_tensor(self.timestamp_to_path['depth_C'][timestamp], scale_factor=1000.0)
        # Convert depth to disparity
        _disp[_disp > 0] = (self.baseline * self.intrinsics[0,0]) / _disp[_disp > 0]
        # Apply scale and shift
        # _disp = self.scales[timestamp] * _disp + self.shifts[timestamp]
        
        # remove 1-th and 99-th percentiles
        # lower_bound = np.percentile(_disp[_disp > 0], 1)
        # upper_bound = np.percentile(_disp[_disp > 0], 99)
        # _disp = np.where((_disp >= lower_bound) & (_disp <= upper_bound), _disp, self.NO_VALUE)
        # _disp = np.where((_disp >= 0) & (_disp <= upper_bound), _disp, self.NO_VALUE)

        # if negative values are present, set them to NO_VALUE
        # _disp = np.where(_disp < self.MIN_DISPARITY, self.NO_VALUE, _disp)
        # # if values are greater than MAX_DISPARITY, set them to NO_VALUE
        # _disp = np.where(_disp > self.MAX_DISPARITY, self.NO_VALUE, _disp)

        for key in self._SCALE_DICT:
            if key in output_dict:
                continue

            if key in self.timestamp_to_path:
                output_dict[self._OUTPUT_DICT[key]] = self.load_tensor(self.timestamp_to_path[key][timestamp], scale_factor=self._SCALE_DICT[key])
            else:
                self.onetime_warning(f"No {key} found. Using zeros instead.", key=f'no_{key}')
                # Assume disparity_C is the base shape
                _tmp_tensor = np.zeros_like(_disp)
                output_dict[self._OUTPUT_DICT[key]] = _tmp_tensor if 'rgb' not in key else np.zeros((*_tmp_tensor.shape, 3), dtype=np.float32)

        # _floaters_remover = 1 - output_dict['ao']
        # _floaters_remover = np.where(_floaters_remover > 0.05, 1.0, 0.0)
        # _floaters_remover_shape = _floaters_remover.shape
        # _floaters_remover = cv2.medianBlur(_floaters_remover.squeeze().astype(np.uint8), 5).astype(np.float32)
        # _floaters_remover = np.reshape(_floaters_remover, _floaters_remover_shape)

        output_dict['myconfidence'] = output_dict['alpha'] * output_dict['sizeconf'] #* _floaters_remover
        # Threshold confidence hardcoded to 0.75 from ablation study
        output_dict['myconfidence'] = np.where(output_dict['myconfidence'] >= 0.75, output_dict['myconfidence'], 0.0)

        # output_dict['myconfidence'] = output_dict['ao']
        # # Threshold confidence hardcoded to 0.25 from NERFStereo supp material
        # output_dict['myconfidence'] = np.where(output_dict['myconfidence'] >= 0.25, output_dict['myconfidence'], 0.0)

        # Filter out invalid disparity values
        _disp = np.where(output_dict['myconfidence'] > 0, _disp, self.NO_VALUE)
        
        # # If density of disparity is too low -- i.e., 50% of H*W pixels are NO_VALUE, set disparity to NO_VALUE
        # _shape = _disp.squeeze().shape
        # H,W = _shape[-2], _shape[-1]
        # if np.sum(_disp == self.NO_VALUE) > 0.85 * H * W:
        #     _disp = np.full_like(_disp, self.NO_VALUE, dtype=np.float32)
        #     onetime_warning("Disparity density too low. Setting disparity to NO_VALUE.", key='low_density')

        output_dict['disp'] = _disp

        return output_dict

    @staticmethod
    def collate_fn(batch):
        # try:
        batch = torch.utils.data._utils.collate.default_collate(batch) # type: ignore
        # except RuntimeError as e:
        #     shapes = [item[0].shape for item in batch]
        #     print(f"Batch: {shapes} failed to collate with error: {str(e)}")
        #     raise e
        return batch

    def load_timestamp(self, root):
        _tmp = np.loadtxt(root, dtype='int64')
        if _tmp.ndim == 0:
            _tmp = np.array([_tmp], dtype='int64')
        return _tmp

    def get_path_list(self, root):
        return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]

    # def load_depth(self, root):
    #     try:
    #         tensor = np.load(root)['depth']
    #     except PermissionError as e:
    #         print(f"Permission denied for {root}. Trying again...")
    #         tensor = np.load(root)['depth']

    #     if tensor.ndim == 2:  # If it's a single channel image
    #         tensor = np.expand_dims(tensor, axis=-1)
            
    #     if tensor.ndim == 3:
    #         # H x W x C -> C x H x W
    #         tensor = np.transpose(tensor, (2, 0, 1))
            
    #     assert tensor.shape[0] == 1, "Depth data should have a single channel"
    #     assert tensor.shape[1] == self.height and tensor.shape[2] == self.width, f"{root}: Depth data shape mismatch: {tensor.shape}, expected ({self.height}, {self.width})"

    #     return tensor.astype(np.float32)

    def load_tensor(self, root, scale_factor=256.0):
        try:
            data = Image.open(root)
        except PermissionError as e:
            print(f"Permission denied for {root}. Trying again...")
            data = Image.open(root)

        tensor = np.array(data).astype(np.float32) / scale_factor

        if tensor.ndim == 2:  # If it's a single channel image
            tensor = np.expand_dims(tensor, axis=-1)

        if tensor.ndim == 3:
            # H x W x C -> C x H x W
            tensor = np.transpose(tensor, (2, 0, 1))
            
        assert tensor.ndim == 3, f"Expected tensor to have 3 dimensions, got {tensor.ndim}"
        assert tensor.shape[0] in [1, 3], f"Expected tensor to have 1 or 3 channels, got {tensor.shape[0]}"
        assert tensor.shape[1] == self.height and tensor.shape[2] == self.width, f"{root}: Tensor shape mismatch: {tensor.shape}, expected (C, {self.height}, {self.width})"

        return tensor

    def onetime_warning(self, x, key="default"):
        if not hasattr(self, "onetime_warning_has_warned"):
            self.onetime_warning_has_warned = {}

        if key not in self.onetime_warning_has_warned:
            self.onetime_warning_has_warned[key] = True
            print(f"Warning: {x}")