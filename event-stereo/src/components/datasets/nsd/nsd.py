import os
import tqdm
# import time
import numpy as np

import torch.utils.data
import torch.utils.data._utils

from .sequence import SequenceDataset
from .constant import DATA_SPLIT

def load_baselines(baseline_file):
    baselines = np.loadtxt(baseline_file)
    baselines = np.round(baselines, 2)
    return baselines

class NSDDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'baselines': 'baselines.txt',
        'timestamps': 'timestamps.txt',
    }
    
    def __init__(self, root, split, args=None, **kwargs):
        self.root = root
        self.split = split
        self.args = args
        assert split in DATA_SPLIT.keys()

        sequence_list = DATA_SPLIT[split]
        self.sequence_data_list = []
        _tmp_seq_list = []

        _missing_seq = []

        for sequence in tqdm.tqdm(sequence_list, desc=f"Checking {split} sequences"):
            sequence_root = os.path.join(root, sequence)
            # Check if the sequence root exists
            if not os.path.exists(sequence_root):
                # print(f"Warning: Sequence root {sequence_root} does not exist. Skipping.")
                _missing_seq.append(sequence_root.split('/')[-3])
                continue
            
            # Check if the sequence contains baselines.txt
            if not os.path.exists(os.path.join(sequence_root, self._PATH_DICT['baselines'])):
                _missing_seq.append(sequence_root.split('/')[-3])
                continue
            
            # if not os.path.exists(os.path.join(sequence_root, self._PATH_DICT['timestamps'])):
            #     _missing_seq.append(sequence_root.split('/')[-3])
            #     continue
            

            _tmp_seq_list.append(sequence)
            
        if len(_missing_seq) > 0:
            _missing_seq = sorted(list(set(_missing_seq)))
            _missing_seq = [f"{int(s):04}" for s in _missing_seq]
            print(f"Warning: The following sequences are missing and will be skipped: {_missing_seq}")
            
        sequence_list = _tmp_seq_list

        for sequence in tqdm.tqdm(sequence_list, desc=f"Loading {split} sequences"):
            sequence_root = os.path.join(root, sequence)
            # Iterate over all folders in the sequence root

            baselines = load_baselines(os.path.join(sequence_root, self._PATH_DICT['baselines']))
            baselines = sorted(baselines)

            for baseline in baselines:
                # start_time = time.time()
                self.sequence_data_list.append(SequenceDataset(root=sequence_root,
                                                                baseline=baseline,
                                                                split=split,
                                                                args=self.args,
                                                                **kwargs))
                # end_time = time.time()
                # print(f"Sequence {sequence} folder {folder} loaded in {end_time - start_time:.2f} seconds")

        if len(self.sequence_data_list) == 0:
            self.dataset = []
        else:
            self.dataset = torch.utils.data.ConcatDataset(self.sequence_data_list)
            
        print(f"Total sequences loaded for {split}: {len(self.sequence_data_list)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        return data

    def collate_fn(self, batch):
        return self.dataset.datasets[0].collate_fn(batch) # type: ignore


def get_dataset(args, dataset_cfg):
    dataset = NSDDataset(root=args.data_root if args.data_root is not None else dataset_cfg.PATH,
                          num_workers=args.num_workers,
                          args=args,
                          **dataset_cfg.PARAMS)

    return dataset
