import os
import tqdm

import torch.utils.data
import torch.utils.data._utils


from utils.dataloader import MultiEpochsDataLoader
from .sequence import SequenceDataset
from .constant import DATA_SPLIT, FRAMES_FILTER


class MVSECDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, sampling_ratio,
                 event_cfg, disparity_cfg,
                 crop_height, crop_width,
                 num_workers=0, args=None, **kwargs):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers
        self.args = args
        assert split in DATA_SPLIT.keys()

        sequence_list = DATA_SPLIT[split]
        self.sequence_data_list = []
        for i in tqdm.tqdm(range(len(sequence_list)), desc=f"Loading {split} sequences"):
            sequence = sequence_list[i]
            filter = FRAMES_FILTER.get(sequence, [0, -1])
            sequence_root = os.path.join(root, sequence)
            self.sequence_data_list.append(SequenceDataset(root=sequence_root,
                                                           split=split,
                                                           split_filter=filter,
                                                           sampling_ratio=sampling_ratio,
                                                           event_cfg=event_cfg,
                                                           disparity_cfg=disparity_cfg,
                                                           crop_height=crop_height,
                                                           crop_width=crop_width,
                                                           num_workers=num_workers,
                                                           args=self.args,
                                                           **kwargs))

        if len(self.sequence_data_list) == 0:
            self.dataset = []
        else:
            self.dataset = torch.utils.data.ConcatDataset(self.sequence_data_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        return data

    def collate_fn(self, batch):
        return self.dataset.datasets[0].collate_fn(batch) # type: ignore

def get_dataset(args, dataset_cfg):
    dataset = MVSECDataset(root=args.data_root if args.data_root is not None else dataset_cfg.PATH,
                          num_workers=args.num_workers,
                          args=args,
                          **dataset_cfg.PARAMS)
    return dataset
