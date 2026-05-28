import numpy as np
import torch.utils.data


class EventSlicerMock(torch.utils.data.Dataset):
    def __init__(self, event_root = None, rectify_map_root = None, num_of_event = None, **kwargs):
        
        self.ms_to_idx = None
        self.t_offset = 0
        self.t_final = 1
        self.min_time = 0
        self.max_time = 1
        self.total_event = 0

    def __len__(self):
        return 0

    def __getitem__(self, ts_end):
        events = dict()
        events['t'] = np.array([], dtype=np.int64)
        events['x'] = np.array([], dtype=np.int32)
        events['y'] = np.array([], dtype=np.int32)
        events['p'] = np.array([], dtype=np.int8)
        return events
