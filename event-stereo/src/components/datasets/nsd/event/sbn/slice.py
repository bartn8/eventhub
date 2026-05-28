import math
import h5py
import os

# Do not remove
import hdf5plugin

from typing import Dict, Tuple
from numba import jit

import numpy as np

import torch.utils.data


MIN_P = None
MAX_P = None

class EventSlicer(torch.utils.data.Dataset):
    def __init__(self, event_root, num_of_event, width, height, **kwargs):
        self.event_root = event_root
        self.num_of_event = num_of_event        
        self.width = width 
        self.height = height
        
    def __len__(self):
        return 0

    def __getitem__(self, ts_end):
        try:
            event_data = self.get_events_base_number(ts_end)
        except OSError as e:
            raise OSError(f"{str(e)}; filename: {self.event_root}") from e

        return event_data

    def get_events_base_number(self,  t_end_ns: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_end_ns: end time in nanoseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """

        # This should not happen, but just in case
        h5_path = os.path.join(self.event_root, f"{t_end_ns}.h5")
        if not os.path.isfile(h5_path):
            events = dict()
            events['t'] = np.array([], dtype=np.int64)
            events['x'] = np.array([], dtype=np.int16)
            events['y'] = np.array([], dtype=np.int16)
            events['p'] = np.array([], dtype=np.int8)
            return events

        with h5py.File(h5_path, 'r') as h5f:
            t_end_ns_idx = len(h5f['t']) # type: ignore
            t_start_ns_idx = max(0, t_end_ns_idx - self.num_of_event)
        
            events = dict()
            # with h5py.File(self.event_root, 'r') as h5f:
            events['t'] = np.asarray(h5f['t'][t_start_ns_idx:t_end_ns_idx])# type: ignore
            for dset_str in ['p', 'x', 'y']:
                events[dset_str] = np.asarray(h5f[dset_str][t_start_ns_idx:t_end_ns_idx])# type: ignore
                assert events[dset_str].size == events['t'].size

            #Normalize polarity to [0, 1]
            if len(events['p']) > 0:
                global MIN_P, MAX_P
                if MIN_P is None:
                    MIN_P = events['p'].min()
                    # print(f"MIN_P initialized to {MIN_P}")
                if MAX_P is None:
                    MAX_P = events['p'].max()
                    # print(f"MAX_P initialized to {MAX_P}")

                events['p'] = (events['p'] - MIN_P) / (MAX_P - MIN_P)

        return events
