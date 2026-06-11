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
    def __init__(self, root, seq_name, num_of_event, width, height, location):
        self.event_root = root
        self.seq_name = seq_name
        self.gt_path = os.path.join(root, f'{seq_name}_gt.hdf5')
        self.data_path = os.path.join(root, f'{seq_name}_data.hdf5')
        self.num_of_event = num_of_event
        self.width = width 
        self.height = height
        self.location = location
        
        assert self.location in ['left', 'right'], "Location must be 'left' or 'right'"
                
        # Carica solo i timestamp per l'indicizzazione
        with h5py.File(self.data_path, 'r') as h5f:
            _timestamps = h5f[f'davis/{self.location}/events'][:, 2].astype(np.float64)  # Solo colonna t
            self.total_event = len(_timestamps)
            self.min_time = _timestamps[0]
            self.max_time = _timestamps[-1]
            # Offset to make left/right timestamps start from a round number
            # Assume that both left and right have a similar starting timestamp up to 1000s
            self.t_offset = (_timestamps[0] // 1000) * 1000 
            
            # load t for caching
            self.t = _timestamps

        self.rectify_map = self.load_rectification_maps()

    def __len__(self):
        return 0

    def __getitem__(self, ts_end):
        try:
            event_data = self.get_events_base_number(self.num_of_event, ts_end)
        except OSError as e:
            raise OSError(f"{str(e)}; filename: {self.event_root}") from e

        rectified_events = self.rectify_events(event_data)

        return rectified_events

    def get_events_base_number(self, number_of_event: int,  t_end_ns: float) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        number_of_event: number of events
        t_end_ns: end time in nanoseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        
        with h5py.File(self.data_path, 'r') as h5f:
            # davis/left/events: a [N,4] list of (x,y,t,p)  
            # x = h5f[f'davis/{self.location}/events'][:,0].astype(np.int32)      # type: ignore
            # y = h5f[f'davis/{self.location}/events'][:,1].astype(np.int32)      # type: ignore
            # t = h5f[f'davis/{self.location}/events'][:,2].astype(np.float64)    # type: ignore
            # p = h5f[f'davis/{self.location}/events'][:,3].astype(np.int8)       # type: ignore
            
            t = self.t
            
            t_end_ns_idx = np.searchsorted(t, t_end_ns, side='right')# type: ignore
            t_start_ns_idx = max(0, t_end_ns_idx - number_of_event)
            t_end_ns_idx = min(self.total_event, t_end_ns_idx)# type: ignore

            events = dict()
            events['t'] = np.asarray(t[t_start_ns_idx:t_end_ns_idx], dtype=np.uint64)  # type: ignore
            events['x'] = np.asarray(h5f[f'davis/{self.location}/events'][t_start_ns_idx:t_end_ns_idx,0]).astype(np.int32)  # type: ignore
            events['y'] = np.asarray(h5f[f'davis/{self.location}/events'][t_start_ns_idx:t_end_ns_idx,1]).astype(np.int32)  # type: ignore
            events['p'] = np.asarray(h5f[f'davis/{self.location}/events'][t_start_ns_idx:t_end_ns_idx,3]).astype(np.int8)  # type: ignore
            
            # Convert timestamp
            events['t'] = events['t'] - np.uint64(self.t_offset)
            events['t'] = (events['t'] * np.uint64(1e6)).astype(np.uint64)  # Convert to us unit

            #Normalize polarity to [0, 1]
            if len(events['p']) > 0:
                global MIN_P, MAX_P
                if MIN_P is None:
                    MIN_P = events['p'].min()
                if MAX_P is None:
                    MAX_P = events['p'].max()

                events['p'] = (events['p'] - MIN_P) / (MAX_P - MIN_P)
            # else:
            #     print(f"Warning: No events found in the time window {t_start_ns} ns to {t_end_ns} ns (root: {self.event_root})")

            return events
        
    def rectify_events(self, event_data):
        xy_rect = self.rectify_map[event_data['y'], event_data['x']]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        mask = (0 <= x_rect) & (x_rect < self.width) & (0 <= y_rect) & (y_rect < self.height)

        return {
            'x': x_rect[mask],
            'y': y_rect[mask],
            't': event_data['t'][mask],
            'p': event_data['p'][mask],
        }        
        
    def load_rectification_maps(self):
        """Load and invert rectification maps from calibration files."""
        _filename = os.path.basename(self.event_root)
        left_x_map_file = os.path.join(self.event_root, 'calib', f"{_filename}_left_x_map.txt")
        left_y_map_file = os.path.join(self.event_root, 'calib', f"{_filename}_left_y_map.txt")
        right_x_map_file = os.path.join(self.event_root, 'calib', f"{_filename}_right_x_map.txt")
        right_y_map_file = os.path.join(self.event_root, 'calib', f"{_filename}_right_y_map.txt")

        left_x_map = np.loadtxt(left_x_map_file).astype(np.float32)
        left_y_map = np.loadtxt(left_y_map_file).astype(np.float32)
        right_x_map = np.loadtxt(right_x_map_file).astype(np.float32)
        right_y_map = np.loadtxt(right_y_map_file).astype(np.float32)
        
        # Combine x and y maps 
        left_map_combined = np.stack([left_x_map, left_y_map], axis=-1).round().astype(np.int32)
        right_map_combined = np.stack([right_x_map, right_y_map], axis=-1).round().astype(np.int32)

        return left_map_combined if self.location == 'left' else right_map_combined
        