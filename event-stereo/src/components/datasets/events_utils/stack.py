import torch
import numpy as np
from abc import ABC, abstractmethod
from .stack_utils import events2ToreFeature, ToTimesurface
from .mixed_density_event_stack import MixedDensityEventStack
from .dropout import EventDropout
from .noise import EventNoise
from .stack_utils import fast_to_eros, fast_to_teros
import pandas as pd

class EventStacking(ABC):
    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch) # type:ignore
        return batch
    
    def __init__(self, stack_size, height, width, normalize = True, num_of_event=0, num_of_event_subset = 0, dropout_rnd_state = None, dropout_params = None, noise_rnd_state = None, noise_params = None, **kwargs):
        self.stack_size = stack_size
        # self.num_of_event = num_of_event
        # self.num_of_event_subset = num_of_event_subset
        self.height = height
        self.width = width
        self.normalize = normalize
        
        if dropout_params is None:
            dropout_params = {
                "dropout_p": 0.0,
                "max_drop_count": 0,
                "max_slice_size": 0,
                "patch_size": 0
            }

        self.dropout = EventDropout(height=height, width=width, random_state=dropout_rnd_state, **dropout_params)
        
        if noise_params is None:
            noise_params = {
                "noise_p": 0.0,
                "max_noise_count": 0
            }

        self.noise = EventNoise(height=height, width=width, random_state=noise_rnd_state, **noise_params)

    @abstractmethod
    def make_stack(self, x, y, p, t):
        pass

    @abstractmethod
    def stack_data(self, x, y, p, t_s):
        pass

    @abstractmethod
    def make_empty_stack(self):
        pass

    def pre_stack(self, event_sequence, last_timestamp):
        x = event_sequence['x'].astype(np.int32)
        y = event_sequence['y'].astype(np.int32)
        p = event_sequence['p'].astype(np.int8)
        t = event_sequence['t'].astype(np.int64)

        assert len(x) == len(y) == len(p) == len(t)
        past_mask = t < last_timestamp
        # print("a",t.min(), t.max(), last_timestamp, np.sum(past_mask), len(t))
        p_x, p_y, p_p, p_t = x[past_mask], y[past_mask], p[past_mask], t[past_mask]

        # print("b",p_t.min(), p_t.max(), last_timestamp, len(p_t))
        
        # Apply noise if configured
        events_dict = self.noise.add_spatial_noise({'x': p_x, 'y': p_y, 'p': p_p, 't': p_t})
        p_x, p_y, p_p, p_t = events_dict['x'], events_dict['y'], events_dict['p'], events_dict['t']

        # Applica dropout se configurato
        events_dict = self.dropout.apply_random_dropout({'x': p_x, 'y': p_y, 'p': p_p, 't': p_t})
        p_x, p_y, p_p, p_t = events_dict['x'], events_dict['y'], events_dict['p'], events_dict['t']

        # print("c", p_t.min(), p_t.max(), last_timestamp, len(p_t))

        if np.sum(past_mask) == 0:
            past_stacked_event = self.make_empty_stack()
        else:
            p_x, p_y, p_p, p_t = x[past_mask], y[past_mask], p[past_mask], t[past_mask]
            p_t = p_t - p_t.min()
            past_stacked_event = self.make_stack(p_x, p_y, p_p, p_t)
            
            # Rollback
            # if self.num_of_event_subset > 0:
            #     sub_events_len = min(self.num_of_event_subset, len(t))
            #     x_sub, y_sub, p_sub, t_sub = p_x[-sub_events_len:], p_y[-sub_events_len:], p_p[-sub_events_len:], p_t[-sub_events_len:]
            #     t_sub = t_sub - t_sub.min()
            #     past_stacked_event_subset = self.make_stack(x_sub, y_sub, p_sub, t_sub)
            #     return past_stacked_event, past_stacked_event_subset

        return past_stacked_event

    # def post_stack(self, pre_stacked_event: np.ndarray | list | tuple):
    def post_stack(self, pre_stacked_event: np.ndarray):
        if not isinstance(pre_stacked_event, np.ndarray):
            raise ValueError("post_stack only supports numpy array input.")

        # if isinstance(pre_stacked_event, tuple):
        #     pre_stacked_event = list(pre_stacked_event)

        # if isinstance(pre_stacked_event, list):
        #     for i in range(len(pre_stacked_event)):
        #         pre_stacked_event[i] = pre_stacked_event[i].transpose(1, 2, 0)  # HxWxS
        #     return np.stack(pre_stacked_event, axis=2)  # HxWx2xS
            

        # return np.expand_dims(pre_stacked_event.transpose(1, 2, 0), axis=2)  # HxWx1xS
        return pre_stacked_event.transpose(1,2,0)

class MixedDensityEventStacking(EventStacking):
    NO_VALUE = 0.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_empty_stack(self):
        mixed = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)
        return mixed
    
    def make_stack(self, x, y, p, t):
        assert len(x) == len(y) == len(p) == len(t)
        t_s = t
        stacked_event_list = []
        cur_num_of_events = len(t_s)
        for _ in range(self.stack_size):
            stacked_event = self.stack_data(x, y, p, t_s)
            stacked_event_list.append(stacked_event)

            cur_num_of_events = cur_num_of_events // 2
            x = x[-cur_num_of_events:]
            y = y[-cur_num_of_events:]
            p = p[-cur_num_of_events:]
            t_s = t_s[-cur_num_of_events:]

        return np.stack(stacked_event_list, axis=0) #SxHxW

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        # assert np.all(0 <= x < self.width), f"Unexpected x values: {x[x<0]};{x[x>=self.width]}"
        # assert np.all(0 <= y < self.height), f"Unexpected y values: {y[y<0]};{y[y>=self.height]}"

        stacked_data = np.zeros([self.height, self.width], dtype=np.int8)

        index = (y * self.width) + x

        stacked_data.put(index, p)

        return stacked_data

class HistogramEventStacking(EventStacking):
    NO_VALUE = 0.

    def __init__(self, **kwargs):
        _kwargs = kwargs.copy()
        _kwargs['stack_size'] = 2
        super().__init__(**_kwargs)

    def make_empty_stack(self):
        histogram = np.zeros([2, self.height, self.width], dtype=np.float32)
        return histogram
    
    def make_stack(self, x, y, p, t):
        assert len(x) == len(y) == len(p) == len(t)
        return self.stack_data(x, y, p, t)
        
    def stack_data(self, x, y, p, t_s):
        stacked_data_pos = np.zeros([self.height, self.width], dtype=np.float32)
        flatten_stacked_data_pos = stacked_data_pos.ravel()
        stacked_data_neg = np.zeros([self.height, self.width], dtype=np.float32)
        flatten_stacked_data_neg = stacked_data_neg.ravel()

        pos_mask = p > 0
        neg_mask = p <= 0

        index_pos = (y[pos_mask] * self.width) + x[pos_mask]
        index_neg = (y[neg_mask] * self.width) + x[neg_mask]
        
        np.add.at(flatten_stacked_data_pos, index_pos, 1)
        np.add.at(flatten_stacked_data_neg, index_neg, 1)

        stacked_data_pos = np.reshape(flatten_stacked_data_pos, (self.height, self.width))
        stacked_data_neg = np.reshape(flatten_stacked_data_neg, (self.height, self.width))

        if self.normalize:#Channel normalization
            stacked_data_pos = (stacked_data_pos-stacked_data_pos.min()) / (stacked_data_pos.max()-stacked_data_pos.min())
            stacked_data_neg = (stacked_data_neg-stacked_data_neg.min()) / (stacked_data_neg.max()-stacked_data_neg.min())

        return np.stack([stacked_data_pos, stacked_data_neg], axis=0)#2xHxW
    
class VoxelGridEventStacking(EventStacking):
    NO_VALUE = 0.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_empty_stack(self):
        voxel_grid = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)
        return voxel_grid
    
    def make_stack(self, x, y, p, t):
        assert len(x) == len(y) == len(p) == len(t)
        time_interval = t.max() - t.min()
        t_s = ((t.astype(np.float32) - t.min()) / time_interval) * (self.stack_size-1)
        # polarity should be +1 / -1
        p = (p - p.min()) / (p.max() - p.min()) if p.max() - p.min() > 0 else p
        p = 2 * p - 1  # Convert to -1, 1
        return self.stack_data(x, y, p, t_s)

    def stack_data(self, x, y, p, t_s):
        voxel_grid = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)
        flatten_voxel_grid = voxel_grid.ravel()

        t_norm = (self.stack_size - 1) * (t_s - t_s[0]) / (t_s[-1] - t_s[0])
        
        for xlim in [x, x+1]:
            for ylim in [y, y+1]:
                for tlim in [t_norm, t_norm+1]:
                    mask = (xlim < self.width) & (xlim >= 0) & (ylim < self.height) & (ylim >= 0) & (tlim >= 0) & (tlim < self.stack_size)
                    interp_weights = p * (1 - np.abs(xlim - x)) * (1 - np.abs(ylim - y)) * (1 - np.abs(tlim - t_norm))
                    
                    index = self.height * self.width * tlim.astype(np.int64) + \
                            ylim.astype(np.int64) * self.width + \
                            xlim.astype(np.int64)

                    np.add.at(flatten_voxel_grid, index[mask], interp_weights[mask])
                    
        voxel_grid = np.reshape(
            flatten_voxel_grid, (self.stack_size, self.height, self.width)
        )
        
        if self.normalize:
            mask = np.nonzero(voxel_grid)
            if mask[0].size > 0:
                values = voxel_grid[mask]
                mean = values.mean()
                std = values.std()
                if std > 0:
                    voxel_grid[mask] = (values - mean) / std
                else:
                    voxel_grid[mask] = values - mean

        return voxel_grid
       
class TencodeEventStacking(EventStacking):
    NO_VALUE = 0.

    def __init__(self, **kwargs):
        _kwargs = kwargs.copy()
        _kwargs['stack_size'] = 3
        super().__init__(**_kwargs)
        self.gamma = kwargs.get('gamma', 1.0)

    def make_empty_stack(self):
        tencode_tensor = np.zeros([3, self.height, self.width], dtype=np.float32)
        return tencode_tensor   

    def make_stack(self, x, y, p, t):
        assert len(x) == len(y) == len(p) == len(t)

        t = t - t.min()
        # _quant_high = np.percentile(t, 90)
        # t = np.clip(t, 0, _quant_high)
        # time_interval = _quant_high
        time_interval = t.max() - t.min()
        t_s = (t / time_interval) if time_interval > 0 else t
        # polarity should be 0 / 1
        p = (p - p.min()) / (p.max() - p.min()) if p.max() - p.min() > 0 else p

        return self.stack_data(x, y, p, t_s)

    def stack_data(self, x, y, p, t_s):
        tencode_tensor = np.zeros((3,self.height, self.width), dtype=np.float32)
        
        index_red = (0 * self.width * self.height) + (y * self.width) + x
        index_green = (1 * self.width * self.height) + (y * self.width) + x
        index_blue = (2 * self.width * self.height) + (y * self.width) + x

        tencode_tensor.put(index_red, 255*p)
        tencode_tensor.put(index_green, 255*((1-t_s)**self.gamma))
        tencode_tensor.put(index_blue, 255*(1-p))

        if self.normalize:
            # normalization between 0 and 1
            tencode_tensor = tencode_tensor / 255

        return tencode_tensor#CxHxW      
