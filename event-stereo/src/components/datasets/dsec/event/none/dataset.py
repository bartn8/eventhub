import os
import numpy as np
import torch.utils.data
import zlib
import io

from .slice import EventSlicerMock as EventSlicer
from . import constant
from ....events_utils import stack

class EventDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = None

    def __init__(self, root, num_of_event, stack_method, **kwargs):
        self.root = root
        self.seq_name = root.split("/")[-2]
        self.num_of_event = num_of_event
        self.stack_method = stack_method
        
        self.event_slicer = {}
        for location in self._LOCATION:
            event_path = os.path.join(root, location, 'events.h5')
            rectify_map_path = os.path.join(root, location, 'rectify_map.h5')
            self.event_slicer[location] = EventSlicer(event_path, rectify_map_path, num_of_event)

        self.stack_function = {}
        self.dropout_rnd_state = round(np.random.rand() * int(2**32 - 1))
        self.noise_rnd_state = round(np.random.rand() * int(2**32 - 1))
        for location in self._LOCATION:
            self.stack_function[location] = getattr(stack, stack_method)(height=constant.EVENT_HEIGHT,
                                                            width=constant.EVENT_WIDTH, dropout_rnd_state=self.dropout_rnd_state, noise_rnd_state=self.noise_rnd_state, **kwargs)
        self.NO_VALUE = self.stack_function[location].NO_VALUE
        self.stack_function_collate_fn = self.stack_function[location].collate_fn

    def __len__(self):
        # Not applicable for the event part of the dataset
        return 0

    def __getitem__(self, mydict):
        timestamp = mydict['timestamp']

        event_data = self._pre_load_event_data(timestamp=timestamp)
        event_data = self._post_load_event_data(event_data)

        return event_data

    def _pre_load_event_data(self, timestamp):
        event_data = {}
        # minimum_time, maximum_time = -float('inf'), float('inf')

        # for location in self._LOCATION:
        #     event_data[location] = self.event_slicer[location][timestamp]
        #     minimum_time = max(minimum_time, event_data[location]['t'].min())
        #     maximum_time = min(maximum_time, event_data[location]['t'].max())

        # for location in self._LOCATION:
        #     mask = np.logical_and(minimum_time <= event_data[location]['t'], event_data[location]['t'] <= maximum_time)
        #     for data_type in ['x', 'y', 't', 'p']:
        #         event_data[location][data_type] = event_data[location][data_type][mask]

        for location in self._LOCATION:
            event_data[location] = self.stack_function[location].make_empty_stack()

        return event_data

    def _post_load_event_data(self, event_data):
        for location in self._LOCATION:
            event_data[location] = self.stack_function[location].post_stack(event_data[location])#HxWx{1,2}xCx1 old. now HxWxC
            
        return event_data

    def collate_fn(self, batch):
        batch = self.stack_function_collate_fn(batch)
        return batch
