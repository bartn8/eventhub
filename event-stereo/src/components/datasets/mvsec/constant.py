MVSEC_HEIGHT = 260
MVSEC_WIDTH = 346

DATA_SPLIT = {
    'train': ['indoor_flying/indoor_flying3_data.hdf5', 'outdoor_day/outdoor_day2_data.hdf5',],
    'proxy': ['indoor_flying/indoor_flying3_data.hdf5', 'motorcycle/motorcycle1_data.hdf5',],
    
    'val_indoor': ['indoor_flying/indoor_flying1_data.hdf5', 'indoor_flying/indoor_flying2_data.hdf5', 'indoor_flying/indoor_flying4_data.hdf5',],
    'val_outdoor_day': ['outdoor_day/outdoor_day1_data.hdf5',],
    'val_outdoor_night': ['outdoor_night/outdoor_night1_data.hdf5'],

    'test': [],
    'none': [],
}

DATA_SPLIT['validation'] = DATA_SPLIT['val_indoor'] + DATA_SPLIT['val_outdoor_day'] + DATA_SPLIT['val_outdoor_night']

# For test we use same frames as
# "Realtime Time Synchronized Event-based Stereo"
# by Alex Zhu et al. for consistency of test results.
FRAMES_FILTER = {
    'indoor_flying/indoor_flying1_data.hdf5': [140, 1201],
    'indoor_flying/indoor_flying2_data.hdf5': [120, 1421],
    'indoor_flying/indoor_flying3_data.hdf5': [73, 1616],
    'indoor_flying/indoor_flying4_data.hdf5': [190, 290],
    'outdoor_day/outdoor_day2_data.hdf5': [50, 12000]
}

# Custom filters for other sequences
FRAMES_FILTER.update({
    'outdoor_day/outdoor_day1_data.hdf5': [200, 4900],
    'outdoor_night/outdoor_night1_data.hdf5': [200, 4900],
    'motorcycle/motorcycle1_data.hdf5': [200, 20000],
})