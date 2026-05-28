DSEC_HEIGHT = 480
DSEC_WIDTH = 640

DATA_SPLIT = {
    # 'train': ['train/interlaken_00_c', 'train/interlaken_00_d', 'train/interlaken_00_e', 'train/zurich_city_00_a',
    #           'train/zurich_city_00_b', 'train/zurich_city_01_a', 'train/zurich_city_01_b', 'train/zurich_city_01_c',
    #           'train/zurich_city_01_d', 'train/zurich_city_01_e', 'train/zurich_city_01_f', 'train/zurich_city_02_a',
    #           'train/zurich_city_02_b', 'train/zurich_city_02_c', 'train/zurich_city_02_d', 'train/zurich_city_02_e',
    #           'train/zurich_city_03_a', 'train/zurich_city_04_a', 'train/zurich_city_04_b', 'train/zurich_city_04_c',
    #           'train/zurich_city_04_d', 'train/zurich_city_04_e', 'train/zurich_city_04_f', 'train/zurich_city_09_a',
    #           'train/zurich_city_09_b', 'train/zurich_city_09_c', 'train/zurich_city_09_e', 'train/zurich_city_10_a',
    #           'train/zurich_city_11_a', 'train/zurich_city_11_b', 'train/zurich_city_11_c'],
    
    'train_outdoor_day': ['train/interlaken_00_c', 'train/interlaken_00_d', 'train/interlaken_00_e', 'train/zurich_city_00_a',
                          'train/zurich_city_00_b', 'train/zurich_city_01_a', 'train/zurich_city_01_b', 'train/zurich_city_01_c',
                          'train/zurich_city_01_d', 'train/zurich_city_01_e', 'train/zurich_city_01_f', 'train/zurich_city_02_a',
                          'train/zurich_city_02_b', 'train/zurich_city_02_c', 'train/zurich_city_02_d', 'train/zurich_city_02_e',
                          'train/zurich_city_04_a', 'train/zurich_city_04_b', 'train/zurich_city_04_c', 'train/zurich_city_04_d',
                          'train/zurich_city_04_e', 'train/zurich_city_04_f', 'train/zurich_city_11_a', 'train/zurich_city_11_b',
                          'train/zurich_city_11_c'],
    
    'train_outdoor_night': ['train/zurich_city_03_a', 'train/zurich_city_09_a',
                            'train/zurich_city_09_b', 'train/zurich_city_09_c', 'train/zurich_city_09_e', 'train/zurich_city_10_a',],
    
    'val_outdoor_day': ['train/interlaken_00_f', 'train/interlaken_00_g', 'train/thun_00_a', 'train/zurich_city_05_a',
                        'train/zurich_city_05_b', 'train/zurich_city_06_a', 'train/zurich_city_07_a',
                        'train/zurich_city_08_a',],
    'val_outdoor_night': ['train/zurich_city_09_d', 'train/zurich_city_10_b'],
            
    'test': ['test/interlaken_00_a', 'test/interlaken_00_b', 'test/interlaken_01_a', 'test/thun_01_a',
             'test/thun_01_b', 'test/zurich_city_12_a', 'test/zurich_city_13_a', 'test/zurich_city_13_b',
             'test/zurich_city_14_a', 'test/zurich_city_14_b', 'test/zurich_city_14_c', 'test/zurich_city_15_a'],
    
    'none': [],
    'render': ['train/interlaken_00_f', 'train/thun_00_a', 'train/zurich_city_06_a', 'train/zurich_city_09_d', 'train/zurich_city_10_b' ],
}

DATA_SPLIT['train'] = DATA_SPLIT['train_outdoor_day'] + DATA_SPLIT['train_outdoor_night']
_BANNED_PROXY = ['zurich_city_10_a', 'zurich_city_10_b', 'zurich_city_03_a', 'zurich_city_09_a', 'zurich_city_09_b', 'zurich_city_09_c', 'zurich_city_09_d', 'zurich_city_09_e']
DATA_SPLIT['proxy'] = [seq for seq in DATA_SPLIT['train'] if seq.split('/')[-1] not in _BANNED_PROXY]
DATA_SPLIT['validation'] = DATA_SPLIT['val_outdoor_day'] + DATA_SPLIT['val_outdoor_night']
DATA_SPLIT['trainval'] = DATA_SPLIT['train'] + DATA_SPLIT['validation']

# DATA_SPLIT['train'] = DATA_SPLIT['train'][:1]