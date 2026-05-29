import os
import argparse

import torch
import numpy as np
import random

from manager import DLManager

from utils.config import get_cfg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_config', type=str, required=True, help="""Path to the test configuration file""")
parser.add_argument('--data_root', type=str, required=False, default=None)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_root', type=str, required=True)
parser.add_argument('--verbose', action='store_true', default=False)

parser.add_argument('--crop_height', type=int, default=0, help="""Crop the image. Must be divisible by 48""")
parser.add_argument('--crop_width', type=int, default=0, help="""Crop the image. Must be divisible by 48""")
parser.add_argument('--resize_height', type=int, default=0, help="""Resize image after crop it. Must be divisible by 48""")
parser.add_argument('--resize_width', type=int, default=0, help="""Resize image after crop it. Must be divisible by 48""")

parser.add_argument('--render', action='store_true')
parser.add_argument('--save_predictions', action='store_true')

parser.add_argument('--num_events', type=int, default=0)
parser.add_argument('--use_preproc_image', action='store_true')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--csv_file', type=str, default=None, help="""Write results and parameters into a CSV file""")
#parser.add_argument('--csv_header', action='store_true', help="""Add CSV header""")
parser.add_argument('--split', type=str, default='validation', help="""Choose data split""")

parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--seq_size', type=int, default=0)
parser.add_argument('--sampling_ratio', type=int, default=1)

args = parser.parse_args()

assert os.path.isdir(args.data_root)

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Set a specific seed, for example, 42
set_global_seed(args.seed)

test_config = get_cfg(args.test_config)

exp_manager = DLManager(args, train_mode=False)
exp_manager.load_inference(test_config)

exp_manager.test()
