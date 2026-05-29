import os
import argparse

import torch
import numpy as np
import random

from manager import DLManager
from utils.config import get_cfg

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='/root/code/configs/config.yaml')
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--save_root', type=str, default='/root/code/save')
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')

#TODO: Add those arguments to the training pipeline
parser.add_argument('--crop_height', type=int, default=0, help="""Crop the image. Must be divisible by 48""")
parser.add_argument('--crop_width', type=int, default=0, help="""Crop the image. Must be divisible by 48""")
parser.add_argument('--resize_height', type=int, default=0, help="""Resize image after crop it. Must be divisible by 48""")
parser.add_argument('--resize_width', type=int, default=0, help="""Resize image after crop it. Must be divisible by 48""")

parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--save_term', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--validate', action='store_true', help="""Run validation after each epoch""")
parser.add_argument('--data_root_validation', type=str, default=None)

parser.add_argument('--seq_size', type=int, default=0)
parser.add_argument('--seq_size_val', type=int, default=0)

args = parser.parse_args()

assert os.path.isfile(args.config_path)
assert args.data_root is None or os.path.isdir(args.data_root)
assert args.data_root_validation is None or os.path.isdir(args.data_root_validation) or not args.validate, "Validation data root must be a directory if validation is enabled"
# assert not (args.pretrain_path is not None and args.resume), "Cannot load pretrained if resuming training"
if args.resume:
    args.pretrain_path = None  # If resuming, do not load pretrained weights

if not os.path.isdir(args.save_root):
    os.makedirs(args.save_root, exist_ok=True)

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Set a specific seed, for example, 42
set_global_seed(args.seed)


# Set Config
cfg = get_cfg(args.config_path)

exp_manager = DLManager(args, cfg, train_mode=True)

if args.resume:
    exp_manager.resume(args.save_root+'/weights/final.pth')

if args.pretrain_path is not None:
    exp_manager.load_pretrain(args.pretrain_path)

exp_manager.train()
#exp_manager.test()
