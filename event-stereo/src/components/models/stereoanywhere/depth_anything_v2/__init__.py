from .dpt import DepthAnythingV2
import torch

# This folder contains the Depth-Anything-V2 code from the official repository
# https://github.com/DepthAnything/Depth-Anything-V2
# The code has minor modifications to make it compatible with our framework.

def get_depth_anything_v2(checkpoint_path = None, encoder = None, map_location = 'cpu', freeze_encoder=False, freeze_decoder=False):
    assert encoder in [None, 'vits', 'vitb', 'vitl', 'vitg'], "Select a valid ViT encoder"

    if encoder is None:
        # Try to infer the encoder from the checkpoint name
        if checkpoint_path is not None and 'vits' in checkpoint_path:
            encoder = 'vits'
        elif checkpoint_path is not None and 'vitb' in checkpoint_path:
            encoder = 'vitb'
        elif checkpoint_path is not None and 'vitl' in checkpoint_path:
            encoder = 'vitl'
        elif checkpoint_path is not None and 'vitg' in checkpoint_path:
            encoder = 'vitg'
        else:
            # raise ValueError("Could not infer the ViT encoder from the checkpoint path")
            print("Could not infer the ViT encoder from the checkpoint path. Using 'vitl' as default.")
            encoder = 'vitl'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder], freeze_encoder=freeze_encoder, freeze_decoder=freeze_decoder)
    if checkpoint_path is None:
        return depth_anything
    
    # Load the checkpoint
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    depth_anything.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
    # depth_anything = depth_anything.eval()

    return depth_anything
