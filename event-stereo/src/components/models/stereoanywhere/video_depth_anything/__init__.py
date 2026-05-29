from .video_depth import VideoDepthAnything
import torch

# This folder contains the Depth-Anything-V2 code from the official repository
# https://github.com/DepthAnything/Depth-Anything-V2
# The code has minor modifications to make it compatible with our framework.

def get_video_depth_anything(loadmonomodel = None, mono_vit_encoder = None, map_location = 'cpu', return_features_dim=False, use_lora=False, lora_rank=3, **kwargs):
    checkpoint_path = loadmonomodel
    encoder = mono_vit_encoder
    assert encoder in [None, 'vits', 'vitl'], "Select a valid ViT encoder"

    if encoder is None:
        # Try to infer the encoder from the checkpoint name
        if checkpoint_path is not None and 'vits' in checkpoint_path:
            encoder = 'vits'
        elif checkpoint_path is not None and 'vitl' in checkpoint_path:
            encoder = 'vitl'
        else:
            # raise ValueError("Could not infer the ViT encoder from the checkpoint path")
            print("Could not infer the ViT encoder from the checkpoint path. Using 'vits' as default.")
            encoder = 'vits'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    config = model_configs[encoder]
    config.update({'use_lora': use_lora, 'lora_rank': lora_rank})
    
    features_dim = model_configs[encoder]['features']

    video_depth_anything = VideoDepthAnything(**config)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        video_depth_anything.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
        # video_depth_anything = video_depth_anything.eval()

    if return_features_dim:
        return video_depth_anything, features_dim

    return video_depth_anything
