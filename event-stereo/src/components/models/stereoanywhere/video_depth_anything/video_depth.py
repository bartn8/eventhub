# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import math

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize
from .lora import LoRA


# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class DummyNoGrad:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass



class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        use_lora: bool = False, lora_rank: int = 3
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        
        self.freeze_model()
                
        # After freezing the encoder, add LoRA layers if specified
        if self.use_lora:
            self._add_lora_layers()
                
    def freeze_model(self):
        model = self.pretrained.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        self.pretrained = model
        
        model = self.head.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        self.head = model                
                
    def _add_lora_layers(self):
        """Add LoRA layers to the pretrained model's attention blocks"""
        # Freeze original parameters
        for param in self.pretrained.parameters():
            param.requires_grad = False
            
        # Add LoRA layers
        self.lora_layers = list(range(len(self.pretrained.blocks)))
        self.w_a = nn.ModuleList()
        self.w_b = nn.ModuleList()
        
        for i, block in enumerate(self.pretrained.blocks):
            if i not in self.lora_layers:
                continue
                
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features
            
            # Create LoRA layers for Q and V
            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, self.lora_rank)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, self.lora_rank)
            
            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])
            
            # Replace the original qkv layer with LoRA version
            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
            
        self._reset_lora_parameters()
        
    def _create_lora_layer(self, dim: int, r: int):
        """Create a pair of LoRA layers (A and B matrices)"""
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b
        
    def _reset_lora_parameters(self) -> None:
        """Initialize LoRA parameters"""
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)
            
    def save_lora_parameters(self, filename: str) -> None:
        """Save the LoRA weights to a .pt file
        
        Args:
            filename (str): Filename of the weights
        """
        if not self.use_lora:
            print("Warning: LoRA is not enabled")
            return
            
        w_a_dict = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
        w_b_dict = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_b))}
        
        torch.save({**w_a_dict, **w_b_dict}, filename)
        
    def load_lora_parameters(self, filename: str) -> None:
        """Load the LoRA weights from a file
        
        Args:
            filename (str): File name of the weights
        """
        if not self.use_lora:
            print("Warning: LoRA is not enabled")
            return
            
        state_dict = torch.load(filename)
        
        for i, w_a_linear in enumerate(self.w_a):
            saved_key = f"w_a_{i:03d}"
            if saved_key in state_dict:
                w_a_linear.weight = nn.Parameter(state_dict[saved_key])
                
        for i, w_b_linear in enumerate(self.w_b):
            saved_key = f"w_b_{i:03d}"
            if saved_key in state_dict:
                w_b_linear.weight = nn.Parameter(state_dict[saved_key])
                
    def freeze_pretrained(self) -> None:
        """Freeze the pretrained backbone parameters"""
        for param in self.pretrained.parameters():
            param.requires_grad = False
            
    def unfreeze_pretrained(self) -> None:
        """Unfreeze the pretrained backbone parameters (except LoRA layers)"""
        for param in self.pretrained.parameters():
            param.requires_grad = True
            
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimization"""
        if not self.use_lora:
            return []
        
        params = []
        for w_a in self.w_a:
            params.extend(w_a.parameters())
        for w_b in self.w_b:
            params.extend(w_b.parameters())
        return params
                    
    def forward(self, x, no_grad=False):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        
        _no_grad = torch.no_grad() if no_grad or not self.use_lora else DummyNoGrad()
        # _no_grad = DummyNoGrad()
        with _no_grad: 
            features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)

        _no_grad = torch.no_grad() if no_grad or not self.use_lora else DummyNoGrad()
        # _no_grad = DummyNoGrad()
        with _no_grad:
            depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]

    def pad_to_multiple(self, img, multiple):
        h, w = img.shape[-2], img.shape[-1]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        img = F.pad(img, pad, mode='constant', value=0)
        return img, pad_h, pad_w

    def infer_image(self, raw_image, input_size_max_width=518, input_size_max_height=518, output_intermediate=False, no_grad=False):
        output_intermediate=False
        
        image, (h, w), (final_h, final_w), padded = self.image2tensor(
            raw_image, input_size_max_width, input_size_max_height
        )

        depth = self.forward(image.unsqueeze(1), no_grad=no_grad) # B T H W
        if padded:
            depth = depth[..., :h, :w]
        else:
            depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)
            
        return depth
    
    def infer_stereo_image(self, left_image, right_image, input_size_max_width=518, input_size_max_height=518, output_intermediate=False, no_grad=False):
        output_intermediate=False
        
        left_image, (h, w), (final_h, final_w), padded = self.image2tensor(
            left_image, input_size_max_width, input_size_max_height
        )
        right_image, _, _, _ = self.image2tensor(
            right_image, input_size_max_width, input_size_max_height
        )

        stereo_input = torch.cat([left_image.unsqueeze(1), right_image.unsqueeze(1)], dim=1) # B T C H W
        depth = self.forward(stereo_input, no_grad=no_grad) # B T H W

        if padded:
            depth = depth[..., :h, :w]
        else:
            depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)

        depth_left = depth[:, [0], ...] # B 1 H W
        depth_right = depth[:, [1], ...] # B 1 H W

        return depth_left, depth_right
    
    def infer_stereo_list(self, left_list, right_list, input_size_max_width=518, input_size_max_height=518, output_intermediate=False, no_grad=False):
        output_intermediate=False

        left_list_padded = []
        right_list_padded = []

        for left_image, right_image in zip(left_list, right_list):
            left_image, (h, w), (final_h, final_w), padded = self.image2tensor(
                left_image, input_size_max_width, input_size_max_height
            )
            left_list_padded.append(left_image)

            right_image, _, _, _ = self.image2tensor(
                right_image, input_size_max_width, input_size_max_height
            )
            right_list_padded.append(right_image)

        left_tensor = torch.cat([left_image.unsqueeze(1) for left_image in left_list_padded], dim=1)
        right_tensor = torch.cat([right_image.unsqueeze(1) for right_image in right_list_padded], dim=1)

        stereo_input = torch.cat([left_tensor, right_tensor], dim=1) # B T C H W
        depth = self.forward(stereo_input, no_grad=no_grad) # B T H W

        if padded:
            depth = depth[..., :h, :w]
        else:
            depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)

        T = depth.shape[1]

        depths_left = depth[:, :T//2, ...] # B T//2 H W
        depths_right = depth[:, T//2:, ...] # B T//2 H W

        return depths_left, depths_right

    def image2tensor(self, raw_image, input_size_max_width=518, input_size_max_height=518):
        h, w = raw_image.shape[-2], raw_image.shape[-1]
        padded = False

        if h <= input_size_max_height and w <= input_size_max_width:
            # Pad to nearest multiple of 14
            image, pad_h, pad_w = self.pad_to_multiple(raw_image, 14)
            final_h, final_w = h + pad_h, w + pad_w
            padded = True
        else:
            if h > w:
                _tmp = input_size_max_height
                input_size_max_height = input_size_max_width
                input_size_max_width = _tmp

            resize = Resize(
                width=input_size_max_width,
                height=input_size_max_height,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            )

            final_w, final_h = resize.get_size(w, h)
            final_w, final_h = int(final_w), int(final_h)
            # image = F.interpolate(raw_image, (final_h, final_w), mode='bicubic', align_corners=False)
            image = F.interpolate(raw_image, (final_h, final_w), mode='nearest')

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            image[:, i].sub_(m).div_(s)

        return image, (h, w), (final_h, final_w), padded
