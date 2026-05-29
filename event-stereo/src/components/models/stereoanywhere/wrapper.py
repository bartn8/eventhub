import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from types import SimpleNamespace
import sys
import os
from collections import deque 

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from models.stereoanywhere.stereoanywhere import StereoAnywhere
    from models.stereoanywhere.depth_anything_v2 import get_depth_anything_v2
    from models.stereoanywhere.video_depth_anything import get_video_depth_anything
else:
    from .stereoanywhere import StereoAnywhere
    from .depth_anything_v2 import get_depth_anything_v2
    from .video_depth_anything import get_video_depth_anything
    from ..nerflosses import image_loss

# Custom wrapper for StereoAnywhere (to handle padding and mono model)
class StereoAnywhereWrapper(torch.nn.Module):
    def __init__(self, args):
        super(StereoAnywhereWrapper, self).__init__()

        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        self.args = args
        self.args.in_channels = 3 if not hasattr(args, "in_channels") else args.in_channels
        self.args.max_mono_width = args.max_mono_width if hasattr(args, "max_mono_width") else 518
        self.args.max_mono_height = args.max_mono_height if hasattr(args, "max_mono_height") else 518

        self.criterion = nn.L1Loss(reduction='none')
        self.number_of_init_disps = 2  # Number of initial disparities to be used in the loss function

        self.stereo_model = StereoAnywhere(args)#, features_output_dim=128)
        # self.mono_model = get_depth_anything_v2(checkpoint_path=self.args.loadmonomodel, encoder=args.mono_vit_encoder)
        self.mono_model = get_video_depth_anything(**vars(args))
        
        self.temporal_queue_left = deque(maxlen=1)  # For storing previous frames if needed
        self.temporal_queue_right = deque(maxlen=1)  # For storing previous frames if needed
        
        # if self.args.freeze_for_finetuning:
        #     self.stereo_model.freeze_for_finetuning()        
            
    # def convert_from_voxels(self, voxels):
    #     n,h,w = voxels.shape
    #     g_map = [255.0 * (1-i) for i in torch.linspace(0,1,n)]
        
    #     tencode = torch.full((3,h,w), 255.0, dtype=torch.float)

    #     for i in range(n):
    #         tencode[0,voxels[i]>0] = 255.0
    #         tencode[1,voxels[i]>0] = g_map[i]
    #         tencode[2,voxels[i]>0] = 0.0
    #         tencode[0,voxels[i]<0] = 0.0
    #         tencode[1,voxels[i]<0] = g_map[i]
    #         tencode[2,voxels[i]<0] = 255.0

    #     # If there are no events, we set the frame to white
    #     tencode[:, tencode.sum(dim=0) == 0] = 255.0

    #     tencode = tencode / 255.0

    #     return tencode              
            
    def convert_from_voxels(self, voxels):
        b, n, h, w = voxels.shape
        
        # Normalize voxels from -1 to 1
        voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())
        voxels = voxels * 2 - 1
        
        g_map = torch.linspace(255.0, 0.0, n, device=voxels.device)

        # Prepare masks
        pos_mask = voxels > 0 # shape: (b, n, h, w)
        neg_mask = voxels < 0 # shape: (b, n, h, w)

        # Initialize tencode
        tencode = torch.full((b, 3, h, w), 255.0, dtype=torch.float, device=voxels.device)

        # Broadcast g_map for vectorized assignment
        g_map_expanded = g_map.view(1, n, 1, 1)  # shape: (1, n, 1, 1)

        # Assign values for positive events
        pos_any = pos_mask.any(dim=1)  # shape: (b, h, w)
        tencode[:, 0] = torch.where(pos_any, torch.full_like(tencode[:, 0], 255.0), tencode[:, 0])
        tencode[:, 1] = torch.where(
            pos_any,
            (g_map_expanded * pos_mask.float()).sum(dim=1),
            tencode[:, 1]
        )
        tencode[:, 2] = torch.where(pos_any, torch.zeros_like(tencode[:, 2]), tencode[:, 2])

        # Assign values for negative events
        neg_any = neg_mask.any(dim=1)  # shape: (b, h, w)
        tencode[:, 0] = torch.where(neg_any, torch.zeros_like(tencode[:, 0]), tencode[:, 0])
        tencode[:, 1] = torch.where(
            neg_any,
            (g_map_expanded * neg_mask.float()).sum(dim=1),
            tencode[:, 1]
        )
        tencode[:, 2] = torch.where(neg_any, torch.full_like(tencode[:, 2], 255.0), tencode[:, 2])

        # Set frame to white if there are no events
        no_event_mask = (tencode.sum(dim=1) == 0)  # shape: (b, h, w)
        tencode.permute(0, 2, 3, 1)[no_event_mask, :] = 255.0  # set all channels to 255

        tencode = tencode / 255.0

        return tencode
            
    def freeze_bn(self):
        self.stereo_model.freeze_bn()

    def forward(self, left_stack, right_stack, test_mode=False, iters=12, **kwargs):

        C = left_stack.shape[1]
        
        if C == 3:
            # If the input is already in RGB format, we can skip the conversion
            left_image = left_stack
            right_image = right_stack
        elif C == 1:
            # If the input is a single channel, we can convert it to RGB by repeating the channel
            left_image = left_stack.repeat(1, 3, 1, 1)
            right_image = right_stack.repeat(1, 3, 1, 1)
            left_stack = left_image
            right_stack = right_image
        elif C > 3:
            left_image = self.convert_from_voxels(left_stack[:, -self.args.in_channels//4:])
            right_image = self.convert_from_voxels(right_stack[:, -self.args.in_channels//4:])
        else:
            raise ValueError(f"Unsupported number of channels: {C}. Expected 1, 3, or more than 3 channels.")
        
        left_image = (left_image - left_image.min()) / (left_image.max() - left_image.min())
        right_image = (right_image - right_image.min()) / (right_image.max() - right_image.min())

        # if self.input_conv is not None:
        #     left_image = self.input_conv(left_bins)
        #     right_image = self.input_conv(right_stack)
        # else:
        #     left_image = left_bins
        #     right_image = right_stack

        # Assuming the model takes a batch of images as input
        if self.mono_model is not None:
            #mono_depths = self.mono_model.infer_image(torch.cat([left_image, right_image], 0), input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            
            #DAV2
            # mono_depth_left = self.mono_model.infer_image(left_image, input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            # mono_depth_right = self.mono_model.infer_image(right_image, input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            
            #VDA
            if test_mode:
                self.temporal_queue_left.append(left_image)
                self.temporal_queue_right.append(right_image)
                
                mono_depths_left, mono_depths_right = self.mono_model.infer_stereo_list(
                    list(self.temporal_queue_left), list(self.temporal_queue_right), input_size_max_width=self.args.max_mono_width, input_size_max_height=self.args.max_mono_height, no_grad=test_mode
                )

                mono_depth_left = mono_depths_left[-1].unsqueeze(0)
                mono_depth_right = mono_depths_right[-1].unsqueeze(0)
            else:
                mono_depth_left, mono_depth_right = self.mono_model.infer_stereo_image(
                    left_image, right_image, input_size_max_width=self.args.max_mono_width, input_size_max_height=self.args.max_mono_height, no_grad=test_mode
                )
            
            mono_depths = torch.cat([mono_depth_left, mono_depth_right], 0)
            mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
            mono_left = mono_depths[:left_image.shape[0]]
            mono_right = mono_depths[left_image.shape[0]:]
        else:
            # mono_left = torch.zeros_like(left_image[:, 0:1]) if mono_left is None else mono_left
            # mono_right = torch.zeros_like(right_image[:, 0:1]) if mono_right is None else mono_right
            raise ValueError("Mono model is not initialized. Please provide a valid mono model.")

        # Pad 32
        ht, wt = left_image.shape[-2], left_image.shape[-1]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

        left_image = F.pad(left_image, _pad, mode='replicate')
        right_image = F.pad(right_image, _pad, mode='replicate')
        mono_left = F.pad(mono_left, _pad, mode='replicate')    
        mono_right = F.pad(mono_right, _pad, mode='replicate')

        _results = self.stereo_model(left_stack, right_stack, mono_left, mono_right, test_mode=test_mode, iters=iters)

        if test_mode:
            pred_disps , *_ = _results    
        else:
            pred_disps, _, init_disps, *_ = _results

            init_disps = [init_disp for init_disp in init_disps if init_disp is not None]

            # Disable init_disps
            # init_disps = []

            # assert len(init_disps) == 2, f"Expected 2 initial disparities, got {len(init_disps)}. Check StereoAnywhere output."
            self.number_of_init_disps = len(init_disps)
            pred_disps = init_disps + pred_disps # Concatenate initial disparities with predictions
            
        if not isinstance(pred_disps, list):
            pred_disps = [pred_disps]

        for i in range(len(pred_disps)):
            pred_disps[i] = - pred_disps[i].squeeze(1) # Assuming the model outputs negative disparities
        
            hd, wd = pred_disps[i].shape[-2:]
            c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
            pred_disps[i] = pred_disps[i][:, c[0]:c[1], c[2]:c[3]]

        return pred_disps
    
    def print_once(self, message, stderr=False):
        if not hasattr(self, 'printed_messages'):
            self.printed_messages = set()
        if message not in self.printed_messages:
            if stderr:
                print(message, file=sys.stderr)
            else:
                print(message)
            self.printed_messages.add(message)
            
    def _cal_loss(self, pred_disparity_pyramid, gt_disparity, loss_gamma=0.9, alpha_disp_loss=1.0, alpha_photometric=0.1, confidence=None, rgb_left=None, rgb_right=None, rgb_center=None):

        #Extract init_disps
        init_disps = pred_disparity_pyramid[:self.number_of_init_disps]
        pred_disparity_pyramid = pred_disparity_pyramid[self.number_of_init_disps:]

        use_trinocular_loss = rgb_left is not None and rgb_right is not None and rgb_center is not None
        use_binocular_loss = not use_trinocular_loss and rgb_center is not None and rgb_right is not None
        # self.print_once(f"Use trinocular loss: {use_trinocular_loss}", stderr=True)
        # self.print_once(f"Use binocular loss: {use_binocular_loss}", stderr=True)
        confidence = confidence if confidence is not None else torch.where(gt_disparity > 0, torch.ones_like(gt_disparity), torch.zeros_like(gt_disparity))

        if alpha_disp_loss is None:
            alpha_disp_loss = 1.0
        if alpha_photometric is None:
            alpha_photometric = 0.1

        n_predictions = len(pred_disparity_pyramid)
        assert n_predictions >= 1
        flow_loss = 0.0

        # exclude extremly large displacements
        valid = ((gt_disparity > 0))#.unsqueeze(1)
        assert valid.shape == gt_disparity.shape, [valid.shape, gt_disparity.shape]
        assert not torch.isinf(gt_disparity[valid.bool()]).any()

        for i in range(n_predictions):
            # instead of assert just skip the invalid predictions
            # assert not torch.isnan(pred_disparity_pyramid[i]).any() and not torch.isinf(pred_disparity_pyramid[i]).any()
            if torch.isnan(pred_disparity_pyramid[i]).any() or torch.isinf(pred_disparity_pyramid[i]).any():
                print(f"Warning: Invalid disparity prediction at index {i}. Skipping this prediction.")
                continue

            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

            disp_diff = self.criterion(pred_disparity_pyramid[i].unsqueeze(1), gt_disparity)
            disp_loss = (disp_diff * confidence * valid.float())

            if use_trinocular_loss:
                photometric_loss = image_loss(
                    pred_disparity_pyramid[i].unsqueeze(1), rgb_left, rgb_center, rgb_right, 1 - confidence, trinocular=True
                )
            elif use_binocular_loss:
                photometric_loss = image_loss(
                    pred_disparity_pyramid[i].unsqueeze(1), None, rgb_center, rgb_right, 1 - confidence, trinocular=False
                )
            else:
                photometric_loss = torch.zeros_like(disp_loss)
            
            flow_loss += i_weight * (alpha_disp_loss * disp_loss + alpha_photometric * photometric_loss)

        for i in range(len(init_disps)):
            if torch.isnan(init_disps[i]).any() or torch.isinf(init_disps[i]).any():
                print(f"Warning: Invalid initial disparity at index {i}. Skipping this initial disparity.")
                continue

            init_disp_diff = self.criterion(init_disps[i].unsqueeze(1), gt_disparity)
            init_disp_loss = (init_disp_diff * confidence * valid.float())
            
            if use_trinocular_loss:
                init_photometric_loss = image_loss(
                    init_disps[i].unsqueeze(1), rgb_left, rgb_center, rgb_right, 1 - confidence, trinocular=True
                )
            elif use_binocular_loss:
                init_photometric_loss = image_loss(
                    init_disps[i].unsqueeze(1), None, rgb_center, rgb_right, 1 - confidence, trinocular=False
                )
            else:
                init_photometric_loss = torch.zeros_like(init_disp_loss)

            flow_loss += alpha_disp_loss * init_disp_loss + alpha_photometric * init_photometric_loss

        return flow_loss

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='StereoAnywhere Wrapper Checkpoint maker')
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_additional_hourglass', type=int, default=0)
    parser.add_argument('--volume_channels', type=int, default=8)
    parser.add_argument('--vol_downsample', type=float, default=0)
    parser.add_argument('--vol_n_masks', type=int, default=8)
    parser.add_argument('--use_truncate_vol', action='store_true')
    parser.add_argument('--mirror_conf_th', type=float, default=0.98)
    parser.add_argument('--mirror_attenuation', type=float, default=0.9)
    parser.add_argument('--use_aggregate_stereo_vol', type=bool, default=False)
    parser.add_argument('--use_aggregate_mono_vol', type=bool, default=True)
    parser.add_argument('--normal_gain', type=int, default=10)
    parser.add_argument('--lrc_th', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=32, help='Number of iterations for recurrent networks')

    parser.add_argument('--loadstereomodel', required=True, help='load stereo model')
    parser.add_argument('--loadmonomodel', required=True, help='load model')
    parser.add_argument('--vit_encoder', type=str, default='vits', help='Encoder type for the mono model')

    args = parser.parse_args()

    """
    Used command:
    python models/stereoanywhere/wrapper.py  --loadstereomodel /home/luca/Scrivania/projects/stereoanywhere/weights/checkpoint_3.tar --loadmonomodel /home/luca/Scrivania/libs/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth --iters 32 --vol_n_masks 8 --volume_channels 8 --n_additional_hourglass 0 --use_aggregate_mono_vol True --vol_downsample 0
    """


    wrapper = StereoAnywhereWrapper(args)
    wrapper.stereo_model = torch.nn.DataParallel(wrapper.stereo_model)

    _stereo_dict = torch.load(args.loadstereomodel, map_location='cpu')['state_dict']

    # Remove 'module.' prefix if present
    # _stereo_dict = {k.replace('module.', ''): v for k, v in _stereo_dict.items()}

    wrapper.stereo_model.load_state_dict(_stereo_dict)
    wrapper.mono_model = get_depth_anything_v2(checkpoint_path=args.loadmonomodel, encoder=args.vit_encoder)


    wrapper.stereo_model = wrapper.stereo_model.module

    print("StereoAnywhere Wrapper initialized with models loaded.")
    
    # Save the wrapper model
    torch.save({'state_dict': wrapper.state_dict()}, 'tmp/stereoanywhere_wrapper.pth')
    print("StereoAnywhere Wrapper model saved as 'stereoanywhere_wrapper.pth'.")
