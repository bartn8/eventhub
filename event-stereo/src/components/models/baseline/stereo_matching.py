import torch
import torch.nn as nn
import torch.nn.functional as F

from .refinement import StereoDRNetRefinement

from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid
from .SparseConvNet import *

from ..nerflosses import image_loss
import sys

def gradient(data):
    D_dy = data[:, 1:, :] - data[:, :-1, :]
    D_dx = data[:, :, 1:] - data[:, :, :-1]
    return D_dx, D_dy

def smooth_loss(disp):
    D_dx, D_dy = gradient(disp)
    return D_dx.abs().mean() + D_dy.abs().mean()

class StereoMatchingNetwork(nn.Module):
    def __init__(self, max_disp,
                 in_channels=3,
                 num_downsample=2,
                 no_mdconv=False,
                 feature_similarity='correlation',
                 num_scales=3,
                 num_fusions=6,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 no_intermediate_supervision=False,
                 num_stage_blocks=1,
                 num_deform_blocks=3,
                 refine_channels=None,
                 **kwargs):
        super(StereoMatchingNetwork, self).__init__()

        refine_channels = in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales

        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)

        self.max_disp = max_disp
        max_disp = max_disp // 3

        # Cost volume construction
        self.cost_volume_constructor = CostVolumePyramid(max_disp, feature_similarity=feature_similarity)

        # Cost aggregation
        self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                               num_scales=num_scales,
                                               num_fusions=num_fusions,
                                               num_stage_blocks=num_stage_blocks,
                                               num_deform_blocks=num_deform_blocks,
                                               no_mdconv=no_mdconv,
                                               mdconv_dilation=mdconv_dilation,
                                               deformable_groups=deformable_groups,
                                               intermediate_supervision=not no_intermediate_supervision)

        # Disparity estimation
        self.disparity_estimation = DisparityEstimationPyramid(max_disp)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(num_downsample):
            refine_module_list.append(StereoDRNetRefinement(img_channels=refine_channels))

        self.refinement = refine_module_list

        self.criterion = nn.L1Loss(reduction='none')

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                curr_right_img = F.interpolate(right_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def forward(self, left_stack, right_stack, left_img=None, right_img=None, **kwargs):
        if left_img is None or right_img is None:
            left_img = left_stack
            right_img = right_stack

        # Pad images to ensure they are divisible by 12
        # B, T, C, H, W = left_img.shape
        B, C, H, W = left_img.shape
        pad_h = (12 - H % 12) % 12
        pad_w = (12 - W % 12) % 12
        
        left_img = F.pad(left_img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        right_img = F.pad(right_img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        left_stack = F.pad(left_stack, (0, pad_w, 0, pad_h), mode='constant', value=0)
        right_stack = F.pad(right_stack, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        left_feature = self.feature_extractor(left_stack)
        right_feature = self.feature_extractor(right_stack)

        # Rollback
        # left_feature, right_feature = None, None
        # for t in range(T):
        #     # Pad ogni frame temporaneamente per l'estrazione delle feature
        #     left_padded = F.pad(left_img[:, t], (0, pad_w, 0, pad_h), mode='constant', value=0)
        #     right_padded = F.pad(right_img[:, t], (0, pad_w, 0, pad_h), mode='constant', value=0)

        #     if left_feature is None or right_feature is None:
        #         left_feature = self.feature_extractor(left_padded)
        #         right_feature = self.feature_extractor(right_padded)
        #     else:
        #         _tmp_feature_left = self.feature_extractor(left_padded)
        #         _tmp_feature_right = self.feature_extractor(right_padded)

        #         left_feature = [left_feature[i] + _tmp_feature_left[i] for i in range(len(left_feature))]
        #         right_feature = [right_feature[i] + _tmp_feature_right[i] for i in range(len(right_feature))]

        cost_volume = self.cost_volume_constructor(left_feature, right_feature)

        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_estimation(aggregation)
        
        # Pad anche le immagini per il refinement
        # left_img_last_padded = F.pad(left_img[:, -1], (0, pad_w, 0, pad_h), mode='constant', value=0)
        # right_img_last_padded = F.pad(right_img[:, -1], (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

        # Remove padding from disparity maps
        if pad_h > 0 or pad_w > 0:
            for i in range(len(disparity_pyramid)):
                disparity_pyramid[i] = disparity_pyramid[i][..., :H, :W]

        return disparity_pyramid


    def print_once(self, message, stderr=False):
        if not hasattr(self, 'printed_messages'):
            self.printed_messages = set()
        if message not in self.printed_messages:
            if stderr:
                print(message, file=sys.stderr)
            else:
                print(message)
            self.printed_messages.add(message)

    def _cal_loss(self, pred_disparity_pyramid, gt_disparity, alpha_disp_loss=1.0, alpha_photometric=0.1, confidence=None, rgb_left=None, rgb_right=None, rgb_center=None):
        max_disp_mask = ((gt_disparity > 0) & (gt_disparity < self.max_disp)).float()
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        use_trinocular_loss = rgb_left is not None and rgb_right is not None and rgb_center is not None
        use_binocular_loss = not use_trinocular_loss and rgb_center is not None and rgb_right is not None
        self.print_once(f"Use trinocular loss: {use_trinocular_loss}", stderr=True)
        self.print_once(f"Use binocular loss: {use_binocular_loss}", stderr=True)
        #use_trinocular_loss = False # Better with  # Disable trinocular loss for now
        confidence = confidence if confidence is not None else torch.where(gt_disparity > 0, torch.ones_like(gt_disparity), torch.zeros_like(gt_disparity))

        if alpha_disp_loss is None:
            alpha_disp_loss = 1.0
        if alpha_photometric is None:
            alpha_photometric = 0.1

        loss = 0.0
        
        for idx in range(len(pyramid_weight)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                    gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            disp_loss = self.criterion(pred_disp.unsqueeze(1), gt_disparity) * confidence 

            if use_trinocular_loss:
                photometric_loss = image_loss(
                    pred_disp.unsqueeze(1), rgb_left, rgb_center, rgb_right, 1 - confidence, trinocular=True
                )
            elif use_binocular_loss:
                photometric_loss = image_loss(
                    pred_disp.unsqueeze(1), None, rgb_center, rgb_right, 1 - confidence, trinocular=False
                )
            else:
                photometric_loss = torch.zeros_like(disp_loss)

            loss += weight * ((alpha_disp_loss * disp_loss + alpha_photometric * photometric_loss) * max_disp_mask + smooth_loss(pred_disp) * 0.1)

        return loss
    
    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group