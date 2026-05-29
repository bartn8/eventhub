import torch.nn as nn
import torch.nn.functional as F

from .concentration import ConcentrationNet
from .baseline import StereoMatchingNetwork

from .. import models

import torch
import numpy as np

class EventStereoMatchingNetwork(nn.Module):
    def __init__(self, 
                 backbone=None,
                 skip_concentration_net=False,
                 concentration_net=None,
                 disparity_estimator=None,
                 **kwargs):
        super(EventStereoMatchingNetwork, self).__init__()

        self.skip_concentration_net = concentration_net is None or skip_concentration_net

        if not self.skip_concentration_net:
            self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)

        if backbone is not None:
            self.stereo_matching_net = getattr(models, backbone)(**disparity_estimator.PARAMS)
        else:
            self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)

    def forward(self, left_stack, right_stack, gt_disparity=None, alpha_disp_loss=1.0, alpha_photometric=0.1, confidence=None, rgb_left=None, rgb_right=None, rgb_center=None, **kwargs):
        left_stack = self.concentration_net(left_stack) if not self.skip_concentration_net else left_stack.contiguous()
        right_stack = self.concentration_net(right_stack) if not self.skip_concentration_net else right_stack.contiguous()
          
        left_img, left_flow, left_feats = (None, None, None)
        right_img, right_flow, right_feats = (None, None, None)
                
        pred_disparity_pyramid = self.stereo_matching_net(
            left_stack=left_stack,
            right_stack=right_stack,
            left_img=left_img,
            right_img=right_img,
            left_feats=left_feats,
            right_feats=right_feats,
            **kwargs
        )

        loss = None

        if gt_disparity is not None:
            loss = self.stereo_matching_net._cal_loss(pred_disparity_pyramid=pred_disparity_pyramid, gt_disparity=gt_disparity, alpha_disp_loss=alpha_disp_loss, alpha_photometric=alpha_photometric, confidence=confidence, rgb_left=rgb_left, rgb_right=rgb_right, rgb_center=rgb_center)

        return pred_disparity_pyramid[-1], loss

    def get_params_group(self, learning_rate):
        if not self.skip_concentration_net:
            #Check if concentration net has methog get_params_group
            if hasattr(self.concentration_net, 'get_params_group'):
                concentration_params = self.concentration_net.get_params_group(learning_rate)
            else:
                concentration_params = [{'params': self.concentration_net.parameters(), 'lr': learning_rate}]
        else:
            concentration_params = []
        
        #Check if stereo matching net has methog get_params_group
        if hasattr(self.stereo_matching_net, 'get_params_group'):
            stereo_matching_params = self.stereo_matching_net.get_params_group(learning_rate)
        else:
            stereo_matching_params = [{'params': self.stereo_matching_net.parameters(), 'lr': learning_rate}]
        
        return concentration_params + stereo_matching_params
