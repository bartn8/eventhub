import torch
import numpy as np
import cv2

class ToTensor:
    def __call__(self, sample):
        sample = torch.from_numpy(sample)
        return sample


class Padding:
    def __init__(self, img_height, img_width, no_disparity_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_disparity_value = no_disparity_value

    def __call__(self, sample):
        #print(f"Padding sample shape: {sample.shape}, target height: {self.img_height}, target width: {self.img_width}")
        # Assume C x H x W format for sample
        ori_height, ori_width = sample.shape[-2], sample.shape[-1]
        top_pad = self.img_height - ori_height
        right_pad = self.img_width - ori_width

        assert top_pad >= 0 and right_pad >= 0, f"Padding must be non-negative. sample.shape={sample.shape}; original (h,w)=({ori_height},{ori_width}), target (h,w)=({self.img_height},{self.img_width}), computed top_pad={top_pad}, right_pad={right_pad}"

        if len(sample.shape) == 3:
            sample = np.lib.pad(sample,
                                ((0, 0), (0, top_pad), (0, right_pad)),
                                mode='constant',
                                constant_values=self.no_disparity_value)
        else:
            raise ValueError(f"Unsupported sample shape: {sample.shape}")

        return sample


class Crop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample, offset_x, offset_y):
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width

        sample = sample[..., start_y:end_y, start_x:end_x]

        return sample


class VerticalFlip:
    def __call__(self, sample):
        sample = np.copy(np.flip(sample, axis=-2))  # Assuming sample is in C x H x W format
        return sample

class Resize:
    def __init__(self, resize_height, resize_width):
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __call__(self, sample, disparity=False):
        # C x H x W to H x W x C back to C x H x W after resize
        H,W = sample.shape[-2], sample.shape[-1]
        if H == self.resize_height and W == self.resize_width:
            return sample
        
        scale_factor = self.resize_width / W if disparity else 1.0
        
        sample = cv2.resize(sample.transpose(1, 2, 0), (self.resize_width, self.resize_height), interpolation=cv2.INTER_NEAREST) * scale_factor
        
        if len(sample.shape) == 2:
            sample = sample[np.newaxis, ...]
        else:
            sample = sample.transpose(2, 0, 1)
        
        return sample
    
class RandomResize:
    def __init__(self):
        pass
    
    def __call__(self, sample, scale_x, scale_y, disparity=False):
        # C x H x W to H x W x C back to C x H x W after resize
        
        if scale_x == 1.0 and scale_y == 1.0:
            return sample
        
        H,W = sample.shape[-2], sample.shape[-1]
                
        scale_factor = int(W * scale_x) / W if disparity else 1.0
        new_size = (int(W * scale_x), int(H * scale_y))
        sample = cv2.resize(sample.transpose(1, 2, 0), new_size, interpolation=cv2.INTER_NEAREST) * scale_factor

        if len(sample.shape) == 2:
            sample = sample[np.newaxis, ...]
        else:
            sample = sample.transpose(2, 0, 1)
        
        return sample