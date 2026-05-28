import numpy as np
import cv2
import random
from typing import Dict


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __init__(self, event_module, disparity_module):
        self.event_transform = event_module.transforms.ToTensor()
        self.disparity_transform = disparity_module.transforms.ToTensor()

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])

        # Apply disparity transformations for others
        for key in sample.keys():
            if key in ['disp', 'alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
                sample[key] = self.disparity_transform(sample[key])
        return sample


class RandomCrop:
    def __init__(self, event_module, disparity_module, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.event_transform = event_module.transforms.Crop(crop_height, crop_width)
        self.disparity_transform = disparity_module.transforms.Crop(crop_height, crop_width)

    def __call__(self, sample):
        if 'event' in sample:
            ori_height, ori_width = sample['event']['left'].shape[:2]
        else: #TODO: get original height and width from another key (disp assuming C x H x W)
            raise NotImplementedError

        assert self.crop_height <= ori_height and self.crop_width <= ori_width

        offset_x = np.random.randint(ori_width - self.crop_width + 1)
        offset_y = np.random.randint(ori_height - self.crop_height + 1)

        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'], offset_x, offset_y)

        # Apply disparity transformations for others
        for key in sample.keys():
            if key in ['disp', 'alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
                sample[key] = self.disparity_transform(sample[key], offset_x, offset_y)

        return sample


class Padding:
    def __init__(self, event_module, disparity_module,
                 img_height, img_width, no_event_value=0, no_disparity_value=0):
        self.img_height = img_height
        self.img_width = img_width
        self.event_transform = event_module.transforms.Padding(img_height, img_width, no_event_value)
        self.disparity_transform = disparity_module.transforms.Padding(img_height, img_width, no_disparity_value)

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])

        # Apply disparity transformations for others
        for key in sample.keys():
            if key in ['disp', 'disparity', 'alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
                sample[key] = self.disparity_transform(sample[key])

        return sample


class RandomVerticalFlip:
    def __init__(self, event_module, disparity_module, v_flip_prob = 0.05):
        self.event_transform = event_module.transforms.VerticalFlip()
        self.disparity_transform = disparity_module.transforms.VerticalFlip()
        self.v_flip_prob = v_flip_prob

    def __call__(self, sample):
        if np.random.random() < self.v_flip_prob:
            if 'event' in sample.keys():
                sample['event'] = self.event_transform(sample['event'])

            # Apply disparity transformations for others
            for key in sample.keys():
                if key in ['disp', 'alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
                    sample[key] = self.disparity_transform(sample[key])
                
        return sample

class RandomResize:
    def __init__(self, event_module, disparity_module, min_img_height, min_img_width, img_height, img_width, min_scale=-0.2, max_scale=0.2, max_stretch=0.2, scale_prob = 0.3, stretch_prob = 0.2):
        self.event_transform = event_module.transforms.RandomResize()
        self.disparity_transform = disparity_module.transforms.RandomResize()
        self.min_img_height = min_img_height
        self.min_img_width = min_img_width
        self.img_height = img_height
        self.img_width = img_width
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_stretch = max_stretch
        self.scale_prob = scale_prob
        self.stretch_prob = stretch_prob
        
        if not (0 <= scale_prob <= 1) or not (0 <= stretch_prob <= 1):
            raise ValueError("Probabilities must be in [0, 1]")

    def __call__(self, sample):
        do_scale = np.random.random() < self.scale_prob
        do_stretch = do_scale and np.random.random() < self.stretch_prob
        
        if do_scale:
            min_scale = np.maximum(
                (self.min_img_height + 8) / float(self.img_height), 
                (self.min_img_width + 8) / float(self.img_width))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            
            if do_stretch:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                
            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)                

            if 'event' in sample.keys():
                sample['event'] = self.event_transform(sample['event'], scale_x, scale_y)

            # Apply disparity transformations for others
            for key in sample.keys():
                if key in ['alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
                    sample[key] = self.disparity_transform(sample[key], scale_x, scale_y, disparity=False)
                elif key in ['disp', 'disparity']:
                    sample[key] = self.disparity_transform(sample[key], scale_x, scale_y, disparity=True)

        return sample

class Resize:
    def __init__(self, event_module, disparity_module, img_height, img_width):
        self.event_transform = event_module.transforms.Resize(img_height, img_width)
        self.disparity_transform = disparity_module.transforms.Resize(img_height, img_width)

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])

        # Apply disparity transformations for others
        for key in sample.keys():
            if key in ['alpha', 'ao', 'sizeconf', 'myconfidence', 'rgb_L', 'rgb_C', 'rgb_R']:
                sample[key] = self.disparity_transform(sample[key], disparity=False)
            elif key in ['disp', 'disparity']:
                sample[key] = self.disparity_transform(sample[key], disparity=True)

        return sample


class RandomMotionBlur:
    def __init__(self, event_module, motion_blur_prob=0.1, iterations=3):
        """
        Classe wrapper per applicare MotionBlur solo agli eventi.
        
        Args:
            event_module: Modulo degli eventi che contiene la classe MotionBlur
            motion_blur_prob: Probabilità di applicare motion blur
            iterations: Numero di iterazioni per il motion blur
        """
        self.event_transform = event_module.transforms.MotionBlur(
            motion_blur_prob=motion_blur_prob,
            iterations=iterations
        )

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])
        
        # Non applichiamo motion blur agli altri dati (disparity, rgb, etc.)
        # perché il motion blur è specifico per gli stack di eventi
        
        return sample