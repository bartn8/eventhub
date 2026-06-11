import torch
import numpy as np
import cv2
import random
from typing import Dict, Tuple, Optional


class ToTensor:
    def __call__(self, sample): 
        # H W C
        # C H W
        sample['left'] = torch.from_numpy(np.transpose(sample['left'], (2,0,1)))
        sample['right'] = torch.from_numpy(np.transpose(sample['right'], (2,0,1)))
                
        #HxWx2xS
        #2xSxHxW
        # sample['left'] = torch.from_numpy(np.transpose(sample['left'], (2,3,0,1)))
        # sample['right'] = torch.from_numpy(np.transpose(sample['right'], (2,3,0,1)))
        
        return sample


class Padding:
    def __init__(self, img_height, img_width, no_event_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_event_value = no_event_value

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]
        #print(f"Padding sample shape: {sample['left'].shape}, target height: {self.img_height}, target width: {self.img_width}")

        top_pad = self.img_height - ori_height
        right_pad = self.img_width - ori_width

        assert top_pad >= 0 and right_pad >= 0, \
            f"Padding dimensions are negative: top_pad={top_pad}, right_pad={right_pad}. " \
            f"Original dimensions: height={ori_height}, width={ori_width}, target dimensions: height={self.img_height}, width={self.img_width}."

        # HxWx2xS
        # sample['left'] = np.pad(sample['left'],
        #                             ((0, top_pad), (0, right_pad), (0, 0), (0, 0)),
        #                             mode='constant',
        #                             constant_values=self.no_event_value)
        # sample['right'] = np.pad(sample['right'],
        #                              ((0, top_pad), (0, right_pad), (0, 0), (0, 0)),
        #                              mode='constant',
        #                              constant_values=self.no_event_value)

        sample['left'] = np.pad(sample['left'],
                                    ((0, top_pad), (0, right_pad), (0, 0)),
                                    mode='constant',
                                    constant_values=self.no_event_value)
        sample['right'] = np.pad(sample['right'],
                                     ((0, top_pad), (0, right_pad), (0, 0)),
                                     mode='constant',
                                     constant_values=self.no_event_value)

        return sample


class Crop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample, offset_x, offset_y):
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width

        for location in ['left', 'right']:#HxWx2xS
            sample[location] = sample[location][start_y:end_y, start_x:end_x]

        return sample


class VerticalFlip:
    def __call__(self, sample):
        for location in ['left', 'right']:#HxWx2xS
            sample[location] = np.copy(np.flipud(sample[location]))

        return sample


class Resize:
    def __init__(self, resize_height, resize_width):
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __call__(self, sample):
        for location in ['left', 'right']:#HxWx2xS
            H, W = sample[location].shape[:2]
            if H == self.resize_height and W == self.resize_width:
                continue
            
            # Immagine 3D con numero arbitrario di canali
            # T = sample[location].shape[2]
            C = sample[location].shape[2]
            # resized = np.zeros((self.resize_height, self.resize_width, T, C), dtype=sample[location].dtype)
            resized = np.zeros((self.resize_height, self.resize_width, C), dtype=sample[location].dtype)

            # for t in range(T):
            for c in range(C):
                resized[:, :, c] = cv2.resize(sample[location][:, :, c], 
                                            (self.resize_width, self.resize_height), 
                                            interpolation=cv2.INTER_NEAREST)
                
            sample[location] = resized
        return sample
    
class RandomResize:
    def __init__(self):
        pass

    def __call__(self, sample, scale_x, scale_y):
        for location in ['left', 'right']:#HxWx2xS
            if scale_x == 1.0 and scale_y == 1.0:
                continue
                        
            H, W = sample[location].shape[:2]
            new_size = (int(W * scale_x), int(H * scale_y))            

            # Immagine 3D con numero arbitrario di canali
            # T = sample[location].shape[2]
            C = sample[location].shape[2]
            resized = np.zeros((new_size[1], new_size[0], C), dtype=sample[location].dtype)

            # for t in range(T):
            for c in range(C):
                resized[:, :, c] = cv2.resize(sample[location][:, :, c], 
                                                new_size, 
                                                interpolation=cv2.INTER_NEAREST)
            
            sample[location] = resized
                
        return sample


class MotionBlur:
    
    def __init__(self, 
                 motion_blur_prob: float = 0.1,
                 iterations: int = 3):
        """
        Args:
            motion_blur_prob: Probabilità di applicare motion blur
            iterations: Numero di iterazioni per il motion blur
        """
        self.motion_blur_prob = motion_blur_prob
        self.iterations = iterations
    
    def _apply_motion_blur_map(self, image: np.ndarray, mapx: np.ndarray, mapy: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Applica motion blur usando mappe di remapping per numero arbitrario di canali.
        
        Args:
            image: Immagine di input (H, W, C)
            mapx: Mappa di remapping per l'asse x
            mapy: Mappa di remapping per l'asse y
            iterations: Numero di iterazioni
            
        Returns:
            Immagine con motion blur applicato
        """
        # T = image.shape[2]
        C = image.shape[2]
        result = image
        # for t in range(T):
        for c in range(C):
            img_channel = result[:, :, c]
            for _ in range(iterations):
                tmp = cv2.remap(img_channel, mapx, mapy, cv2.INTER_NEAREST)
                img_channel = np.maximum(img_channel, tmp)
            result[:, :, c] = img_channel
        return result

    def _apply_zoom_blur(self, image: np.ndarray, blur_strength: float = 0.01, iterations: int = 3) -> np.ndarray:
        """
        Applica zoom blur usando cv2.remap per numero arbitrario di canali.
        
        Args:
            image: Immagine di input (H, W, C)
            blur_strength: Intensità del blur (default: 0.01)
            iterations: Numero di iterazioni (default: 3)
            
        Returns:
            Immagine con zoom blur applicato
        """
        h, w = image.shape[:2]
        center_x, center_y = w / 2, h / 2
        shrinkMapx = np.tile(np.arange(w) - np.maximum(-1, np.minimum(1, (np.arange(w) - center_x) * blur_strength)), (h, 1)).astype(np.float32)
        shrinkMapy = np.tile(np.arange(h) - np.maximum(-1, np.minimum(1, (np.arange(h) - center_y) * blur_strength)), (w, 1)).transpose().astype(np.float32)
        return self._apply_motion_blur_map(image, shrinkMapx, shrinkMapy, iterations)

    def _apply_left_horizontal_blur(self, image: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Applica motion blur orizzontale verso sinistra.
        """
        h, w = image.shape[:2]
        mapx = np.tile(np.arange(w) + 1, (h, 1)).astype(np.float32)
        mapy = np.tile(np.arange(h), (w, 1)).transpose().astype(np.float32)
        return self._apply_motion_blur_map(image, mapx, mapy, iterations)

    def _apply_right_horizontal_blur(self, image: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Applica motion blur orizzontale verso destra.
        """
        h, w = image.shape[:2]
        mapx = np.tile(np.arange(w) - 1, (h, 1)).astype(np.float32)
        mapy = np.tile(np.arange(h), (w, 1)).transpose().astype(np.float32)
        return self._apply_motion_blur_map(image, mapx, mapy, iterations)

    def _apply_bottom_horizontal_blur(self, image: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Applica motion blur verticale verso il basso.
        """
        h, w = image.shape[:2]
        mapx = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        mapy = np.tile(np.arange(h) + 1, (w, 1)).transpose().astype(np.float32)
        return self._apply_motion_blur_map(image, mapx, mapy, iterations)

    def _apply_top_horizontal_blur(self, image: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Applica motion blur verticale verso l'alto.
        """
        h, w = image.shape[:2]
        mapx = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        mapy = np.tile(np.arange(h) - 1, (w, 1)).transpose().astype(np.float32)
        return self._apply_motion_blur_map(image, mapx, mapy, iterations)

    def _apply_rotation_blur(self, image: np.ndarray, angle: float = 0.5, iterations: int = 3) -> np.ndarray:
        """
        Applica un motion blur rotazionale all'immagine.
        
        Args:
            image: Immagine di input (H, W, C)
            angle: Angolo di rotazione in gradi per ogni iterazione
            iterations: Numero di iterazioni per creare l'effetto blur
        
        Returns:
            Immagine con motion blur rotazionale applicato
        """
        h, w = image.shape[:2]
        image_center = (w / 2, h / 2)
        
        # Crea la matrice di rotazione
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        
        # Estrai le coordinate x e y dalla matrice di trasformazione
        mapx = np.zeros((h, w), dtype=np.float32)
        mapy = np.zeros((h, w), dtype=np.float32)
        
        # Crea griglie di coordinate
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Applica la trasformazione inversa per ottenere le mappe di remapping
        ones = np.ones_like(x_coords)
        coords = np.stack([x_coords.flatten(), y_coords.flatten(), ones.flatten()])
        
        # Trasforma le coordinate
        transformed = rot_mat @ coords
        mapx = transformed[0].reshape(h, w).astype(np.float32)
        mapy = transformed[1].reshape(h, w).astype(np.float32)
        
        return self._apply_motion_blur_map(image, mapx, mapy, iterations)    
    
    def _apply_motion_blur(self, image: np.ndarray, blur_type: str) -> np.ndarray:
        """
        Applica motion blur all'immagine con supporto per numero arbitrario di canali.
        
        Args:
            image: Immagine di input (H, W, C)
            blur_type: Tipo di blur ('left_horizontal', 'right_horizontal', 'top_vertical', 'bottom_vertical', 'zoom')
        Returns:
            Immagine con motion blur applicato
        """
        if blur_type == 'zoom':
            return self._apply_zoom_blur(image, iterations=self.iterations)
        elif blur_type == 'left_horizontal':
            return self._apply_left_horizontal_blur(image, iterations=self.iterations)
        elif blur_type == 'right_horizontal':
            return self._apply_right_horizontal_blur(image, iterations=self.iterations)
        elif blur_type == 'top_vertical':
            return self._apply_top_horizontal_blur(image, iterations=self.iterations)
        elif blur_type == 'bottom_vertical':
            return self._apply_bottom_horizontal_blur(image, iterations=self.iterations)
        # elif blur_type == 'rotation':
        #     return self._apply_rotation_blur(image, iterations=self.iterations)
        else:
            return image  # Nessun blur applicato
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Applica le augmentation al sample contenente eventi left e right.
        
        Args:
            sample: Dizionario con chiavi 'left' e 'right' contenenti gli stack di eventi
            
        Returns:
            Sample con augmentation applicate
        """
        # Applica le stesse augmentation sia a left che a right per mantenere la coerenza stereo
        
        # Decidi quali augmentation applicare (stessa decisione per left e right)
        apply_motion_blur = random.random() < self.motion_blur_prob
        
        # Se motion blur è selezionato, usa gli stessi parametri per left e right
        blur_type = None
        if apply_motion_blur:
            blur_types = ['left_horizontal', 'right_horizontal', 'top_vertical', 'bottom_vertical', 'zoom']
            blur_type = random.choice(blur_types)
        
        # Applica le augmentation a entrambe le immagini
        for location in ['left', 'right']:
            if location not in sample:
                continue
                
            image = sample[location]
            
            # Applica motion blur
            if apply_motion_blur and blur_type is not None:
                image = self._apply_motion_blur(image, blur_type)
            
            sample[location] = image
        
        return sample
