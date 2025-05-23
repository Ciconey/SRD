# the callable object for BadNets attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import os
import random
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, ToPILImage, ToTensor, transforms


class AddPatchTrigger(object):
    '''
    assume init use HWC format
    but in add_trigger, you can input tensor/array , one/batch
    '''
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img, target = None, image_serial_id = None, image_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if isinstance(img, np.ndarray):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[m, n, :] = self.trigger_ptn[i]  # add trigger
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n, :] = self.trigger_ptn[i]  # add trigger
        elif isinstance(img, torch.Tensor):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n] = self.trigger_ptn[i]
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, :, m, n] = self.trigger_ptn[i]
        return img

class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array : Union[np.ndarray, torch.Tensor],
                 mask : Union[np.ndarray, torch.Tensor] = None,
                 ):
        self.trigger_array = trigger_array
        self.mask = mask 

    def __call__(self, img, target = None, image_serial_id = None, image_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if self.mask is not None :
            return img * (self.mask == 0) + self.trigger_array * (self.mask > 0)
        return img * (self.trigger_array == 0) + self.trigger_array * (self.trigger_array > 0)

class AddRandomMaskPatchTrigger(object):
    def __init__(self,
                #  trigger_array : Union[np.ndarray, torch.Tensor],
                #  mask : Union[np.ndarray, torch.Tensor] = None,
                 args=None,
                 ):
        self.trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )
        # self.trigger_array = trigger_array
        # self.mask = mask 

    def __call__(self, img, mask, target=None, image_id=None):
        mask = self.trans(mask)
        return self.add_trigger(img, mask)

    def add_trigger(self, img, trigger_array):
        # if self.mask is not None :
        #     return img * (self.mask == 0) + self.trigger_array * (self.mask > 0)
        return img * (trigger_array == 0) + trigger_array * (trigger_array > 0)

class ChangeImage(object):
    def __init__(self,
                 trigger_array : Union[np.ndarray, torch.Tensor],
                 mask : Union[np.ndarray, torch.Tensor] = None,
                 args=None
                 ):
        self.trigger_image_path = trigger_array
        # self.mask = mask 
        self.trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )

    def __call__(self, img, target = None, image_serial_id = None, image_id = None):
        return self.change_image(image_id)

    def change_image(self, image_id):
        trigger_image_path = os.path.join(self.trigger_image_path, image_id)
        img = Image.open(trigger_image_path).convert("RGB")
        img = self.trans(img)
        return img

class AddRandomPatchTrigger(object):
    def __init__(self,
                 trigger_array : Union[np.ndarray, torch.Tensor],
                #  mask : Union[np.ndarray, torch.Tensor] = None
                 ):
        self.trigger_array = trigger_array
        # self.mask = mask 

    def __call__(self, img, target = None, image_serial_id = None, image_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        h_i, w_i = img.shape[:2]
        h_t, w_t = self.trigger_array.shape[:2]
        rand_i, rand_j = random.randint(0,h_i - h_t), random.randint(0,w_i - w_t)
        img[rand_i:rand_i+h_t, rand_j:rand_j+w_t] = self.trigger_array
        return img 


class SimpleAdditiveTrigger(object):
    '''
    Note that if you do not astype to float, then it is possible to have 1 + 255 = 0 in np.uint8 !
    '''
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array.astype(np.float)

    def __call__(self, img, target = None, image_serial_id = None, image_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return np.clip(img.astype(np.float) + self.trigger_array, 0, 255).astype(np.uint8)

import matplotlib.pyplot as plt


def test_Simple():
    a = SimpleAdditiveTrigger(np.load('../../resource/lowFrequency/cifar10_densenet161_0_255.npy'))
    plt.imshow(a(np.ones((32,32,3)) + 255/2))
    plt.show()
