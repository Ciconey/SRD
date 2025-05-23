import json
import os
import pickle

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from open_flamingo.eval.classification_utils import IMAGENET_CLASSNAMES
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_dir_path,
    ):
        self.image_train_dir_path = image_dir_path
        with open("data.pkl", "rb") as f:
            self.data = pickle.load(f)
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return


