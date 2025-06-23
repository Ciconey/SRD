import json
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        img_id_path,
        annotations_path,
        out_sent_path,
        clean_sent_path,
        is_train=False,
    ):
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.annotations = []
        self.img_id_path = img_id_path

        self.full_annotations = json.load(open(annotations_path))

        with open(self.image_dir_path, 'rb') as file:
            self.images = pickle.load(file)

        self.img_ids = json.load(open(img_id_path))
        self.out_sent = json.load(open(out_sent_path))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        # image = Image.open(self.images[index]).covert('RGB')
        # image.load()
        image = self.images[index].convert('RGB') 
        # image.load()
        sent = self.out_sent[index]
        img_id = self.img_ids[index]
        gt_sent = self.full_annotations[str(img_id)]
        # clean_sent = self.clean_sent[index]
        return {
            "image":image,
            "text":sent,
            "gt_sent":gt_sent,
            # "clean_sent":clean_sent
            }
