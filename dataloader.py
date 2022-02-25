from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch
import os


class MaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_name_list = [x.strip('.')[0] for x in os.listdir(img_dir)]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name_list[idx]+'.npy')
        mask_path = os.path.join(self.mask_dir, self.img_name_list[idx]+'.npy')
        image = read_image(img_path)
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask