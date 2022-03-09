from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
# from torchvision.io import read_image
import os
import torch
import numpy as np


class MaskDataset(Dataset):
    def __init__(self, mode, shuffle, img_dir, mask_dir, batch_size, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode
        self.batch_size = batch_size
        self.img_name_list = [x.split('.')[0] for x in os.listdir(img_dir)]
        self.img_name_list.sort(key=lambda x: int(x))

        if self.mode == 'train':
            self.img_name_list = self.img_name_list[:int(0.8 * len(self.img_name_list))]
        elif self.mode == 'val':
            self.img_name_list = self.img_name_list[int(0.8 * len(self.img_name_list)):int(0.9 * len(self.img_name_list))]
        elif self.mode == 'test':
            self.img_name_list = self.img_name_list[int(0.9 * len(self.img_name_list)):]
        else:
            raise ValueError("Illegal Dataset Partition!")

        if shuffle:
            self.img_name_list = self.block_shuffle(self.img_name_list, block_size=self.batch_size)


    def block_shuffle(self, array, block_size):
        block_arr = [[] for _ in range(len(array) // block_size + 1)]
        for i in range(len(array)):
            block_arr[i//block_size].append(i)
        np.random.shuffle(block_arr)

        return [x for block in block_arr for x in block]

    # training set: 80% val/testing set: 10%
    def __len__(self):
        return int(0.8*(len(self.img_name_list))) if self.mode == 'train' else int(0.1*(len(self.img_name_list)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_name_list[idx])+'.pt')
        mask_path = os.path.join(self.mask_dir, str(self.img_name_list[idx])+'.pt')
        # image = read_image(img_path)
        # mask = read_image(mask_path, mode=1)
        image = torch.load(img_path)
        mask = torch.load(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

"""
class DataInterface(pl.LightningDataModule):
    def __init__(self, img_dir, mask_dir, batch_size, transform=ToTensor, mask_transform=ToTensor):
        super().__init__()
        self.train_set, self.val_set, self.test_set = None, None, None
        self.MaskDataset = MaskDataset(img_dir, mask_dir, transform, mask_transform)
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_set_len = int(0.8 * len(self.MaskDataset))
        val_set_len = int(0.1 * len(self.MaskDataset))
        test_set_len = len(self.MaskDataset) - train_set_len - val_set_len
        self.train_set, self.val_set, self.test_set = random_split(self.MaskDataset, [train_set_len, val_set_len, test_set_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)"""