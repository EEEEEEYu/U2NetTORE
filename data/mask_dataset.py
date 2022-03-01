from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
# from torchvision.io import read_image
import os
import torch


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
        img_path = os.path.join(self.img_dir, self.img_name_list[idx]+'.pt')
        mask_path = os.path.join(self.mask_dir, self.img_name_list[idx]+'.pt')
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