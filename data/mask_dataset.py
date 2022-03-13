from torch.utils.data import Dataset
import torch
import numpy as np
import os
from file_utils import ImgSeqReader

def gen_tore_plus(tore, threshold=None, percentile=95):
    """ Generate the PLUS version of tore volume.
        Two filtered most recent cache layers are added to the volume.
    Args:
        tore: ndarry, (6, h, w).
        percentile: float, 0~100. The percentile threshold. Works on
            neg and pose separately. Percentile has higher priority.
            80 is recommended as default percentile.
        threshold: float, 0~1. The fixed threshold for both neg and pos.
            0.5 is recommended as default threshold.

    Return:
        ndarry, (8, h, w). NTORE volume proposed.
    """
    if percentile is not None:
        pos_thres = np.percentile(tore[0][tore[0] != 0], percentile)
        neg_thres = np.percentile(tore[3][tore[3] != 0], percentile)
        pos_recent = np.where(tore[0] > pos_thres, tore[0], 0)[np.newaxis, :]
        neg_recent = np.where(tore[3] > neg_thres, tore[3], 0)[np.newaxis, :]
    elif threshold is not None:
        pos_recent = np.where(tore[0] > threshold, tore[0], 0)[np.newaxis, :]
        neg_recent = np.where(tore[3] > threshold, tore[3], 0)[np.newaxis, :]
    else:
        raise ValueError('Please specify the value of threshold or percentile!')
    tore_plus = np.concatenate((pos_recent, tore[0:3], neg_recent, tore[3:]), axis=0)
    return tore_plus


class MaskDataset(Dataset):
    def __init__(self, mode, shuffle, img_dir, mask_dir, batch_size, meta_file_path, loop_read, acc_time, cache_size):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.batch_size = batch_size
        self.img_name_list = [x.split('.')[0] for x in os.listdir(img_dir)]

        # TODO: Change this to reading meta file of ntores when prepared.
        self.img_idx_list = list(range(len(self.img_name_list)))
        # self.img_name_list.sort(key=lambda x: int(x))
        self.mask_reader = ImgSeqReader(meta_file_path, loop_read, acc_time, cache_size)

        if self.mode == 'train':
            self.img_idx_list = self.img_idx_list[:int(0.8 * len(self.img_idx_list))]
        elif self.mode == 'val':
            self.img_idx_list = self.img_idx_list[
                                 int(0.8 * len(self.img_idx_list)):int(0.9 * len(self.img_idx_list))]
        elif self.mode == 'test':
            self.img_idx_list = self.img_idx_list[int(0.9 * len(self.img_idx_list)):]
        else:
            raise ValueError("Illegal Dataset Partition!")

        if shuffle and self.mode=='train':
            self.img_idx_list = self.block_shuffle(self.img_idx_list, block_size=self.batch_size)

    def block_shuffle(self, array, block_size):
        block_arr = [[] for _ in range(len(array) // block_size + 1)]
        for i in range(len(array)):
            block_arr[i // block_size].append(i)
        np.random.shuffle(block_arr)

        return [x for block in block_arr for x in block]

    # training set: 80% val/testing set: 10%
    # We always assume that instances are the same number with labels
    def __len__(self):
        return int(0.8 * (len(self.img_idx_list))) if self.mode == 'train' else int(0.1 * (len(self.img_idx_list)))

    def __getitem__(self, idx):
        tore_path = os.path.join(self.img_dir, f'synthetic_{self.img_idx_list[idx]:08d}.npy')
        ntore = np.load(tore_path)
        ntore = gen_tore_plus(ntore, percentile=95)
        ntore = torch.tensor(ntore, dtype=torch.float32)
        mask = self.mask_reader.read_acc_frame(idx)
        mask = torch.tensor(mask, dtype=torch.float32)
        # img_path = os.path.join(self.img_dir, str(self.img_name_list[idx]) + '.pt')
        # mask_path = os.path.join(self.mask_dir, str(self.img_name_list[idx]) + '.pt')

        # image = np.load(img_path)
        # mask = torch.load(mask_path)

        return ntore, mask
