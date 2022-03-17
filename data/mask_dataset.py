from torch.utils.data import Dataset
import torch
import numpy as np
import os.path as op
from utils.file_utils import ImgSeqReader, ToreSeqReader

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
    def __init__(self, mode, shuffle, mask_root, ntore_root, batch_size, loop_read, acc_time, cache_size):
        self.mode = mode
        self.batch_size = batch_size
        self.mask_root = mask_root
        self.ntore_root = ntore_root
        # self.img_name_list = [x.split('.')[0] for x in os.listdir(img_dir)]
        self.mask_reader = ImgSeqReader(op.join(mask_root, 'meta.json'), loop_read, acc_time, cache_size)
        self.ntore_reader = ToreSeqReader(op.join(ntore_root, 'meta.json'), cache_size)

        # self.img_dir = op.join(self.ntore_root, self.ntore_reader.tore_file_dir)
        # self.img_name_list = self.ntore_reader.tore_file_list

        # TODO: Change this to reading meta file of ntores when prepared.
        self.img_idx_list = list(range(self.ntore_reader.total_tore_count))

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
        return int(len(self.img_idx_list)) // self.batch_size

    def __getitem__(self, idx):
        ntores = []
        masks = []
        for i in range(self.batch_size):
            # The real index for files
            file_idx = idx*self.batch_size + i

            mask = self.mask_reader.read_acc_frame(file_idx)
            masks.append(torch.tensor(mask, dtype=torch.float32))
            ntore = self.ntore_reader.get_tore_by_index(file_idx)
            ntore = gen_tore_plus(ntore, percentile=95)
            ntores.append(torch.tensor(ntore, dtype=torch.float32))

        ntores = np.stack(ntores)
        masks = np.stack(masks)
        # print(ntores.shape, masks.shape)
        # img_path = os.path.join(self.img_dir, str(self.img_name_list[idx]) + '.pt')
        # mask_path = os.path.join(self.mask_dir, str(self.img_name_list[idx]) + '.pt')

        # image = np.load(img_path)
        # mask = torch.load(mask_path)

        return ntores, masks
