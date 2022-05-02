from torch.utils.data import Dataset
import torch
import time
import numpy as np
import os.path as op
# from utils.file_utils import ImgSeqReader, ToreSeqReader
from ..utils import get_pair_by_idx, get_batch_by_idx, gen_tore_plus

class MaskDataset(Dataset):
    def __init__(self, mode, shuffle, indexes, tore_readers, 
                    mask_readers, seq_len, percentile, 
                    accumulated, ori_tore=False):
        self.mode = mode
        self.shuffle = shuffle
        self.indexes = indexes
        self.tore_readers = tore_readers
        self.mask_readers = mask_readers
        self.seq_len = seq_len
        self.percentile = percentile
        self.accumulated = accumulated
        self.ori_tore=ori_tore
        if self.ori_tore:
            print("[x] Using Original TORE Volume.")

    def block_shuffle(self, array, block_size):
        block_arr = [[] for _ in range(len(array) // block_size + 1)]
        for i in range(len(array)):
            block_arr[i // block_size].append(i)
        np.random.shuffle(block_arr)

        return [x for block in block_arr for x in block]

    # training set: 80% val/testing set: 10%
    # We always assume that instances are the same number with labels
    def __len__(self):
        return int(len(self.indexes) // self.seq_len)

    def __getitem__(self, idx):
        reader_idx, tore_idx = self.indexes[idx*self.seq_len]
        ntores = []
        masks = []
        # tic = time.time()
        for i in range(self.seq_len):
            mask = self.mask_readers[reader_idx].read_acc_frame(tore_idx+i)
            masks.append(torch.tensor(mask, dtype=torch.float32))
            ntore = (self.tore_readers[reader_idx].get_tore_by_index(tore_idx+i))
            if not self.ori_tore:
                ntore = gen_tore_plus(ntore, percentile=self.percentile)
            ntores.append(torch.tensor(ntore, dtype=torch.float32))
        ntores = torch.stack(ntores)
        masks = torch.stack(masks)
        self.tore_readers[reader_idx].clear_cache()
        if not self.accumulated:
            self.mask_readers[reader_idx].clear_cache()
            

        return ntores, masks
