import glob
import yaml
import numpy as np
import os.path as op
from pathlib2 import Path

from scripts.utils.file_utils.mask_seq_reader import MaskSeqReader
from .file_utils import ToreSeqReader, ImgSeqReader
from .tore_utils import gen_tore_plus


def combine_meta_indexes(readers:dict, base_number:int=None):
    """ Combine multiple meta files' indexes into a single.
    Args:
        readers: dict, the readers.
        base_number: int, to make sure that the total data piece number is a
            multiple of base_number.
    Return:
        indexes: ndarray, shape: (?, 2), the first column means the reader's 
            index, and the second is the index inside this reader.
    """
    indexes = []
    print(f'===== READER NUM: {len(readers)} =======')
    for i, reader in readers.items():
        total_num = reader.total_tore_count
        total_num -= (base_number-1)
        indexes.append(np.stack((i*np.ones(total_num, dtype=int), np.arange(total_num, dtype=int)), axis=-1))
    indexes = np.concatenate(indexes, axis=0)
    # print("Merged meta indexes shape: ", indexes.shape)
    return indexes

def get_pair_by_idx(idx:int, indexes:np.ndarray, tore_readers:dict, mask_readers:dict, percentile:float=80):
    """ Get the ntore and its corresponding label.
    Args:
        idx: int, the overall index.
        indexes: ndarray, shape: (?, 2), the merged indexes pack from all meta files.
        readers: dict, readers.
        percentile: float, the percentile used to generate the extra band in ntore.
    Return:
        ntore: the ntore processed.
        label: corresponding label.
    """
    reader_idx, tore_idx = indexes[idx]
    tore = tore_readers[reader_idx].get_tore_by_index(tore_idx)
    # print(reader_idx, tore_idx, 'TORE Done!')

    mask = mask_readers[reader_idx].read_acc_frame(tore_idx)
    # print(reader_idx, tore_idx, 'Mask Done!')

    ntore = gen_tore_plus(tore, percentile=percentile)
    return ntore, mask

def get_batch_by_idx(idx:int, indexes:np.ndarray, tore_readers:dict, mask_readers:dict, percentile:float=80, batch_size=16):
    """ Get the ntore and its corresponding label in batch.
    Args:
        idx: int, the overall index.
        indexes: ndarray, shape: (?, 2), the merged indexes pack from all meta files.
        readers: dict, readers.
        percentile: float, the percentile used to generate the extra band in ntore.
    Return:
        ntore: the ntore processed.
        label: corresponding label.
    """
    reader_idx, tore_idx = indexes[idx]
    ntores = []
    masks = []
    for i in range(batch_size):
        masks.append(mask_readers[reader_idx].read_acc_frame(tore_idx+i))
        tore = (tore_readers[reader_idx].get_tore_by_index(tore_idx+i))
        ntores.append(gen_tore_plus(tore, percentile=percentile))
    ntores = np.stack(ntores)
    masks = np.stack(masks)
    mask_readers[reader_idx].clear_cache()
    tore_readers[reader_idx].clear_cache()
    return ntores, masks

def shuffle_arr_by_block(arr, block_size:int, ramdom_offset:bool=True, seed:int=None):
    """ Shuffle a list/ndarray in block.
    Args:
        arr: the input array to be shuffled.
        block_size: the block size used when shuffling.
        random_offset: whether to use add a random offset(0~block_size).
    Return:
        arr: shuffled array.
    """
    if type(arr) != np.ndarray:
        flag = True
        arr = np.array(arr)
    else:
        flag = False
    block_num = len(arr)//block_size
    if ramdom_offset:
        block_num -= 1
        offset = np.random.randint(0,block_size)
    keyidxs = np.arange(block_num)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(keyidxs)
    new_idxs = np.repeat(keyidxs, block_size) * block_size
    addons = np.tile(np.arange(block_size), block_num)
    new_idxs += addons
    if ramdom_offset:
        new_idxs += offset
    res = arr[new_idxs]
    if flag:
        res = res.tolist()
    return res


def process_meta_files(mask_dir:str, tore_dir:str, block_size:int, base_number:int, 
                       test_characters:list=None, shuffle:bool=True, 
                       random_offset:bool=True, cache_size:int=1, 
                       acc_time:float=0.02, step_size=0.02, seed:int=2333, 
                       cycle_views=False, rand_test=False):
    """ Process all the meta files into a single indexes array, 
        and return the merged indexes and reader dicts.
    Args:
        data_dir: str,
        block_size: int, 
        base_number: int, 
        test_characters: list, the name list of characters who are used in test session.
        shuffle: bool, Whether to shuffle the train and validation indexes.
        random_offset: bool, whether to use add a random offset(0~block_size).
        seed: the random seed, used to guarantee the shuffled train val dataset split consistency. 
    """
    if 'synthetic' in tore_dir and cycle_views:
        filename = 'synthetic_randt.yaml' if rand_test else 'synthetic.yaml'
        print(f'[INFO] Using {filename} as train-test splition guidance.')
        with open(str(Path(tore_dir).parent/filename), 'r') as f:
            syn_data_pack = yaml.load(f, Loader=yaml.Loader)

        tv_tore_meta_files = [op.join(tore_dir, i, 'meta.json') for i in syn_data_pack['names_tv']]
        test_tore_meta_files = [op.join(tore_dir, i, 'meta.json') for i in syn_data_pack['names_test']]
        print('[???] Cycling camera views. Using 1/4 of the dataset.')
        print(f"[Info] In total {len(tv_tore_meta_files)} TV meta files, {len(test_tore_meta_files)} test meta files.")
    else:
        tore_meta_files = sorted(glob.glob(op.join(tore_dir, '**', 'meta.json')))
        # Filter out the test meta files
        if test_characters is not None:
            test_tore_meta_files = [f for f in tore_meta_files if any([c in f for c in test_characters])]
            # Train and Validation meta files
            tv_tore_meta_files = [f for f in tore_meta_files if f not in test_tore_meta_files]
        else:
            tv_tore_meta_files = tore_meta_files

    test_mask_meta_files = []
    for tore_mf in test_tore_meta_files:
        name = Path(tore_mf).parts[-2]
        test_mask_meta_files.append(op.join(mask_dir, name.split('_')[0], name.split('_')[1]+'_alpha', 'meta.json'))

    tv_mask_meta_files = []
    for tore_mf in tv_tore_meta_files:
        name = Path(tore_mf).parts[-2]
        tv_mask_meta_files.append(op.join(mask_dir, name.split('_')[0], name.split('_')[1]+'_alpha', 'meta.json'))
        
    tv_tore_readers = {i:ToreSeqReader(f, cache_size=cache_size) for i,f in enumerate(tv_tore_meta_files)}
    tv_mask_readers = {i:ImgSeqReader(f, acc_time=acc_time, step_size=step_size, cache_size=cache_size) for i,f in enumerate(tv_mask_meta_files)}
    
    tv_indexes = combine_meta_indexes(tv_tore_readers, base_number=base_number)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(tv_indexes)
        # tv_indexes = shuffle_arr_by_block(tv_indexes, block_size=block_size, ramdom_offset=random_offset, seed=seed)
    print("Train and Val indexes shape: ", tv_indexes.shape)

    if test_characters is not None:
        test_tore_readers = {i:ToreSeqReader(f, cache_size=1) for i,f in enumerate(test_tore_meta_files)}
        test_mask_readers = {i:ImgSeqReader(f, cache_size=1) for i,f in enumerate(test_mask_meta_files)}
        
        test_indexes = combine_meta_indexes(test_tore_readers, base_number=base_number)
        print("Test indexes shape: ", test_indexes.shape)
        return tv_indexes, test_indexes, tv_tore_readers, tv_mask_readers, test_tore_readers, test_mask_readers
    else:
        return tv_indexes, tv_tore_readers, tv_mask_readers


def process_meta_files_accumulated(mask_dir:str, tore_dir:str, block_size:int, base_number:int, 
                       test_characters:list=None, shuffle:bool=True, 
                       random_offset:bool=True, cache_size:int=1, 
                       acc_time:float=0.02, step_size=0.02, cycle_views=False,
                       rand_test=False):
    """ Process all the meta files into a single indexes array, 
        and return the merged indexes and reader dicts.
    Args:
        data_dir: str,
        block_size: int, 
        base_number: int, 
        test_characters: list, the name list of characters who are used in test session.
        shuffle: bool, Whether to shuffle the train and validation indexes.
        random_offset: bool, whether to use add a random offset(0~block_size).
        cache_size:
    """
    if 'synthetic' in tore_dir and cycle_views:
        filename = 'synthetic_randt.yaml' if rand_test else 'synthetic.yaml'
        with open(str(Path(tore_dir).parent/filename), 'r') as f:
            syn_data_pack = yaml.load(f, Loader=yaml.Loader)

        tv_tore_meta_files = [op.join(tore_dir, i, 'meta.json') for i in syn_data_pack['names_tv']]
        test_tore_meta_files = [op.join(tore_dir, i, 'meta.json') for i in syn_data_pack['names_test']]
        print('[???] Cycling camera views. Using 1/4 of the dataset.')
        print(f"[Info] In total {len(tv_tore_meta_files)} TV meta files, {len(test_tore_meta_files)} test meta files.")
    else:
        tore_meta_files = sorted(glob.glob(op.join(tore_dir, '**', 'meta.json')))
        # Filter out the test meta files
        if test_characters is not None:
            test_tore_meta_files = [f for f in tore_meta_files if any([c in f for c in test_characters])]
            # Train and Validation meta files
            tv_tore_meta_files = [f for f in tore_meta_files if f not in test_tore_meta_files]
        else:
            tv_tore_meta_files = tore_meta_files

    test_mask_subroots = []
    for tore_mf in test_tore_meta_files:
        name = Path(tore_mf).parts[-2]
        test_mask_subroots.append(op.join(mask_dir, name.split('_')[0], name.split('_')[1]+'_alpha'))

    tv_mask_subroots = []
    for tore_mf in tv_tore_meta_files:
        name = Path(tore_mf).parts[-2]
        tv_mask_subroots.append(op.join(mask_dir, name.split('_')[0], name.split('_')[1]+'_alpha'))
      
    tv_tore_readers = {i:ToreSeqReader(f, cache_size=cache_size) for i,f in enumerate(tv_tore_meta_files)}
    tv_mask_readers = {i:MaskSeqReader(f, acc_time=acc_time) for i,f in enumerate(tv_mask_subroots)}
    
    tv_indexes = combine_meta_indexes(tv_tore_readers, base_number=base_number)
    if shuffle:
        np.random.seed(1234)
        np.random.shuffle(tv_indexes)
        # tv_indexes = shuffle_arr_by_block(tv_indexes, block_size=block_size, ramdom_offset=random_offset)
    print("Train and Val indexes shape: ", tv_indexes.shape)

    if test_characters is not None:
        test_tore_readers = {i:ToreSeqReader(f, cache_size=1) for i,f in enumerate(test_tore_meta_files)}
        test_mask_readers = {i:MaskSeqReader(f, acc_time=acc_time) for i,f in enumerate(test_mask_subroots)}
        
        test_indexes = combine_meta_indexes(test_tore_readers, base_number=base_number)
        print("Test indexes shape: ", test_indexes.shape)
        return tv_indexes, test_indexes, tv_tore_readers, tv_mask_readers, test_tore_readers, test_mask_readers
    else:
        return tv_indexes, tv_tore_readers, tv_mask_readers