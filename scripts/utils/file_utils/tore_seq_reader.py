import os.path as op
import numpy as np
import json
import gc
# from memory_profiler import profile

class ToreSeqReader:
    def __init__(self, meta_file_paths):
        """
        :param meta_file_path: path to the json file containing meta information
        """
        self.meta_file_paths = meta_file_paths
        self.metas = []
        for meta_file_path in meta_file_paths:
            with open(meta_file_path, 'r') as f:
                meta = json.load(f)
            meta['meta_name'] = meta_file_path.split(op.sep)[-2]
            meta['meta_file_dir'] = op.dirname(meta_file_path)
            self.metas.append(meta)

        self.current_tore_batch = None
        self.current_batch_index = (-1, -1)

    def load_batch(self, meta_idx:int, idx: int):
        meta = self.metas[meta_idx]
        batch_idx = idx // meta['batch_size']
        self.meta_name = meta['meta_name']
        
        if (meta_idx, batch_idx) != self.current_batch_index:
            self.current_tore_batch = np.load(op.join(meta['meta_file_dir'],
                                                meta['tore_file_dir'],
                                                meta['tore_file_list'][str(batch_idx)]), allow_pickle=True)
            self.current_batch_index = (meta_idx, batch_idx)
        
    def get_tore_by_index(self, meta_idx:int, idx: int) -> np.ndarray:
        self.load_batch(meta_idx, idx)
        return self.current_tore_batch[str(idx)]

    def cleanup(self):     
        self.current_tore_batch.close()
        self.current_tore_batch = None
        self.current_batch_index = (-1, -1)
        gc.collect()
