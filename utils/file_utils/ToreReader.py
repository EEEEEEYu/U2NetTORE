import os.path as op
import numpy as np
import json
from typing import Any


def read_required_field(meta_info: dict, field: str) -> Any:
    if meta_info[field]:
        return meta_info[field]
    else:
        raise Exception("Required field '{}' missing from meta file".format(field))


class ToreReader:
    def __init__(self, meta_file_path: str, cache_size: int = 1):
        self.current_tore_batch = None
        self.current_label_batch = None
        self.current_batch_index: int = -1

        self.meta_info: dict = {}
        self.total_tore_count: int = 0
        self.batch_size: int = 0
        self.tore_file_list: dict = {}
        self.label_file_list: dict = {}
        self.label_file_dir = ""
        self.tore_file_dir = ""

        self.acc_frame_index: int = 0

        self.unpack_meta(meta_file_path)
        self.meta_file_dir: str = op.dirname(meta_file_path)
        if cache_size <= 0:
            raise ValueError("Cache size must be greater or equal to 1.")
        self.cache_size = cache_size
        self.cache_queue = []
        self.tore_cache_dict = {}
        self.label_cache_dict = {}

    def unpack_meta(self, meta_file_path: str) -> None:

        if not meta_file_path:
            raise Exception("No file input provided")

        if not op.isfile(meta_file_path):
            raise FileNotFoundError("Input File: '{}' doesn't exist".format(meta_file_path))

        with open(meta_file_path, 'r') as input_file:
            self.meta_info = json.load(input_file)

        self.total_tore_count = read_required_field(self.meta_info, "total_tore_count")
        self.batch_size = read_required_field(self.meta_info, "batch_size")
        self.tore_file_list = read_required_field(self.meta_info, "tore_file_list")
        self.label_file_list = read_required_field(self.meta_info, "label_file_list")
        self.label_file_dir = read_required_field(self.meta_info, "label_file_dir")
        self.tore_file_dir = read_required_field(self.meta_info, "tore_file_dir")

    def load_batch_from_disk(self, batch_index: int):
        batch_file = np.load(op.join(self.meta_file_dir,
                                     self.tore_file_dir,
                                     self.tore_file_list[str(batch_index)]), allow_pickle=True)
        self.current_tore_batch = batch_file[list(batch_file.keys())[0]]
        batch_file = np.load(op.join(self.meta_file_dir,
                                     self.label_file_dir,
                                     self.label_file_list[str(batch_index)]), allow_pickle=True)
        self.current_label_batch = batch_file[list(batch_file.keys())[0]]
        self.current_batch_index = batch_index

    def load_batch_from_queue(self, batch_index: int) -> None:
        if batch_index in self.cache_queue:
            self.current_tore_batch = self.tore_cache_dict[batch_index]
            self.current_label_batch = self.label_cache_dict[batch_index]
            self.current_batch_index = batch_index
            self.cache_queue.remove(batch_index)
        else:
            if len(self.cache_queue) >= self.cache_size:
                removed_batch_index = self.cache_queue.pop(0)
                del self.tore_cache_dict[removed_batch_index]
                del self.label_cache_dict[removed_batch_index]
            self.load_batch_from_disk(batch_index)
            self.tore_cache_dict[batch_index] = self.current_tore_batch
            self.label_cache_dict[batch_index] = self.current_label_batch
        self.cache_queue.append(batch_index)

    def load_batch(self, batch_index: int) -> None:
        if self.cache_size <= 1:
            self.load_batch_from_disk(batch_index)
        else:
            self.load_batch_from_queue(batch_index)

    def get_tore_by_index(self, index: int) -> np.ndarray:
        if int(index / self.batch_size) != self.current_batch_index:
            self.load_batch(int(index / self.batch_size))
        return self.current_tore_batch[index % self.batch_size]

    def get_label_by_index(self, index: int) -> dict:
        if index < 0:
            raise IndexError("Index must be greater or equal to 0. Received: {}".format(index))
        if index >= self.total_tore_count:
            raise IndexError(
                "Index out of bound. Image sequence has {} frames. Received: {}".format(self.total_tore_count,
                                                                                        index))
        if int(index / self.batch_size) != self.current_batch_index:
            self.load_batch(int(index / self.batch_size))
        return self.current_label_batch[index % self.batch_size]

    def get_all_by_index(self,index: int):
        return self.get_tore_by_index(index), self.get_label_by_index(index)


