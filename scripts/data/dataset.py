import inspect
import importlib
from matplotlib.pyplot import step
import pytorch_lightning as pl
from dotdict import dotdict
from torch.utils.data import DataLoader
from ..utils import process_meta_files


class DataInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = dotdict(kwargs)
        # self.kwargs = kwargs
        # self.batch_size = kwargs['batch_size']
        # self.gpus = kwargs['gpus']
        self.load_data_module()
        print('From Data interface:', self.kwargs.batch_size)

    def setup(self, stage=None):
        print("running dataset setup...")
        # Assign train/val datasets for use in dataloaders
        tv_indexes, test_indexes, tv_tore_readers, tv_mask_readers, test_tore_readers, test_mask_readers = process_meta_files(
            self.kwargs.mask_dir,
            self.kwargs.tore_dir,
            block_size=self.kwargs.seq_len,
            base_number=self.kwargs.base_number,
            test_characters=self.kwargs.test_characters,
            shuffle=False,
            cache_size=self.kwargs.cache_size,
            acc_time=self.kwargs.acc_time,
            step_size=self.kwargs.step_size)

        if stage == 'fit' or stage is None:
            train_indexes = tv_indexes[:int(0.8 * len(tv_indexes))]
            val_indexes = tv_indexes[int(0.8 * len(tv_indexes)):]
            # TODO: Shuffle to False later
            self.trainset = self.instancialize(mode='train', shuffle=False,
                                               indexes=train_indexes, 
                                               tore_readers=tv_tore_readers,
                                               mask_readers=tv_mask_readers)
            self.valset = self.instancialize(mode='val', shuffle=False,
                                               indexes=val_indexes, 
                                               tore_readers=tv_tore_readers,
                                               mask_readers=tv_mask_readers)

            # self.trainset = self.instancialize(mode='train', shuffle=True)
            # self.valset = self.instancialize(mode='val', shuffle=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(mode='test', shuffle=False,
                                               indexes=test_indexes, 
                                               tore_readers=test_tore_readers,
                                               mask_readers=test_mask_readers)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        print(camel_name, )
        try:
            self.data_module = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, mode, shuffle, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.kwargs dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(mode, shuffle, **args1)
