import pytorch_lightning as pl
from argparse import ArgumentParser

import torch.cuda
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from model import ModelInteface
from data import DataInterface
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DataInterface(**vars(args))

    if load_path is None:
        model = ModelInteface(**vars(args))
    else:
        model = ModelInteface(**vars(args))
        args.resume_from_checkpoint = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    # args.callbacks = load_callbacks()
    # args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpus', default=4 if torch.cuda.is_available() else 0, type=int)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='mask_dataset', type=str)
    parser.add_argument('--data_dir', default='ref/data', type=str)
    parser.add_argument('--model_name', default='u2net', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()

    main(args)
