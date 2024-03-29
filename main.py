import os
import pytorch_lightning as pl
from argparse import ArgumentParser

import torch.cuda
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.model import ModelInteface
from scripts.data import DataInterface
from scripts.utils import load_model_path_by_args, SBool, build_working_tree, get_gpu_num


def load_callbacks(checkpoint_dir=None):
    callbacks = []
    
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10
    ))

    callbacks.append(plc.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
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
    logger_dir, checkpoint_dir, recorder_dir, log_profiler = build_working_tree(name='')

    print(os.getcwd())
    load_path = load_model_path_by_args(args)
    data_module = DataInterface(**vars(args))

    args.callbacks = load_callbacks(checkpoint_dir=checkpoint_dir)

    if load_path is None:
        model = ModelInteface(**vars(args))
    else:
        if args.load_weights_only:
            model = ModelInteface.load_from_checkpoint(load_path, **vars(args))
        else:
            model = ModelInteface(**vars(args))
            args.resume_from_checkpoint = load_path

    if args.use_profiler:
        # log_profiler = os.path.join(os.getcwd(), "profile.txt")
        profiler = pl.profiler.AdvancedProfiler(log_profiler)
        args.profiler = profiler

    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=logger_dir, name='')
    args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

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
    parser.add_argument('--load_weights_only', type=SBool, default=False, nargs='?', const=True)

    # Training Info
    parser.add_argument('--dataset', default='mask_dataset', type=str)
    parser.add_argument('--data_dir', default='ref/data', type=str)
    parser.add_argument('--model_name', default='unet', type=str)
    parser.add_argument('--loss', default='bce', type=str, choices=('bce', 'mbce'))
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument("--use_profiler", type=SBool, default=False, nargs='?', const=True)

    # Model Info
    parser.add_argument("--use_convlstm", type=SBool, default=True, nargs='?', const=True)
    parser.add_argument("--use_dilated_conv", type=SBool, default=True, nargs='?', const=True)
    parser.add_argument("--bilinear", type=SBool, default=False, nargs='?', const=True, help='Whether to use bilinear upsampling or transposed conv in unet.')
    parser.add_argument("--out_ch", type=int, default=16, nargs='?', const=True, help='Output channel number for unet.')
    parser.add_argument("--base_dim", type=int, default=16, nargs='?', const=True, help='The base dimension number for unet.')
    parser.add_argument("--separate_punish", type=SBool, default=False, nargs='?', const=True, help='Whether to calculate the loss for 0/1 pixels in mask GT separately.')
    parser.add_argument("--score_order_punish", type=SBool, default=False, nargs='?', const=True, help='Whether to add a loss to guarantee the predicted mask score order.')
    parser.add_argument("--add_fb_loss", type=SBool, default=False, nargs='?', const=True, help='Whether to add a first layer mask(corresponding to the input frame) loss.')
    parser.add_argument("--time_weighted", type=SBool, default=False, nargs='?', const=True, help='Whether to add a first layer mask(corresponding to the input frame) loss.')
    parser.add_argument('--teacher_path', default=None, type=str, help="Pretrained teacher network checkpoint path.")
    parser.add_argument('--teacher_alpha', default=0.5, type=float, help="The weights from teacher network loss.")
    parser.add_argument('--loss_alpha', default=1, type=float, help='The parameter multiplied on score loss.')

    # Data Info
    # parser.add_argument("--img_dir", default='dummy_data/frames', type=str)
    parser.add_argument("--mask_dir", default="", type=str)
    parser.add_argument("--tore_dir", default="", type=str)
    parser.add_argument("--accumulated", type=SBool, default=True, nargs='?', const=True)
    parser.add_argument("--loop_read", type=SBool, default=False, nargs='?', const=True)
    parser.add_argument("--acc_time", default=0.02, type=float, help='Accumulation time for mask readers.')
    parser.add_argument("--step_size", default=0.02, type=float, help='Step size for mask readers.')
    parser.add_argument("--cache_size", default=1, type=int)
    parser.add_argument("--seq_len", default=16, type=int, help='The sequence length used in the convlstm.')
    parser.add_argument('--base_number', default=16, type=int, help="The base number of each meta file data piece number.")
    parser.add_argument('--test_characters', type=str, nargs='*', default=['Diluc', 'Xiangling'], help="Characters whos are only used in the test session.")
    parser.add_argument('--percentile', default=90, type=float, help="The percentile used to generate the extra band in ntore.")
    parser.add_argument('--ori_tore', type=SBool, default=True, nargs='?', const=True, help="Use the original TORE instead, for comparison usage.")
    parser.add_argument('--cycle_views', type=SBool, default=False, nargs='?', const=True, help="Cycle the camera views. If applied, the total data size will be cut to 1/4.")
    parser.add_argument('--rand_test', type=SBool, default=False, nargs='?', const=True, help="Use randomly selected test cases.")
    parser.add_argument('--partial_dataset', default=1.0, type=float, help="The percentage of data that is going to be use in training and validation.")
    
    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # TORE Loader Setting

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    parser.set_defaults(strategy='dp')
    parser.set_defaults(find_unused_parameters=False)
    parser.set_defaults(gpus=4 if torch.cuda.is_available() else 0)

    args = parser.parse_args()
    args.gpus = get_gpu_num(args.gpus)

    if args.ori_tore:
        args.in_ch=6

    # args.batch_size = args.batch_size // args.gpus
    # print(args.batch_size)
    # print(args.loop_read, args.use_convlstm, args.shuffle)

    main(args)
