import sys
import glob
import torch
import argparse
import os.path as op

sys.path.append('..')
from model.unet_interface import ModelInteface

def get_model_path(date, root='/mnt/nfs/scratch1/zhongyangzha/DVS_HPE/dvs-hpe-light/lightning_logs', best=True):
    model_subdir = op.join(root, date, 'checkpoints')
    if best:
        model_path = glob.glob(op.join(model_subdir, 'best*'))[0]
    else:
        model_path = op.join(model_subdir, 'last.ckpt')
    return model_path

def main(args):
    model_ori_path = get_model_path(args.datetime, root=args.root, best=args.best) #05-19-18-28-41
    model_new_path = model_ori_path.replace('.ckpt', '.pt')

    model_ori = ModelInteface.load_from_checkpoint(model_ori_path)
    print('[√] Original model loaded.')
    torch.save(model_ori.model, model_new_path)
    print('[√] Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='/mnt/nfs/scratch1/zhongyangzha/DVS_HPE/U2NetTORE/lightning_logs/', help='root folder of logs.')
    parser.add_argument('-d', '--datetime', type=str, default='', help='date time string for the folder going to be processed.')
    parser.add_argument('-b', '--best', action='store_true', help='Use the best model.')
    args = parser.parse_args()

    main(args)
    