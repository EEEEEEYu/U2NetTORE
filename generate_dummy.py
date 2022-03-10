import os
import torch


def generate(frames_dir, mask_dir, size=1024):
    for i in range(size):
        torch.save(torch.randn([6, 320, 320]), os.path.join(frames_dir, str(i) + '.pt'))
        torch.save((torch.randn([1, 320, 320]) > 0).float(), os.path.join(mask_dir, str(i) + '.pt'))


def main():
    frames_dir = './frames'
    mask_dir = './mask'
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    generate(frames_dir, mask_dir)


if __name__ == '__main__':
    main()

