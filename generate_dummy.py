import os
import torch


def generate(frames_dir, mask_dir, size=128):
    for i in range(size):
        torch.save(torch.randn([6, 320, 320]), os.path.join(frames_dir, str(i) + '.pt'))
        torch.save((torch.randn([1, 320, 320]) > 0).float(), os.path.join(mask_dir, str(i) + '.pt'))


def main():
    frames_dir = './frames'
    mask_dir = './mask'
    generate(frames_dir, mask_dir)


if __name__ == '__main__':
    main()
