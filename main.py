from argparse import ArgumentParser

from pytorch_lightning import Trainer

from model import U2NET
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataloader import MaskDataset
from pytorch_lightning.loggers import TensorBoardLogger


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(size_average=True)
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


class ModelInterface(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def forward(self, x):
        # use forward for inference/predictions
        mask, _ = self.model(x)
        return mask

    def training_step(self, batch, batch_idx):
        img, mask = batch
        d0, d1, d2, d3, d4, d5, d6 = self.model(img)

        _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        d0, d1, d2, d3, d4, d5, d6 = self.model(img)

        _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
        self.log('val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        d0, d1, d2, d3, d4, d5, d6 = self.model(img)

        _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
        self.log('test_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]


def main(args):
    pl.seed_everything(args.seed)

    # ------------
    # logs and checkpoint
    # ------------
    logger = TensorBoardLogger(save_dir=args.log_dir)
    args.logger = logger

    print("Logs set up complete!")

    # ------------
    # data
    # ------------
    dataset = MaskDataset('frames', 'mask', transform=transforms.ToTensor())
    train_len, test_len = int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))
    train, test = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    print("Dataloader ready!")

    # ------------
    # model
    # ------------
    model = ModelInterface(U2NET(args.in_channel, args.out_channel), args.learning_rate)

    print("Model instantiated!")

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)

    print("Training complete!")

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print("Testing complete!")
    print(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Training params
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpus', default=4, type=int)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # logger
    parser.add_argument('--log_dir', default='logs/', type=str)

    # Model hyperparams
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--out_channel', default=1, type=int)

    parser = Trainer.add_argparse_args(
        parser.add_argument_group(title='pl.Trainer args')
    )

    args = parser.parse_args()

    main(args)
