import inspect
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

# def calc_bce_loss(pred, label):
#     pred = pred.to(torch.float32)
#     return (pred, label) #size_average=True)


def multi_bce_loss_fusion(ds, labels_v):
    # labels_v = labels_v.to(torch.float16)
    # print(d0.dtype, labels_v.dtype)
    d0, d1, d2, d3, d4, d5, d6 = ds
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss #loss0, loss


class ModelInteface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if 'callbacks' in self.hparams.keys():
            del self.hparams['callbacks']
        print('Model hparams saved!')
        # print(self.hparams.keys())
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        # print('From model interface 1:', img[0].shape)

        img = img.reshape((img.shape[0]* img.shape[1], *img.shape[2:]))
        # print('From model interface 2:', img[0].shape)
        labels = labels.reshape(labels.shape[0] * labels.shape[1], *list(labels.shape[2:]))
        
        ds = self(img)
        loss = self.loss_function(ds, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        # print('From model interface 1:', img[0].shape)

        img = img.reshape((img.shape[0]* img.shape[1], *img.shape[2:]))
        labels = labels.reshape(labels.shape[0] * labels.shape[1], *list(labels.shape[2:]))
        
        ds = self(img)
        loss = self.loss_function(ds, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, labels = batch
        img = img.reshape((img.shape[0]* img.shape[1], *img.shape[2:]))
        labels = labels.reshape(labels.shape[0] * labels.shape[1], *list(labels.shape[2:]))
        ds = self(img)
        return F.sigmoid(ds)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

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

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mbce':
            self.loss_function = multi_bce_loss_fusion
        elif loss == 'bce':
            self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise KeyError("Invalid Loss Choice!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)