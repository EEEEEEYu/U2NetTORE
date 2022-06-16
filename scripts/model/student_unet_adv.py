""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Down, Up, OutConv, _regular_block

class StudentUnetAdv(nn.Module):
    def __init__(self, in_ch=8, seq_len=16, bilinear=False, base_dim=16):
        super(StudentUnetAdv, self).__init__()
        self.n_channels = in_ch
        self.out_ch = seq_len
        self.bilinear = bilinear

        # base_dim = 16 # 64
        self.inc = DoubleConv(in_ch, base_dim)
        self.down1 = Down(base_dim, base_dim*2)
        self.down2 = Down(base_dim*2, base_dim*4)
        self.down3 = Down(base_dim*4, base_dim*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_dim*8, base_dim*16 // factor)
        self.up1 = Up(base_dim*16, base_dim*8 // factor, bilinear)
        self.up2 = Up(base_dim*8, base_dim*4 // factor, bilinear)
        self.up3 = Up(base_dim*4, base_dim*2 // factor, bilinear)
        self.up4 = Up(base_dim*2, base_dim, bilinear)
        self.outc = OutConv(base_dim, self.out_ch)

        self.score_branch_mid  = nn.Sequential(*[
            _regular_block(base_dim*16, base_dim*8),
            _regular_block(base_dim*8, base_dim*4),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        self.score_branch_tail = nn.Sequential(*[
            _regular_block(base_dim, base_dim*2),
            nn.Conv2d(base_dim*2, base_dim*4, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        self.fc = nn.Linear(base_dim*8 , self.out_ch)

        print(f"Output channel number: {self.out_ch}")

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        masks = self.outc(x)

        score_mid = self.score_branch_mid(x5)
        score_tail = self.score_branch_tail(x)
        score_feature = torch.concat((score_mid, score_tail), dim=1)
        scores = self.fc(torch.flatten(score_feature, 1))
        scores = torch.sigmoid(scores)

        # x = self.score_conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # scores = torch.sigmoid(self.fc(x))
        return masks.squeeze(), scores

    def predict_mask(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        masks = self.outc(x).squeeze()
        # masks, scores = self.forward(x)
        masks = torch.sigmoid(masks)[:,0:1]
        return masks