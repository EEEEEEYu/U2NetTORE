""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .convlstm import BiConvLSTM

class Unet(nn.Module):
    def __init__(self, in_ch=8, out_ch=1, bilinear=False, use_convlstm=False, seq_len=16):
        super(Unet, self).__init__()
        self.n_channels = in_ch
        self.n_classes = out_ch
        self.bilinear = bilinear
        self.use_convlstm = use_convlstm
        self.seq_len = seq_len

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        if self.use_convlstm:
            self.conv_lstm = BiConvLSTM(64, 16, (3, 3), 1, batch_first=True)
            self.hid = 16
            # self.outc = OutConv(16, out_ch)
            print("Using BiConvLSTM in U2Net.")
        else:
            self.hid = 64
        self.outc = OutConv(self.hid, out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.use_convlstm:
            x = x.reshape(x.shape[0]//self.seq_len, self.seq_len, *x.shape[1:])
            # print(f'From inside: {x.shape}')
            x = self.conv_lstm(x)[0].reshape(x.shape[0]*x.shape[1], self.hid, *x.shape[3:])

        logits = self.outc(x)
        return logits.squeeze()
