import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
from torch.autograd import Variable

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

# Encoding block in U-Net
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.down = nn.MaxPool3d(2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        return y, y_conv

# Decoding block in U-Net
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        return y, y_conv

def concatenate(x1, x2):
    # input is CHWD
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    diffZ = x2.size()[4] - x1.size()[4]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2,
                    diffZ // 2, diffZ - diffZ//2))        
    y = torch.cat([x2, x1], dim=1)
    return y

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class outsoftmax(nn.Module):
    def __init__(self):
        super(outsoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(x)
        return x

class att_module(nn.Module):
    def __init__(self, in1_ch, in2_ch, out_ch, downsample):
        super(att_module, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv3d(in_channels=in1_ch + in2_ch, out_channels=in1_ch + in2_ch, kernel_size=1, padding=0),
            nn.BatchNorm3d(in1_ch + in2_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in1_ch + in2_ch, out_channels=in1_ch + in2_ch, kernel_size=1, padding=0),
            nn.BatchNorm3d(in1_ch + in2_ch),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in1_ch + in2_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )
        self.downsample = downsample
        if self.downsample:
            self.resample = nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.resample = nn.ConvTranspose3d(in_channels=out_ch, out_channels=out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        y = torch.cat([x1, x2], dim=1)
        att_mask = self.att_conv(y)
        y = att_mask * y
        y = self.conv(y)
        y = self.resample(y)
        return y

class AMTA_Net3D_A(nn.Module):
    def __init__(self, in_ch = 1):
        super(AMTA_Net3D_A, self).__init__()
        self.in_ch = in_ch
        
        self.enc1 = enc_block(in_ch, 64)
        self.enc2 = enc_block(64, 128)
        self.enc3 = enc_block(128, 256)
        self.enc4 = enc_block(256, 512)
        self.dec1 = dec_block(512, 512, bilinear=False)
        self.dec2 = dec_block(1024, 256, bilinear=False)
        self.dec3 = dec_block(512, 128, bilinear=False)
        self.dec4 = dec_block(256, 64, bilinear=False)
        self.outconv = double_conv(128, 64)

        self.enc1_att = att_module(64, 64, 128, downsample=True)
        self.enc2_att = att_module(128, 128, 256, downsample=True)
        self.enc3_att = att_module(256, 256, 512, downsample=True)
        self.enc4_att = att_module(512, 512, 512, downsample=True)
        self.dec1_att = att_module(512, 512, 256, downsample=False)
        self.dec2_att = att_module(256, 256, 128, downsample=False)
        self.dec3_att = att_module(128, 128, 64, downsample=False)
        self.dec4_att = att_module(64, 64, 64, downsample=False)
        self.outconv_att = double_conv(64, 64)

        self.oar_outc = outconv(64, 4)
        self.oar_outs = outsoftmax()
        self.pb_outc = outconv(64, 2)
        self.pb_outs = outsoftmax()

    def forward(self, x):

        enc1, enc1_conv = self.enc1(x)
        enc2, enc2_conv = self.enc2(enc1)
        enc3, enc3_conv = self.enc3(enc2)
        enc4, enc4_conv = self.enc4(enc3)
        dec1, dec1_conv = self.dec1(enc4)
#        print(dec1.shape,enc4_conv.shape)
        dec2, dec2_conv = self.dec2(concatenate(dec1, enc4_conv))
        dec3, dec3_conv = self.dec3(concatenate(dec2, enc3_conv))
        dec4, dec4_conv = self.dec4(concatenate(dec3, enc2_conv))
        dec_out = self.outconv(concatenate(dec4, enc1_conv))

        enc1_att = self.enc1_att(dec_out, enc1_conv)
        enc2_att = self.enc2_att(enc1_att, enc2_conv)
        enc3_att = self.enc3_att(enc2_att, enc3_conv)
        enc4_att = self.enc4_att(enc3_att, enc4_conv)
        dec1_att = self.dec1_att(enc4_att, dec1_conv)
        dec2_att = self.dec2_att(dec1_att, dec2_conv)
        dec3_att = self.dec3_att(dec2_att, dec3_conv)
        dec4_att = self.dec4_att(dec3_att, dec4_conv)
        att_out = self.outconv_att(dec4_att)

        oar = self.oar_outc(dec_out)
        pb = self.pb_outc(att_out)
        oar = self.oar_outs(oar)
        pb = self.pb_outs(pb)
        return pb , oar

    def name(self):
        return 'Asymmetric Multi-Task Attention Network (Input channel = {0:d})'.format(self.in_ch)
