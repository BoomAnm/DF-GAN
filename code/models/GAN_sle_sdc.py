import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils import spectral_norm


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size, lstm=None):
        super(NetG, self).__init__()
        nfc_multi = {4: 8, 8: 8, 16: 8, 32: 8, 64: 4, 128: 2, 256: 1, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.ngf = ngf
        self.lstm = lstm
        # input noise (batch_size, 100)
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        # build GBlocks
        # self.GBlocks = nn.ModuleList([])
        # in_out_pairs = get_G_in_out_chs(ngf, imsize)
        # for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
        #     self.GBlocks.append(G_Block(cond_dim + nz, in_ch, out_ch, upsample=True))
        self.block0 = G_Block(cond_dim, ngf * 8, ngf * 8, lstm, upsample=True)  # 4x4
        self.block1 = G_Block(cond_dim, ngf * 8, ngf * 8, lstm, upsample=True)  # 8x8
        self.block2 = G_Block(cond_dim, ngf * 8, ngf * 8, lstm, upsample=True)  # 16x16
        self.block3 = G_Block(cond_dim, ngf * 8, ngf * 8, lstm, upsample=True)  # 32x32
        self.block4 = G_Block(cond_dim, ngf * 8, ngf * 4, lstm, upsample=True)  # 64x64
        self.block5 = G_Block(cond_dim, ngf * 4, ngf * 2, lstm, upsample=True)  # 128x128
        self.block6 = G_Block(cond_dim, ngf * 2, ngf * 1, lstm, upsample=True)  # 256x256

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ch_size, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, c):  # x=noise, c=ent_emb
        # concat noise and sentence
        out = self.fc(noise)
        out = out.view(noise.size(0), 8 * self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        # for GBlock in self.GBlocks:
        #     out = GBlock(out, cond)
        out_4 = self.block0(out, c)
        out_8 = self.block1(out_4, c)
        out_16 = self.block2(out_8, c)
        out_32 = self.block3(out_16, c)
        out_64 = self.se_64(out_4, self.block4(out_32, c))
        out_128 = self.se_128(out_8, self.block5(out_64, c))
        out_256 = self.se_256(out_16, self.block6(out_128, c))

        # convert to RGB image
        out = self.to_rgb(out_256)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)  # 128
        # build DBlocks
        # self.DBlocks = nn.ModuleList([])
        # in_out_pairs = get_D_in_out_chs(ndf, imsize)
        # for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
        #     self.DBlocks.append(D_Block(in_ch, out_ch))
        self.block0 = D_Block(ndf * 1, ndf * 2)  # 64
        self.block1 = D_Block(ndf * 2, ndf * 4)  # 32
        self.block2 = D_Block(ndf * 4, ndf * 8)  # 16
        self.block3 = D_Block(ndf * 8, ndf * 16)  # 8
        self.block4 = D_Block(ndf * 16, ndf * 16)  # 4
        self.block5 = D_Block(ndf * 16, ndf * 16)  # 4

        self.se_4_32 = SEBlock(nfc[128], nfc[16])
        self.se_8_64 = SEBlock(nfc[64], nfc[8])

        self.down_from_small = nn.Sequential(
            conv2d(ch_size, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]), )

        self.decoder_big = SimpleDecoder(nfc[4], ch_size)
        self.decoder_part = SimpleDecoder(nfc[8], ch_size)
        self.decoder_small = SimpleDecoder(nfc[32], ch_size)

    def forward(self, x, label, part=None):
        # out = self.conv_img(x)
        # for DBlock in self.DBlocks:
        #     out = DBlock(out)
        # return out
        if type(x) is not list:
            x = [F.interpolate(x, size=256), F.interpolate(x, size=128)]
        out_1 = self.conv_img(x[0])
        out_2 = self.block0(out_1)
        out_3 = self.block1(out_2)

        out_4 = self.block2(out_3)
        out_4 = self.se_4_32(out_1, out_4)

        out_5 = self.block3(out_4)
        out_5 = self.se_8_64(out_2, out_5)

        out_6 = self.block4(out_5)

        feat_small = self.down_from_small(x[1])

        if label == 'real':
            rec_img_big = self.decoder_big(out_6)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(out_5[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(out_5[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(out_5[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(out_5[:, :, 8:, 8:])

            return out_6, [rec_img_big, rec_img_small, rec_img_part]

        return out_6


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes * 2), GLU())
            return block

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                  upBlock(nfc_in, nfc[16]),
                                  upBlock(nfc[16], nfc[32]),
                                  upBlock(nfc[32], nfc[64]),
                                  upBlock(nfc[64], nfc[128]),
                                  conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                  nn.Tanh())

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


class NetC(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 8 + cond_dim, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

        self.block = D_Block(ndf * 16 + 256, ndf * 16)  # 4

        self.joint_conv_att = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(2)

    def forward(self, out, y_):
        y = y_.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        p = self.joint_conv_att(h_c_code)
        p = self.softmax(p.view(-1, 1, 64))
        p = p.reshape(-1, 1, 8, 8)
        self.p = p
        p = p.repeat(1, 256, 1, 1)
        y = torch.mul(y, p)
        h_c_code = torch.cat((out, y), 1)
        h_c_code = self.block(h_c_code)

        y = y_.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((h_c_code, y), 1)
        out = self.joint_conv(h_c_code)
        return out


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, lstm, upsample):
        super(G_Block, self).__init__()
        self.lstm = lstm
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch, lstm)
        self.fuse2 = DFBLK(cond_dim, out_ch, lstm)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):

        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, x, y):
        if self.upsample == True:
            x = F.interpolate(x, scale_factor=2)
        return self.shortcut(x) + self.residual(x, y)


class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
            # return x + res
        return x + self.gamma * res


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch, lstm):
        super(DFBLK, self).__init__()
        self.lstm = lstm
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, yy=None):
        lstm_input = yy
        y, _ = self.lstm(lstm_input)
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)

        lstm_input = yy
        y, _ = self.lstm(lstm_input)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize)) - 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize)) - 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs
