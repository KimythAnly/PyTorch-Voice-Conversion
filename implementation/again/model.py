import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x, mask=None):
        mn = reduce(x, 'b c t -> b c', 'mean')
        sd = (reduce(x, 'b c t -> b c', torch.var) + self.eps).sqrt()
        mn = rearrange(mn, 'b c -> b c 1')
        sd = rearrange(sd, 'b c -> b c 1')
        return mn, sd

    def forward(self, x, return_mean_std=False):
        """
        :param x: has either shape (b, c, x, y) or shape (b, c, x, y, z)
        :return:
        """
        mean, std = self.calc_mean_std(x)
        x = (x - mean) / std
        if return_mean_std:
            return x, mean, std
        else:
            return x


class ConvNorm1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        padding_mode='zeros',
        dilation=1,
        groups=1,
        bias=True,
        w_init_gain='linear',
    ):
        super().__init__()

        if padding is None:
            assert(dilation * (kernel_size-1) % 2 == 0)
            padding = int(dilation * (kernel_size-1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, padding_mode=padding_mode,
                                    dilation=dilation, groups=groups, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvNorm2d(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True,
            w_init_gain='linear',
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if padding is None:
            padding = []
            for k, d in zip(kernel_size, dilation):
                assert((d * k-1) % 2 == 0)
                p = int(d * (k-1) / 2)
                padding.append(p)
            padding = tuple(padding)

        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, padding_mode=padding_mode,
                                    dilation=dilation, groups=groups, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class EncConvBlock(nn.Module):
    def __init__(self, c_in, c_h, subsample=1):
        super().__init__()

        self.seq = nn.Sequential(
            ConvNorm1d(c_in, c_h, kernel_size=3, stride=1),
            nn.BatchNorm1d(c_h),
            nn.LeakyReLU(),
            ConvNorm1d(c_h, c_in, kernel_size=3, stride=subsample),
        )
        self.subsample = subsample

    def forward(self, x):
        y = self.seq(x)
        if self.subsample > 1:
            x = F.avg_pool1d(x, kernel_size=self.subsample)
        return x + y


class DecConvBlock(nn.Module):
    def __init__(self, c_in, c_h, c_out, upsample=1):
        super().__init__()

        self.dec_block = nn.Sequential(
            ConvNorm1d(c_in, c_h, kernel_size=3, stride=1),
            nn.BatchNorm1d(c_h),
            nn.LeakyReLU(),
            ConvNorm1d(c_h, c_in, kernel_size=3),
        )
        self.gen_block = nn.Sequential(
            ConvNorm1d(c_in, c_h, kernel_size=3, stride=1),
            nn.BatchNorm1d(c_h),
            nn.LeakyReLU(),
            ConvNorm1d(c_h, c_in, kernel_size=3),
        )
        self.upsample = upsample

    def forward(self, x):
        y = self.dec_block(x)
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample)
            y = F.interpolate(y, scale_factor=self.upsample)
        y = y + self.gen_block(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        n_conv_blocks,
        c_h,
        subsample
    ):
        super().__init__()

        self.inorm = InstanceNorm()

        # 2d Conv blocks
        self.conv2d_blocks = nn.Sequential(
            ConvNorm2d(1, 8),
            ConvNorm2d(8, 8),
            ConvNorm2d(8, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )

        # 1d Conv blocks
        self.conv1d_first = ConvNorm1d(c_in, c_h)
        self.conv1d_blocks = nn.ModuleList([])
        for _, sub in zip(range(n_conv_blocks), subsample):
            self.conv1d_blocks.append(
                EncConvBlock(c_h, c_h, subsample=sub)
            )

        self.out_layer = ConvNorm1d(c_h, c_out)

    def forward(self, x):
        y = x
        # y = self.conv2d_blocks(y)
        y = y.squeeze(1)
        y = self.conv1d_first(y)

        mns = []
        sds = []

        for block in self.conv1d_blocks:
            y = block(y)
            y, mn, sd = self.inorm(y, return_mean_std=True)
            mns.append(mn)
            sds.append(sd)

        y = self.out_layer(y)

        return y, mns, sds


class Decoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_h,
        c_out,
        n_conv_blocks,
        upsample,
    ):
        super().__init__()

        self.act = nn.LeakyReLU()
        self.inorm = InstanceNorm()

        self.in_layer = ConvNorm1d(c_in, c_h, kernel_size=3)

        self.conv_blocks = nn.ModuleList([])
        for _, up in zip(range(n_conv_blocks), upsample):
            self.conv_blocks.append(
                DecConvBlock(c_h, c_h, c_h, upsample=up)
            )

        self.last_cnn = nn.Sequential(
            ConvNorm1d(c_h, c_h),
            ConvNorm1d(c_h, c_h),
        )

        self.out_layer = nn.Linear(c_h, c_out)

    def forward(self, enc, cond, return_c=False, return_s=False):
        y1, _, _ = enc
        y2, mns, sds = cond
        mn, sd = self.inorm.calc_mean_std(y2)

        c = self.inorm(y1)
        c_affine = c*sd + mn

        y = self.in_layer(c_affine)
        y = self.act(y)

        for i, (block, mn, sd) in enumerate(zip(self.conv_blocks, mns, sds)):
            y = block(y)
            y = self.inorm(y)
            y = y * sd + mn

        y = self.last_cnn(y)
        y = y.transpose(1, 2)

        y = self.out_layer(y)
        y = y.transpose(1, 2)

        if return_c:
            return y, c
        elif return_s:
            mn = torch.cat(mns, -2)
            sd = torch.cat(sds, -2)
            s = mn * sd
            return y, s
        else:
            return y


class VariantSigmoid(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        y = 1 / (1+torch.exp(-self.alpha*x))
        return y


class Activation(nn.Module):
    dct = {
        'none': lambda x: x,
        'sigmoid': VariantSigmoid,
        'tanh': nn.Tanh
    }

    def __init__(self, act, params=None):
        super().__init__()
        if params:
            self.act = Activation.dct[act](**params)
        else:
            self.act = Activation.dct[act]

    def forward(self, x):
        return self.act(x)
