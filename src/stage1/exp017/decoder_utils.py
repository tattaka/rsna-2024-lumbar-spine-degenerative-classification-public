import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // re, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // re, ch, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Conv2dReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        **batchnorm_params
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not (use_batchnorm),
            ),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=4,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return sinusoid_table


def get_sinusoid_encoding_table_2d(H, W, d_hid):
    """Sinusoid position encoding table"""
    n_position = H * W
    sinusoid_table = get_sinusoid_encoding_table(n_position, d_hid)
    sinusoid_table = sinusoid_table.reshape(H, W, d_hid)
    return sinusoid_table


class CBAMModule(nn.Module):
    def __init__(
        self, channels, reduction=4, attention_kernel_size=3, position_encode=False
    ):
        super(CBAMModule, self).__init__()
        self.position_encode = position_encode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        if self.position_encode:
            k = 3
        else:
            k = 2
        self.conv_after_concat = nn.Conv2d(
            k,
            1,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=attention_kernel_size // 2,
        )
        self.sigmoid_spatial = nn.Sigmoid()
        self.position_encoded = None

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        if self.position_encode:
            if self.position_encoded is None:
                pos_enc = get_sinusoid_encoding_table(h, w)
                pos_enc = Variable(torch.FloatTensor(pos_enc), requires_grad=False)
                if x.is_cuda:
                    pos_enc = pos_enc.cuda()
                self.position_encoded = pos_enc
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        if self.position_encode:
            pos_enc = self.position_encoded
            pos_enc = pos_enc.view(1, 1, h, w).repeat(b, 1, 1, 1)
            x = torch.cat((avg, mx, pos_enc), 1)
        else:
            x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class CenterBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(
                in_channels, out_channels, kernel_size=1, use_batchnorm=use_batchnorm
            ),
            Conv2dReLU(
                out_channels, out_channels, kernel_size=1, use_batchnorm=use_batchnorm
            ),
        )

    def forward(self, x):
        return self.block(x)


class FPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Feature Pyramid Attention
        https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch/blob/master/networks.py
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(in_channels / 4)

        self.channels_cond = in_channels

        # Master branch
        self.conv_master = nn.Conv2d(
            self.channels_cond, out_channels, kernel_size=1, bias=False
        )
        self.bn_master = nn.BatchNorm2d(out_channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(
            self.channels_cond, out_channels, kernel_size=1, bias=False
        )
        self.bn_gpb = nn.BatchNorm2d(out_channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(
            self.channels_cond,
            channels_mid,
            kernel_size=(7, 7),
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(
            channels_mid,
            channels_mid,
            kernel_size=(5, 5),
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(
            channels_mid,
            channels_mid,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(
            channels_mid,
            out_channels,
            kernel_size=(7, 7),
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        self.conv5x5_2 = nn.Conv2d(
            channels_mid,
            out_channels,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn2_2 = nn.BatchNorm2d(out_channels)
        self.conv3x3_2 = nn.Conv2d(
            channels_mid,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Master branch
        h, w = x.size(2), x.size(3)
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x1_2 = self.relu(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.relu(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_2 = self.relu(x3_2)

        # Merge branch 1 and
        x3_upsample = nn.Upsample(size=(h // 4, w // 4), mode="nearest")(x3_2)
        x2_merge = x2_2 + x3_upsample
        x2_upsample = nn.Upsample(size=(h // 2, w // 2), mode="nearest")(x2_merge)
        x1_merge = x1_2 + x2_upsample
        x_master = x_master * nn.Upsample(size=(h, w), mode="nearest")(x1_merge)

        out = x_master + x_gpb

        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        planes = int(planes)
        self.atrous_conv = nn.Conv2d(
            inplanes,
            int(planes),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(
            inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1]
        )
        self.aspp3 = _ASPPModule(
            inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2]
        )
        self.aspp4 = _ASPPModule(
            inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3]
        )
        mid_c = int(mid_c)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="nearest")
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, use_batchnorm=True, attention_type=None
    ):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == "scse":
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)
        elif attention_type == "cbam":
            self.attention1 = CBAMModule(in_channels)
            self.attention2 = CBAMModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
        )

    def forward(self, x):
        x, skip = x[:-1], x[-1]
        if skip is not None:
            x = [
                F.interpolate(xi, size=skip.shape[-2:], mode="nearest")
                if skip.shape[-1] > xi.shape[-1]
                else F.max_pool2d(
                    xi, math.ceil(xi.shape[-1] / skip.shape[-1]), ceil_mode=True
                )
                for xi in x
            ]
            x.append(skip)
            x = torch.cat(x, dim=1)
            x = self.attention1(x)
        else:
            x = [F.interpolate(xi, scale_factor=2, mode="nearest") for xi in x]
            x = torch.cat(x, dim=1)
        x = self.block(x)
        x = self.attention2(x)
        return x


class UNetHead(nn.Module):
    __name__ = "UNetHead"

    def __init__(
        self,
        encoder_channels,
        decoder_channels=[1024, 512, 256, 128, 64],
        num_class=1,
        use_batchnorm=True,
        center=None,
        attention_type=None,
        classification=False,
        deep_supervision=False,
    ):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        decoder_channels = decoder_channels[: len(encoder_channels)]
        if center == "fpa":
            self.center = FPA(encoder_channels[0], decoder_channels[0])
        elif center == "aspp":
            self.center = ASPP(
                encoder_channels[0],
                decoder_channels[0],
                dilations=[1, (1, 6), (2, 12), (3, 18)],
            )
        else:
            self.center = CenterBlock(
                encoder_channels[0], decoder_channels[0], use_batchnorm=use_batchnorm
            )
        in_channels = self.compute_channels(encoder_channels[1:], decoder_channels[:-1])
        layers = []
        for i in range(len(in_channels)):
            layers.append(
                DecoderBlock(
                    in_channels[i],
                    decoder_channels[i + 1],
                    use_batchnorm=use_batchnorm,
                    attention_type=attention_type,
                )
            )
        for i in range(5 - len(decoder_channels)):
            layers.append(
                DecoderBlock(
                    decoder_channels[-1],
                    decoder_channels[-1],
                    use_batchnorm=use_batchnorm,
                    attention_type=attention_type,
                )
            )
        layers.append(
            DecoderBlock(
                decoder_channels[-1],
                decoder_channels[-1],
                use_batchnorm=use_batchnorm,
                attention_type=attention_type,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_class, kernel_size=(1, 1))

        self.classification = classification
        if self.classification:
            self.linear_feature = nn.Sequential(
                nn.Conv2d(encoder_channels[0], 512, kernel_size=1),
                AdaptiveConcatPool2d(1),
                Flatten(),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_class),
            )
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            layers_ds = []
            for i in range(len(decoder_channels)):
                layers_ds.append(
                    nn.Conv2d(decoder_channels[i], num_class, kernel_size=(1, 1))
                )
            layers_ds.append(
                nn.Conv2d(decoder_channels[i + 1], num_class, kernel_size=(1, 1))
            )
            self.layers_ds = nn.ModuleList(layers_ds)

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [e + d for e, d in zip(encoder_channels, decoder_channels)]
        return channels

    def forward(self, x):
        x = x[::-1]
        encoder_head = x[0]
        skips = x[1:]
        x_o = []
        x_o.append(self.center(encoder_head))
        for i in range(len(self.layers)):
            if i < len(skips):
                skip = skips[i]
            else:
                skip = None
            x_o.append(self.layers[i]([x_o[-1], skip]))
        x_final = self.final_conv(x_o[-1])
        output = [x_final]
        if self.classification:
            class_logits = self.linear_feature(encoder_head)
            output.append(class_logits)
        if self.deep_supervision:
            x_ds = []
            for i in range(len(self.layers_ds)):
                x_ds.append(self.layers_ds[i](x_o[i]))
            output.append(x_ds)
        return output[0] if len(output) == 1 else output


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        BatchNorm=nn.BatchNorm2d,
    ):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=inplanes,
            bias=bias,
        )
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[0], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.dilation1 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.dilation4 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=8, dilation=8, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def forward(self, *inputs):
        feats = [self.conv5(inputs[0]), self.conv4(inputs[1]), self.conv3(inputs[2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], size=(h, w), mode="nearest")
        feats[-3] = F.interpolate(feats[-3], size=(h, w), mode="nearest")
        feat = torch.cat(feats, dim=1)
        feat = torch.cat(
            [
                self.dilation1(feat),
                self.dilation2(feat),
                self.dilation3(feat),
                self.dilation4(feat),
            ],
            dim=1,
        )
        return feat


class FastFCNImproveHead(nn.Module):
    __name__ = "FastFCNImproveHead"

    def __init__(
        self,
        encoder_channels,
        decoder_channels=(256, 128, 64),
        num_class=1,
        use_batchnorm=True,
        attention_type=None,
        classification=False,
        deep_supervision=False,
    ):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        self.jpu = JPU(
            [encoder_channels[0], encoder_channels[1], encoder_channels[2]],
            decoder_channels[0],
        )
        self.aspp = ASPP(
            decoder_channels[0] * 4,
            decoder_channels[0],
            dilations=[1, (1, 4), (2, 8), (3, 12)],
        )
        self.decoder1 = DecoderBlock(
            encoder_channels[3] + decoder_channels[0],
            decoder_channels[1],
            use_batchnorm,
            attention_type,
        )
        self.decoder2 = DecoderBlock(
            encoder_channels[4] + decoder_channels[1],
            decoder_channels[2],
            use_batchnorm,
            attention_type,
        )
        self.decoder3 = DecoderBlock(decoder_channels[2], decoder_channels[2])
        self.final_conv = nn.Conv2d(decoder_channels[2], num_class, kernel_size=(1, 1))

        self.classification = classification
        if self.classification:
            self.linear_feature = nn.Sequential(
                nn.Conv2d(encoder_channels[0], 512, kernel_size=1),
                AdaptiveConcatPool2d(1),
                Flatten(),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_class),
            )
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.layer0_ds = nn.Conv2d(
                decoder_channels[0], num_class, kernel_size=(1, 1)
            )
            self.layer1_ds = nn.Conv2d(
                decoder_channels[1], num_class, kernel_size=(1, 1)
            )
            self.layer2_ds = nn.Conv2d(
                decoder_channels[2], num_class, kernel_size=(1, 1)
            )

    def forward(self, x):
        x = x[::-1]
        skips = x
        x_0 = self.jpu(skips[0], skips[1], skips[2])
        x_0 = self.aspp(x_0)
        x_1 = self.decoder1([x_0, skips[3]])
        x_2 = self.decoder2([x_1, skips[4]])
        x_3 = self.decoder3([x_2, None])
        x_final = self.final_conv(x_3)
        output = [x_final]
        if self.classification:
            class_refine = self.linear_feature(x[0])
            output.append(class_refine)
        if self.deep_supervision:
            x_ds = []
            x_ds.append(self.layer0_ds(x_0))
            x_ds.append(self.layer1_ds(x_1))
            x_ds.append(self.layer2_ds(x_2))
            output.append(x_ds)
        return output[0] if len(output) == 1 else output