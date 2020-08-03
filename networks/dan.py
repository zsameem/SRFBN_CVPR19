from torch import nn
import torch
import pdb

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # pdb.set_trace()
        return x * self.module(x)


class SpatialAttention(nn.Module):
    def __init__(self, num_features, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        channel_mean = torch.mean(x, dim=1).unsqueeze(1)
        channel_max = torch.max(x, dim=1)[0].unsqueeze(1)
        combined = torch.cat([channel_mean, channel_max], dim=1)
        spatial_attention = self.module(combined)
        return x * spatial_attention


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        # pdb.set_trace()
        return x + self.module(x)


class RDAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RDAB, self).__init__()
        self.conv_module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.ca_module = ChannelAttention(num_features, reduction)
        self.sa_module = SpatialAttention(num_features, kernel_size=3)
        self.compress = nn.Conv2d(num_features*2, num_features, kernel_size=1)

    def forward(self, x):
        # pdb.set_trace()
        conv_out = self.conv_module(x)
        ca_out = self.ca_module(conv_out)
        sa_out = self.sa_module(conv_out)
        da_out = self.compress(torch.cat([ca_out, sa_out], dim=1))
        return x + da_out


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RDAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        # pdb.set_trace()
        return x + self.module(x)

#     def __init__(self, args_n_colors,  args_n_feats, args_n_resgroups, 
#                   args_n_resblocks, args_scale,  args_rgb_range, args_reduction=16,  conv=common.default_conv):

class RDAN(nn.Module):
    def __init__(self, args_num_features, args_num_rg, args_num_rcab, args_scale, args_reduction=16):
        super(RDAN, self).__init__()
        scale = args_scale
        num_features = args_num_features
        num_rg = args_num_rg
        num_rcab = args_num_rcab
        reduction = args_reduction

        self.sf = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sf(x)
        residual = x
        # pdb.set_trace()
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        return x