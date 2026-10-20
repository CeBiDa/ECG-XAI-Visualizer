"""
Cardiac-Ischemia-XNet architecture

Residual 1-D CNN for binary ischemia classification of 12-lead ECGs: a
dilated multi-scale stem, bottleneck residual blocks with squeeze-and-
excitation and stochastic depth (DropPath, inactive at inference), global
average pooling, and a dropout + linear head that also takes age and sex.
The shipped weights use num_layers=3 and width_scale=1.0.
Attribute names must stay as they are: the checkpoint in models/ is a state_dict keyed by them.
"""
import torch
import torch.nn as nn


# Architecture
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7), dilation_rate=2):
        super().__init__()
        self.convs = nn.ModuleList()
        total_out = 0
        for k in kernel_sizes:
            pad = ((k - 1) // 2) * dilation_rate
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=k,
                          padding=pad, dilation=dilation_rate, bias=False)
            )
            total_out += out_channels
        self.bn = nn.BatchNorm1d(total_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.relu(self.bn(x))

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)

class BottleneckSEBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 reduction=16, activation=nn.ReLU(inplace=True), drop_path=0.0):
        super().__init__()
        bot = out_channels // self.expansion
        self.conv1 = nn.Conv1d(in_channels, bot, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bot)
        self.conv2 = nn.Conv1d(bot, bot, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bot)
        self.conv3 = nn.Conv1d(bot, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExciteBlock(out_channels, reduction)
        self.activation = activation
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out) + identity
        return self.activation(out)

class ECGResidualModelBinary(nn.Module):
    def __init__(self, segment_length=1500, n_channels=12,
                 dropout_rate=0.5, n_classes=2, activation_choice="relu",
                 num_layers=3, width_scale=1.0, drop_path_rate=0.2):
        super().__init__()
        self.segment_length = segment_length
        self.n_channels = n_channels

        if activation_choice == "leaky":
            self.act = nn.LeakyReLU(0.01)
        elif activation_choice == "gelu":
            self.act = nn.GELU()
        elif activation_choice == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        total_blocks = 3 + (1 if num_layers >= 4 else 0) + (1 if num_layers >= 5 else 0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        ms_out = int(64 * width_scale)
        self.ms_block = MultiScaleBlock(n_channels, ms_out, (3, 5, 7), dilation_rate=2)
        total_ms = ms_out * 3

        ch_progression = [128, 196, 256, 320, 384]
        scaled = [int(c * width_scale) for c in ch_progression]

        self.layer1 = self._make_layer(total_ms,  scaled[0], stride=2, drop_p=dpr[0])
        self.layer2 = self._make_layer(scaled[0], scaled[1], stride=2, drop_p=dpr[1])
        self.layer3 = self._make_layer(scaled[1], scaled[2], stride=2, drop_p=dpr[2])

        self.layer4 = None
        current_out = scaled[2]
        if num_layers >= 4:
            self.layer4 = self._make_layer(current_out, scaled[3], stride=2, drop_p=dpr[3])
            current_out = scaled[3]

        self.layer5 = None
        if num_layers >= 5:
            self.layer5 = self._make_layer(current_out, scaled[4], stride=2, drop_p=dpr[4])
            current_out = scaled[4]

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(current_out + 2, n_classes)
        )

    def _make_layer(self, in_c, out_c, stride, drop_p):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )
        return BottleneckSEBlock(
            in_channels=in_c, out_channels=out_c, stride=stride,
            downsample=downsample, reduction=16, activation=self.act, drop_path=drop_p
        )

    def forward(self, ecg, age, gender):
        if ecg.dim() == 4:
            ecg = ecg.squeeze(1)

        if ecg.dim() == 3:
            if ecg.shape[1] == self.segment_length and ecg.shape[2] == self.n_channels:
                ecg = ecg.permute(0, 2, 1)
            elif ecg.shape[1] == self.n_channels:
                pass
            elif ecg.shape[-1] == self.n_channels:
                ecg = ecg.permute(0, 2, 1)

        x = self.ms_block(ecg)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)

        x = self.pool(x).flatten(1)
        ag = torch.cat([age.view(-1, 1), gender.view(-1, 1)], dim=1)
        return self.fc(torch.cat([x, ag], dim=1))
