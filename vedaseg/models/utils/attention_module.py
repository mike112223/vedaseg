from torch import nn


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class cSEModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(cSEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSEModule(nn.Module):
    def __init__(self, channel):
        super(sSEModule, self).__init__()
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSEModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(scSEModule, self).__init__()
        self.cSE = cSEModule(channel, ratio)
        self.sSE = sSEModule(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)
