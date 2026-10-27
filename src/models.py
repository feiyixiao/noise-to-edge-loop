import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import config as C


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetScratch(nn.Module):
    """4 levels, base 16 channels."""

    def __init__(self, base=C.BASE_CH):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(1, ch[0]), _block(ch[0], ch[1]), _block(ch[1], ch[2]), _block(ch[2], ch[3])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([_block(ch[i] * 2, ch[i]) for i in range(3)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips[-1]
        for i in (2, 1, 0):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


class UNetResNet18(nn.Module):
    """U-Net decoder on an ImageNet-pretrained ResNet18 encoder (conv1 averaged over RGB -> 1 channel)."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv1.weight.data = r.conv1.weight.data.mean(1, keepdim=True)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu)           # 64 @ /2
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # 64/2,128/4... see forward
        self.d4 = _block(512 + 256, 256)
        self.d3 = _block(256 + 128, 128)
        self.d2 = _block(128 + 64, 64)
        self.d1 = _block(64 + 64, 32)
        self.d0 = _block(32 + 1, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x0 = x
        s1 = self.stem(x)                 # /2, 64
        s2 = self.l1(self.pool(s1))       # /4, 64
        s3 = self.l2(s2)                  # /8, 128
        s4 = self.l3(s3)                  # /16, 256
        s5 = self.l4(s4)                  # /32, 512
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        y = self.d4(up(s5, s4))
        y = self.d3(up(y, s3))
        y = self.d2(up(y, s2))
        y = self.d1(up(y, s1))
        y = self.d0(up(y, x0))
        return self.head(y)


def build(name):
    return {"unet_scratch": UNetScratch, "unet_resnet18": UNetResNet18}[name]()
