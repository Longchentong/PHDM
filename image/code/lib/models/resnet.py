import torch.nn as nn
from lib.Euclidean.blocks.resnet_blocks import BasicBlock


class ResNet(nn.Module):
    """Euclidean ResNet backbone for the hybrid PHDM classifiers."""

    def __init__(self, block, num_blocks, img_dim=[3, 32, 32], embed_dim=512,
                 num_classes=100, bias=True, remove_linear=False):
        super().__init__()
        self.img_dim = img_dim[0]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim
        self.bias = bias
        self.block = block
        self.manifold = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.img_dim, self.in_channels, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(self.in_channels), nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, self.in_channels, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, self.conv3_dim, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, self.conv4_dim, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, self.embed_dim, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.predictor = None if remove_linear else nn.Linear(self.embed_dim * block.expansion, num_classes, bias=self.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        for stride in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_channels, out_channels, stride, self.bias))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out).view(out.size(0), -1)
        return self.predictor(out) if self.predictor is not None else out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
