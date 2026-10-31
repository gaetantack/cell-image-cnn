"""CNN models. All take x of shape (batch, 2, 66, 66): channel 0 = brightfield, channel 1 = darkfield.

Each model has forward(x) -> class scores (logits) and embed(x) -> penultimate features (for UMAP).
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class SmallCNN(nn.Module):
    """Small CNN trained from scratch: 4 blocks of conv-batchnorm-ReLU-maxpool, global average pooling,
    dropout and a linear head. 66 px -> 33 -> 16 -> 8 -> 4 px after the four poolings."""

    def __init__(self, n_classes, channels=config.SMALLCNN_CHANNELS, dropout=config.SMALLCNN_DROPOUT):
        super().__init__()
        blocks, c_in = [], len(config.INPUT_CHANNELS)
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, n_classes)

    def embed(self, x):
        """Penultimate features: global average of the last conv block, shape (batch, channels[-1])."""
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.dropout(self.embed(x)))

    def last_conv_layer(self):
        """The layer Grad-CAM hooks onto: the last conv block before pooling (its ReLU output)."""
        return self.features[-2]


class ResNet18(nn.Module):
    """ResNet18 (timm) with a 2-channel first convolution. Input is upscaled to config.RESNET_INPUT_SIZE.

    pretrained=True: ImageNet weights; conv1 gets, for both input channels, the mean of the three RGB
    filters (our channels are grey-scale images, not colours). pretrained=False: same architecture,
    random weights.
    """

    def __init__(self, n_classes, pretrained, input_size=config.RESNET_INPUT_SIZE):
        super().__init__()
        self.input_size = input_size
        net = timm.create_model("resnet18", pretrained=pretrained, num_classes=n_classes)
        old = net.conv1  # 3 input channels
        conv1 = nn.Conv2d(len(config.INPUT_CHANNELS), old.out_channels, kernel_size=old.kernel_size,
                          stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                mean_rgb = old.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                conv1.weight.copy_(mean_rgb.repeat(1, len(config.INPUT_CHANNELS), 1, 1))
        net.conv1 = conv1
        self.net = net

    def _resize(self, x):
        if x.shape[-1] == self.input_size:
            return x
        return F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

    def embed(self, x):
        """Penultimate features: global average pooled output of the last ResNet stage, shape (batch, 512)."""
        return self.net.forward_head(self.net.forward_features(self._resize(x)), pre_logits=True)

    def forward(self, x):
        return self.net.fc(self.embed(x))

    def last_conv_layer(self):
        """The layer Grad-CAM hooks onto: the last residual stage (layer4)."""
        return self.net.layer4


MODEL_NAMES = ("smallcnn", "resnet18_pretrained", "resnet18_scratch")


def build_model(name, n_classes):
    if name == "smallcnn":
        return SmallCNN(n_classes)
    if name == "resnet18_pretrained":
        return ResNet18(n_classes, pretrained=True)
    if name == "resnet18_scratch":
        return ResNet18(n_classes, pretrained=False)
    raise ValueError(f"model must be one of {MODEL_NAMES}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
