"""224-channel baselines; the two early DeepLab variants are explicit adaptations."""
import torch
from torch import nn
from torch.nn import functional as F

NAMES = ['unet', 'unetplusplus', 'deeplabv1', 'deeplabv2',
         'deeplabv3', 'deeplabv3plus', 'segformer']


class DeepLabV1LargeFOV(nn.Module):
    """VGG16 LargeFOV, stride 8, atrous fc6 (rate 12), without dense CRF.

    Adapted to 224 input channels and two classes. This is a CNN baseline,
    not a reproduction of the paper's full CNN+CRF evaluation pipeline.
    https://arxiv.org/abs/1412.7062
    """
    def __init__(self):
        super().__init__()
        layers, channels = [], 224
        for block, (out_channels, repeats) in enumerate(
                [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]):
            dilation = 2 if block == 4 else 1
            for _ in range(repeats):
                layers += [nn.Conv2d(channels, out_channels, 3,
                                     padding=dilation, dilation=dilation), nn.ReLU(inplace=True)]
                channels = out_channels
            layers += [nn.MaxPool2d(3, stride=2 if block < 3 else 1, padding=1)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=12, dilation=12), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Conv2d(1024, 1024, 1), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Conv2d(1024, 2, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return F.interpolate(self.classifier(self.features(x)), x.shape[-2:],
                             mode='bilinear', align_corners=False)


class DeepLabV2(nn.Module):
    """ResNet101 stride 8 + summed class-score ASPP (6,12,18,24), no CRF.

    Adapted from the network design in https://arxiv.org/abs/1606.00915 .
    No multi-scale inference, auxiliary objective, or RGB projection.
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet101
        backbone = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        backbone.conv1 = nn.Conv2d(224, 64, 7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.features = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                      backbone.maxpool, backbone.layer1, backbone.layer2,
                                      backbone.layer3, backbone.layer4)
        self.aspp = nn.ModuleList([nn.Conv2d(2048, 2, 3, padding=r, dilation=r)
                                   for r in (6, 12, 18, 24)])
        for conv in self.aspp:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        features = self.features(x)
        scores = sum(branch(features) for branch in self.aspp)
        return F.interpolate(scores, x.shape[-2:], mode='bilinear', align_corners=False)


def build_model(name):
    import segmentation_models_pytorch as smp
    common = dict(in_channels=224, classes=2, encoder_weights=None, activation=None)
    if name == 'deeplabv1':
        return DeepLabV1LargeFOV()
    if name == 'deeplabv2':
        return DeepLabV2()
    if name == 'segformer':
        return smp.Segformer(encoder_name='mit_b2', **common)
    constructors = dict(unet=smp.Unet, unetplusplus=smp.UnetPlusPlus,
                        deeplabv3=smp.DeepLabV3, deeplabv3plus=smp.DeepLabV3Plus)
    return constructors[name](encoder_name='resnet50', **common)


def model_description(name):
    return {
        'implementation': 'local_adaptation_no_CRF' if name in ('deeplabv1', 'deeplabv2') else 'local_SMP',
        'backbone': {'deeplabv1': 'VGG16_LargeFOV', 'deeplabv2': 'ResNet101',
                     'segformer': 'MiT-B2'}.get(name, 'ResNet50'),
        'input_channels': 224, 'classes': 2, 'pretrained': False,
        'postprocessing': None, 'auxiliary_loss': None,
    }
