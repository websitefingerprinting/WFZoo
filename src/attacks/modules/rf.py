import math

import torch.nn as nn


class RFNet(nn.Module):

    def __init__(self, num_classes: int = 100, init_weights: bool = True):
        super(RFNet, self).__init__()
        cfg = {
            'N': [128, 128, 'M', 256, 256, 'M', 512]
        }
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        self.first_layer = self.make_first_layers()
        self.features = self.make_layers(cfg['N'] + [num_classes], in_channels=self.first_layer_out_channel)
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dimension
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def make_layers(cfg, in_channels=32):
        layers = []

        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
            else:
                conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
                layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
                in_channels = v

        return nn.Sequential(*layers)

    @staticmethod
    def make_first_layers(in_channels=1, out_channel=32):
        layers = []
        conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

        conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
        layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

        layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
