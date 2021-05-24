import torch.nn as nn
import torch

from functools import reduce

models = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}

class VGG16(nn.Module):
    in_channels = 3
    loss_class = nn.BCELoss

    def __init__(self, model='VGG11'):
        super().__init__()

        self.conv = self.make_layers(models[model])

        self.classifier = nn.Sequential(
            nn.Linear(2*2*512, 1),
            nn.Sigmoid(),
        )
        self._loss = self.loss_class()

    def make_layers(self, cfgs):
        layers = []
        ch_in = self.in_channels
        for cfg in cfgs:
            if cfg == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(cfg, int):
                layers.append(nn.Sequential(
                    nn.Conv2d(ch_in, cfg, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(cfg),
                    nn.ReLU(),
                ))
                ch_in = cfg

        return nn.Sequential(*layers)

    def forward(self, img):
        o1 = self.conv(img)
        return self.classifier(o1.view(o1.size()[0], -1))

    def loss(self, input, labels):
        # print(input.size())
        o = self(input)
        return self._loss(o.view_as(labels), labels.type_as(o))

    def memory(self):
        gega, mega = 1024 ** 3, 1024 ** 2

        n_params = sum([reduce(lambda x, y: x*y, list(param.size())) for param in self.parameters()])
        print('Memory by float32: %fGB %fMB' % (n_params*4 / gega, n_params*4 / mega))
        print('Memory by float64: %fGB %fMB' % (n_params*8 / gega, n_params*8 / mega))