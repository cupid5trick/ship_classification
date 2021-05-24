import torch.nn as nn
from torch.nn.functional import pad
from collections import OrderedDict

from functools import reduce

models = {
    'resnet18': [64, 'M22', 64, 64, 64, 64, 'M22', 128, 128, 128, 128, 'M22', 256, 256, 256, 256,
                 'M22', 512, 512, 512, 512, 'M22']
}


class ResNet18(nn.Module):

    def __init__(self, model_name='resnet18', loss_class=nn.BCELoss):
        super().__init__()
        self.model_name = model_name

        # 18 weight layers
        self.features = self._make_layers(models[model_name])
        self.classifier = nn.Sequential(
            nn.Linear(4 * 512, 1),
            nn.Sigmoid(),
        )

        self.c_64_64 = nn.Conv2d(64, 64, kernel_size=1, stride=2)
        self.c_64_128 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.c_128_256 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.c_256_512 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.loss_class = loss_class

    def c(self, conv_idx, sub_idx, img):
        i, j = conv_idx, sub_idx
        sum_1 = 2
        idx = 0
        if i >= 2:
            idx = sum_1 + 5*(i-2) + j
        elif i == 1:
            idx = 1
        # print(i,j, idx)
        return self.features[idx-1](img)

    def m(self, pool_idx, img):
        i = pool_idx*5 - 3
        return self.features[i-1](img)



    def _make_layers(self, conv_cfg):
        layers = []
        in_channels = 3
        for x in conv_cfg:
            if isinstance(x, int):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(),
                    )
                )
                in_channels = x
            elif isinstance(x, str) and x.startswith('M'):
                x = tuple(x)
                layers.append(nn.MaxPool2d(kernel_size=int(x[1]), stride=int(x[2])))
        return nn.Sequential(*layers)

    def loss(self, input, labels):
        o = self(input)
        loss = self.loss_class()
        return loss(o.view_as(labels), labels.type_as(o))

    def forward(self, img):
        o11 = self.c(1,1, img)
        om1 = self.m(1, o11)

        o21 = self.c(2,1, om1)
        o22 = self.c(2,2, o21)
        # residual layer
        o22 += self.c_64_64(o11)
        o23 = self.c(2,3, o22)
        o24 = self.c(2,4, o23)
        # residual layer
        o24 += o22
        om2 = self.m(2, o24)

        o31 = self.c(3,1, om2)
        o32 = self.c(3,2, o31)
        # residual layer
        o32 += self.c_64_128(o24)
        o33 = self.c(3,3, o32)
        o34 = self.c(3,4, o33)
        o34 += o32
        om3 = self.m(3, o34)

        o41 =self.c(4,1, om3)
        o42 = self.c(4,2, o41)
        o42 += self.c_128_256(o34)
        o43 = self.c(4,3, o42)
        o44 = self.c(4,4, o43)
        o44 += o42
        om4 = self.m(4, o44)

        o51 = self.c(5,1, om4)
        o52 = self.c(5,2, o51)
        o52 += self.c_256_512(o44)
        o53 = self.c(5,3, o52)
        o54 = self.c(5,4, o53)
        o54 += o53
        om5 = self.m(5, o54)

        o = self.classifier(om5.view(om5.size(0), -1))
        return o

    def memory(self):
        gega, mega = 1024 ** 3, 1024 ** 2

        n_params = sum([reduce(lambda x, y: x*y, list(param.size())) for param in self.parameters()])
        print('Memory by float32: %fGB %fMB' % (n_params*4 / gega, n_params*4 / mega))
        print('Memory by float64: %fGB %fMB' % (n_params*8 / gega, n_params*8 / mega))