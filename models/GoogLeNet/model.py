import torch.nn as nn
import torch

import re
import math
import types
from functools import reduce

models = {
    'GoogLeNet': [
        ['C7/1-64', 'M2/2', 'LRN',], ['C1-64', 'C3-192', 'LRN', 'M2/2',],
        ['In-64-96-128-16-32-32', 'In-128-128-192-32-96-64', 'M2/2',],
        ['In-192-96-208-16-48-64', 'In-160-112-224-24-64-64', 'In-128-128-256-24-64-64', 'In-112-144-288-32-64-64',
        'In-256-160-320-32-128-128', 'M2/2',],
        ['In-256-160-320-32-128-128', 'In-384-192-384-48-128-128',],
    ],
    'GoogLeNet18': [
        ['C7/1-64', 'M2/2', 'LRN'], ['C1-32', 'C3-64', 'LRN', 'M2/2',],
        ['In-16-32-64-8-32-16', 'M2/2',],
        ['In-32-48-96-32-96-32', 'M2/2',],
        ['In-64-96-192-64-160-96',],
    ],
    'GoogLeNet13': [
        ['In-16-8-16-8-16-16', 'M2/2',],
        ['In-8-8-24-8-16-16', 'M2/2',],
        ['In-16-32-64-8-32-16', 'M2/2',],
        ['In-32-48-96-32-96-32', 'M2/2',],
        ['In-64-96-192-64-160-96',],
    ]
}


class Inception(nn.Module):
    """
    The Inception Module of GoogLeNet
    """

    def __init__(self, idx, ch_in, cfg):
        super().__init__()
        n1, nr3, n3, nr5, n5, np = list(map(lambda x: int(x), re.match(r'In-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)', cfg).groups()))
        setattr(self, 'Inception-%d-Conv1' % (idx,), nn.Sequential(
            nn.Conv2d(ch_in, n1, kernel_size=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
        ))
        setattr(self, 'Inception-%d-Conv3' % (idx,), nn.Sequential(
            nn.Conv2d(ch_in, nr3, kernel_size=1),
            nn.BatchNorm2d(nr3),
            nn.ReLU(),
            nn.Conv2d(nr3, n3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3),
            nn.ReLU(),
        ))
        setattr(self, 'Inception-%d-Conv5' % (idx,), nn.Sequential(
            nn.Conv2d(ch_in, nr5, kernel_size=1),
            nn.BatchNorm2d(nr5),
            nn.ReLU(),
            nn.Conv2d(nr5, n5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5),
            nn.ReLU(),
        ))
        setattr(self, 'Inception-%d-MaxPool' % (idx,), nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_in, np, kernel_size=1),
            nn.BatchNorm2d(np),
            nn.ReLU(),
        ))

        self.out_channel = n1 + n3 + n5 + np

    def forward(self, img):
        dim = 2
        res = torch.cat([m(img) for m in self.children()], dim=dim-1)
        # print(res.size())
        return res



class Classifier(nn.Module):
    """
    Auxiliary Classifier: AvgPool5/3-pad0 -> Conv1-128 -> FC-1024 -> dropout-0.7 -> FC-<No.labels> -> softmax |
    sigmoid
    Master Classifier: AvgPool -> FC -> softmax | sigmoid
    """
    auxiliary = False
    ks_pool = 2
    padding = 0
    stride = 2
    n1 = 512
    nfc = 2048
    dropout = 0.7
    dropout_master = 0.4
    nlabels = 2

    def __init__(self, feature_size, ch_in, **kwargs):
        super().__init__()
        self.feature_size = feature_size

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.auxiliary:
            self.classifier = self.make_auxiliary(ch_in)
        else:
            self.classifier = self.make_master(ch_in)

    def make_auxiliary(self, ch_in):
        self.feature_size = math.ceil((self.feature_size - self.ks_pool + 2 * self.padding) / self.stride) + 1
        self.feature_size *= self.feature_size * self.n1
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=self.ks_pool, stride=self.stride, padding=self.padding),
            nn.Conv2d(ch_in, self.n1, kernel_size=1),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.nfc),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nfc, self.nlabels if self.nlabels > 2 else 1),
            nn.Softmax(self.nlabels) if self.nlabels > 2 else nn.Sigmoid(),
        )

    def make_master(self, ch_in):
        # print(self.feature_size)
        self.feature_size = math.floor((self.feature_size - self.ks_pool + 2*self.padding) / self.stride + 1)
        self.feature_size *= self.feature_size * ch_in
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=self.ks_pool, stride=self.stride, padding=self.padding),
            nn.Dropout(p=self.dropout_master,),
            nn.Linear(self.feature_size, self.nlabels if self.nlabels > 2 else 1),
            nn.Softmax(self.nlabels) if self.nlabels > 2 else nn.Sigmoid(),
        )

    def get_loss_class(self):
        return nn.CrossEntropyLoss if self.nlabels > 2 else nn.BCELoss

    def forward(self, img):
        if not self.auxiliary:
            o1 = self.classifier[0](img)
            o2 = self.classifier[1](o1)
            o3 = self.classifier[2](o2.view(o2.size()[0], -1))
            o4 = self.classifier[3](o3)
            # print(o3.size())
            return o4
        else:
            o1 = self.classifier[0](img)
            o2 = self.classifier[1](o1)
            o3 = self.classifier[2](o2)
            o4 = self.classifier[3](o3.view(o3.size()[0], -1))
            o5 = self.classifier[4](o4)
            o6 = self.classifier[5](o5)
            o7 = self.classifier[6](o6)
            o8 = self.classifier[7](o7)
            return o8





class GoogLeNet(nn.Module):

    aux_idx = [(4,1), ]
    n_auxs = 0
    in_channels = 3
    size = 80

    def __init__(self, model='GoogLeNet13'):
        super().__init__()
        cfgs = models[model]
        self.conv = self.make_layers(cfgs)
        # print(self.in_channels)
        # print(self.size, self.in_channels)
        self.classifier = Classifier(self.size, self.in_channels)

        self.loss_class = self.classifier.get_loss_class()

        self.add_auxiliary()
        self.memory()

    def make_layers(self, cfgs):
        layers = []
        # print(layers)
        for i, _cfgs in enumerate(cfgs):
            l = []
            for j, cfg in enumerate(_cfgs):
                if cfg == 'LRN':
                    l.append(nn.LocalResponseNorm(3))
                elif cfg.startswith('C'):
                    kwargs = re.match(r'C(?P<kernel_size>\d+)/?(?P<stride>\d+)?-(?P<out_channels>\d+)', cfg).groupdict()
                    for k, v in kwargs.items():
                        if not v:
                            kwargs[k] = 1
                        else:
                            kwargs[k] = int(v)
                    pad = int((kwargs['kernel_size']-1)/2)
                    kwargs.update(in_channels=self.in_channels, padding=pad)
                    # print(kwargs)
                    l.append(nn.Sequential(
                        nn.Conv2d(**kwargs),
                        nn.BatchNorm2d(kwargs['out_channels']),
                        nn.ReLU(),
                    ))
                    self.in_channels = kwargs['out_channels']
                elif cfg.startswith('M'):
                    kwargs = re.match(r'M(?P<kernel_size>\d+)/?(?P<stride>\d+)?', cfg).groupdict()
                    for k, v in kwargs.items():
                        if not v:
                            continue
                        kwargs[k] = int(v)
                    # pad = int(kwargs['kernel_size'] / 2)
                    kwargs.update(padding=0)
                    l.append(nn.MaxPool2d(**kwargs))
                    self.size = math.ceil((self.size-kwargs['kernel_size']+2*kwargs['padding'])/kwargs['stride']) + 1
                elif cfg.startswith('In'):
                    inc = Inception(j+1, self.in_channels, cfg)
                    l.append(inc)
                    self.in_channels = inc.out_channel
                    setattr(self, 'c_%d_%d' % (i + 1, j + 1), l[j])
                    if (i+1,j+1) in self.aux_idx:
                        self.n_auxs += 1
                        setattr(self, '_aux%d' % (self.n_auxs,), Classifier(self.size, self.in_channels,
                                                                             auxiliary=True))
            # print(l)
            layers.append(nn.Sequential(*(l)))
            if i < 2:
                setattr(self, 'c_%d' % (i + 1, ), layers[i])

        return nn.Sequential(*layers)

    def loss(self, input, labels):
        o0 = self.aux1(input)
        o2 = self(input)
        loss = self.loss_class()
        l1 = loss(o2.view_as(labels), labels.type_as(o2))
        l2 = loss(o0.view_as(labels), labels.type_as(o0))
        return l1 + 0.3*l2

    def forward(self, img):
        # print("Input size: ", img.size())
        o_conv = self.conv(img)
        return self.classifier(o_conv)

    def l(self, idx, input):
        if len(idx) == 1:
            return self.conv[idx[0]-1](input)
        return self.conv[idx[0]-1][idx[1]-1](input)

    def add_auxiliary(self):
        """
        动态添加辅助分类器输出
        """
        def aux_output(self, idx, i,j):
            if not getattr(self, 'l') or not getattr(self, '_aux1'):
                return None
            def f(self, input):
                for _i in range(1, i):
                    input = self.l((_i,), input)
                for _j in range(1, j+1):
                    input = self.l((i,_j), input)
                # print(input.size())
                aux = getattr(self, '_aux%d' %idx)
                return aux(input)
            return types.MethodType(f, self)
        for i, idx in enumerate(self.aux_idx):
            setattr(self, 'aux%d' % (i+1), aux_output(self, i+1, *idx))

    def memory(self):
        gega, mega = 1024 ** 3, 1024 ** 2

        n_params = sum([reduce(lambda x, y: x*y, list(param.size())) for param in self.parameters()])
        print('Memory by float32: %fGB %fMB' % (n_params*4 / gega, n_params*4 / mega))
        print('Memory by float64: %fGB %fMB' % (n_params*8 / gega, n_params*8 / mega))
