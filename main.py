import torch

from models.ResNet18 import run
from models.ResNet18 import ResNet18
from models.VGG16 import VGG16
from models.GoogLeNet import GoogLeNet


if __name__ == '__main__':
    # from tensorboardX import SummaryWriter
    #
    # with SummaryWriter('runs') as wr:
    #     # wr.add_graph(GoogLeNet(), torch.randn(10, 3, 80, 80), verbose=True)
    #     wr.add_graph(VGG16(), torch.randn(10, 3, 80, 80), verbose=True)
    #     # wr.add_graph(ResNet18(), torch.randn(10, 3, 80, 80), verbose=True)

    run()