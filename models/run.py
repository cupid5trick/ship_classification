import torch
import torch.nn as nn
import torch.optim as optimizers

import visdom
import onnx

import time
from pathlib import Path
import os.path as ospath


from datasets import train_loader, test_loader
from models.ResNet18.model import ResNet18
from models.GoogLeNet.model import GoogLeNet
from models.VGG16.model import VGG16



class TrainEvaluation(object):
    loss = nn.BCELoss()
    optimizer_class = optimizers.SGD
    # lr_schedueler = optimizers.lr_scheduler.ReduceLROnPlateau
    lr_schedueler = None
    epoches = 10
    err_trace_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Batch Number',
        'ylabel': 'Loss',
        'width': 600,
        'height': 300,
    }
    acc_trace_opts = {
        'title': 'Epoch Acc Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Acc',
        'width': 600,
        'height': 300,
    }

    def __init__(self, model=ResNet18, lr=1e-2, train_loader=train_loader, test_loader=test_loader,
                 errviz=visdom.Visdom(env='Error-Epoch Trace'),
                 accviz=visdom.Visdom(env='Acc Epoch Trace')):
        self.model = model() if model else None
        self.lr = lr if lr else None
        self.train_loader = train_loader if train_loader else None
        self.test_loader = test_loader if test_loader else None

        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        if self.lr_schedueler:
            self.lr_schedueler = self.lr_schedueler(self.optimizer, verbose=True, patience=3, factor=0.3)
        self.errviz = errviz if isinstance(errviz, visdom.Visdom) and errviz.check_connection() else visdom.Visdom()
        self.accviz = accviz if isinstance(accviz, visdom.Visdom) and accviz.check_connection() else visdom.Visdom()
        self.loss_trace = None
        self.acc_trace = None
        self.losses, self.batches = [], []
        self.acc = []

        self.loss = getattr(self.model, 'loss', None)

        if torch.cuda.is_available():
            cur_device = torch.cuda.current_device()
            print('Train on gpu device %s' % torch.cuda.get_device_name(cur_device))
            torch.cuda.reset_max_memory_allocated(cur_device)
            torch.cuda.reset_max_memory_cached(cur_device)
            self.cur_device = cur_device
            self.model.cuda(torch.cuda.current_device())


    def train(self, epoch):
        assert torch.cuda.is_available(), "GPU not available"
        print("GPU %d: %s" % (torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
        self.model.train()

        running_loss = 0
        for i, (imgs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            imgs = imgs.cuda(device=torch.cuda.current_device())
            output = self.model(imgs)
            # print(output, labels)

            loss = self.loss(imgs, labels)

            # 释放缓存，防止爆显存
            torch.cuda.empty_cache()
            loss.backward()
            self.optimizer.step()


            self.losses.append(loss.item())
            self.batches.append((epoch-1)*len(self.train_loader)+i+1)
            running_loss += self.losses[-1]

            if i % 10 == 0:
                cur_loss = running_loss
                # self.lr_schedueler.step(cur_loss)
                running_loss = 0
                print("Train - Epoch %d, Batch %d, Loss %f" % (epoch, i, self.losses[-1]))


            if self.errviz.check_connection():
                # print(self.loss_trace)
                self.loss_trace = self.errviz.line(
                    torch.Tensor(self.losses), torch.Tensor(self.batches),
                    win=self.loss_trace, name="current_batch_loss",
                    update=None if not self.loss_trace else 'replace',
                    opts=self.err_trace_opts
                )



    def test(self):
        self.model.eval()
        total_correct = 0
        total_sea = 0
        total_ship = 0
        avg_loss = 0
        count = len(self.test_loader.dataset)
        sea_count = 0
        ship_count = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.test_loader):
                # count += 1
                imgs = imgs.cuda(self.cur_device)
                output = self.model(imgs)
                labels = labels.cuda(self.cur_device)

                avg_loss += self.loss(imgs, labels).sum()

                # maxv, pred = output.detach().max(1)
                # print(output.detach().max(1), labels)
                pred = output.view_as(labels) > 0.5
                # print(output)
                total_correct += pred.eq(labels).sum(dtype=torch.float32)
                total_sea += torch.bitwise_and(
                    pred.eq(labels), labels.eq(torch.zeros(labels.size()).cuda(self.cur_device))
                ).sum(dtype=torch.float32)
                total_ship += torch.bitwise_and(
                    pred.eq(labels), labels.eq(torch.ones(labels.size()).cuda(self.cur_device))
                ).sum(dtype=torch.float32)
                sea_count += labels.eq(0).sum(dtype=torch.float32)
                ship_count += labels.eq(1).sum(dtype=torch.float32)

        avg_loss /= len(self.test_loader)
        print('samples: %d ships: %d seas: %d correct ships: %d correct seas: %d' %
              (count, ship_count, sea_count, total_ship, total_sea))
        print('Test Avg. Loss: %f, Acc: %f Acc. sea %f Acc. ship %f' %
              (avg_loss, total_correct/count, total_sea/sea_count, total_ship/ship_count))
        torch.cuda.empty_cache()
        return total_correct/count

    def train_and_test(self, epoch):

        self.train(epoch)
        st_time = time.time()
        acc = self.test()
        self.acc.append(acc)
        self.acc_trace = self.accviz.line(
            torch.Tensor(self.acc), torch.Tensor(range(1, epoch+1)),
            win=self.acc_trace, name="current_epoch_acc",
            update=None if not self.acc_trace else 'replace',
            opts=self.acc_trace_opts
        )
        end_time = time.time()
        print("Time used on %d test samples: %f sec" %
              (len(self.test_loader.dataset), end_time - st_time))

        dummy_input = torch.randn(1, 3, 80, 80,).cuda(self.cur_device)

        model_name = self.model.__class__.__name__.lower()
        name = '%s_epoch_%d_acc_%s.onnx' % (model_name, epoch, str(float(acc)).replace('.', '_')[2:])
        save_path = Path(__file__).parent.parent / "trained_models"
        if not ospath.exists(save_path / model_name):
            (save_path / model_name).mkdir(parents=True)
        save_path = save_path / model_name / name

        # print('model name: ', save_path)
        torch.onnx.export(self.model, (dummy_input,), save_path)

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        torch.cuda.empty_cache()

    def run(self):
        for e in range(self.epoches):
            self.train_and_test(e+1)


def run():

    TrainEvaluation(
        model=VGG16
    ).run()

    TrainEvaluation(
        model=GoogLeNet
    ).run()

    TrainEvaluation(

    ).run()

if __name__  == '__main__':
    run()