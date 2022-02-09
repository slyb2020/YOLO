import torchvision.models as tvmodel
import torch.nn as nn
import torch
from YOLOv1Pytorch import *
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as Loader
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # python的进度条模块


# NUM_BBOX = 2
# CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

class YOLOv1_resnet(nn.Module):
    def __init__(self):
        super(YOLOv1_resnet, self).__init__()
        resnet = tvmodel.resnet101(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0], -1)
        input = self.Conn_layers(input)
        return input.reshape(-1, (5 * NUM_BBOX + len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据


# if __name__ == '__main__':
#     x = torch.randn((1,3,448,448))
#     net = YOLOv1_resnet()
#     print(net)
#     y = net(x)
#     print(y.size())
if __name__ == '__main__':
    epoch = 50
    batchsize = 5
    lr = 0.001

    train_data = VOC2012()
    train_dataloader = DataLoader(VOC2012(is_train=True), batch_size=batchsize, shuffle=True)

    model = YOLOv1_resnet().cpu()  # cuda()
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    is_vis = False  # 是否进行可视化，如果没有visdom可以将其设置为false
    if is_vis:
        vis = visdom.Visdom()
        viswin1 = vis.line(np.array([0.]), np.array([0.]),
                           opts=dict(title="Loss/Step", xlabel="100*step", ylabel="Loss"))

    for e in range(epoch):
        model.train()
        yl = torch.Tensor([0]).cpu()  # cuda()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.cpu()  # cuda()
            labels = labels.float().cpu()  # cuda()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (e, epoch, i, len(train_data) // batchsize, loss))
            yl = yl + loss
            if is_vis and (i + 1) % 100 == 0:
                vis.line(np.array([yl.cpu().item() / (i + 1)]), np.array([i + e * len(train_data) // batchsize]),
                         win=viswin1, update='append')
        if (e + 1) % 10 == 0:
            torch.save(model, "./models_pkl/YOLOv1_epoch" + str(e + 1) + ".pkl")
            # compute_val_map(model)
