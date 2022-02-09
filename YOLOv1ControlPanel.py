import cv2
import wx
from ID_DEFINE import *
import os
import torch
import torchvision.models as tvmodel
import torch.nn as nn
import torch
from YOLOv1Pytorch import *
from YOLOv1Pytorch2 import *
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as Loader
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # python的进度条模块
import wx.lib.scrolledpanel as scrolled
from ListDataPanel import ResultDisplayPanel

# class ResultDisplayPanel(scrolled.ScrolledPanel):
#     def __init__(self, parent, log):
#         self.log = log
#         self.parent = parent
#         scrolled.ScrolledPanel.__init__(self, parent, -1)
#         self.Recreate()
#
#     def Recreate(self):
#         self.DestroyChildren()
#         vbox = wx.BoxSizer(wx.VERTICAL)
#         if self.parent.currentPosition:
#             row = self.parent.currentPosition[0]
#             col = self.parent.currentPosition[1]
#             bbox = []
#             bbox.append(self.parent.bbox[row*7+col])
#             bbox.append(self.parent.bbox[2*(row*7+col)])
#             vbox.Add((-1, 5))
#             hhbox = wx.BoxSizer()
#             hhbox.Add((10, -1))
#             hhbox.Add(wx.StaticText(self, label="第%d行，第%d列BOX数据：" % (row + 1, col + 1)))
#             vbox.Add(hhbox, 0, wx.EXPAND)
#             for box in bbox:
#                 hhbox = wx.BoxSizer()
#                 hhbox.Add((10, -1))
#                 hhbox.Add(wx.StaticText(self, label="中心点:", size=(50, -1)), 0, wx.TOP, 5)
#                 self.centerPointTXT=wx.TextCtrl(self, size=(60, -1))
#                 hhbox.Add(self.centerPointTXT, 0)
#                 vbox.Add(hhbox, 0, wx.EXPAND)
#         self.SetSizer(vbox)
#         self.SetAutoLayout(1)
#         self.SetupScrolling()


# 注意检查一下输入数据的格式，到底是xywh还是xyxy
def labels2bbox(matrix):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果
    """
    if matrix.size()[0:2] != (7, 7):
        raise ValueError("Error: Wrong labels size:", matrix.size())
    bbox = torch.zeros((98, 25))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):  # i是网格的行方向(y方向)
        for j in range(7):  # j是网格的列方向(x方向)
            bbox[2 * (i * 7 + j), 0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                       (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i, j, 4]
            bbox[2 * (i * 7 + j), 5:] = matrix[i, j, 10:]
            bbox[2 * (i * 7 + j) + 1, 0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                           (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j) + 1, 4] = matrix[i, j, 9]
            bbox[2 * (i * 7 + j) + 1, 5:] = matrix[i, j, 10:]
    return bbox
    # return NMS(bbox)  # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox


def ChangeImageSize(img, inputSize):
    h, w = img.shape[0:2]
    # 输入YOLOv1网络的图像尺寸为448x448
    # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
    # 然后再将Padding后的正方形图像缩放成448x448
    padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
    if h > w:
        padw = (h - w) // 2
        img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
    elif w > h:
        padh = (w - h) // 2
        img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.resize(img, (inputSize, inputSize))


class YOLOv1ControlPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.parent = parent
        self.log = log
        self.currentPosition = None
        self.SetBackgroundColour(wx.Colour(240, 240, 240))
        self.pretrainedModelList = os.listdir(modelsDir)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hhbox = wx.BoxSizer()
        vbox.Add((-1, 5))
        hhbox.Add((10, -1))
        hhbox.Add(wx.StaticText(self, label="选择预训练模型："), wx.TOP, 5)
        self.pretrainedModelCOMBO = wx.ComboBox(self, value=self.pretrainedModelList[0], size=(200, -1),
                                                choices=self.pretrainedModelList)
        hhbox.Add(self.pretrainedModelCOMBO, 1, wx.RIGHT, 10)
        vbox.Add(hhbox)
        vbox.Add((-1, 5))
        hhbox = wx.BoxSizer()
        hhbox.Add((8, -1))
        self.startDetectionBTN = wx.Button(self, label="开始检测", size=(100, 30))
        hhbox.Add(self.startDetectionBTN, 0)
        hhbox.Add((15, -1))
        hhbox.Add(wx.StaticText(self, label="检测用时："), wx.TOP, 5)
        self.spendTimeTXT = wx.TextCtrl(self, size=(100, -1))
        self.spendTimeTXT.Enable(False)
        hhbox.Add(self.spendTimeTXT, 0, wx.LEFT | wx.TOP, 3)
        vbox.Add(hhbox)
        vbox.Add((-1, 3))
        vbox.Add(wx.StaticLine(self, style=wx.HORIZONTAL), 0, wx.EXPAND)
        vbox.Add((-1, 3))
        hhbox = wx.BoxSizer()
        hhbox.Add((10, -1))
        if self.currentPosition:
            self.label = wx.StaticText(self, label="第%d行，第%d列BOX数据：" % (self.currentPosition[0] + 1, self.currentPosition[1] + 1))
        else:
            self.label = wx.StaticText(self, label="第 行，第 列BOX数据：")
        hhbox.Add(self.label,0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        self.resultDisplayPanel = ResultDisplayPanel(self, self.log)
        vbox.Add(self.resultDisplayPanel, 1, wx.EXPAND)
        self.SetSizer(vbox)
        self.Bind(wx.EVT_BUTTON, self.OnStartDetection)

    def OnStartDetection(self, event):
        modelName = modelsDir + self.pretrainedModelCOMBO.GetValue()
        model = torch.load(modelName)  # 加载训练好的模型,但模型的定义一定要在main.py文件中
        img = cv2.imread(self.parent.middlePanel.filename)
        # transforms.Resize()方法不需要channel维度，但是增广的是最后2个维度，所以需要先ToTensor，然后再Resize
        transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((448, 448))])
        input = transformer(img)
        input = torch.unsqueeze(input, dim=0)
        input = input.cpu()
        startTime = time.time()
        self.pred = model(input)  # pred的尺寸是(1,30,7,7)
        endTime = time.time()
        self.spendTimeTXT.SetValue("%.4f毫秒" % ((endTime - startTime) * 1000))
        self.pred = np.squeeze(self.pred)
        self.pred = self.pred.permute((1, 2, 0))
        self.bbox = labels2bbox(self.pred)


# 发现一个问题，使用torch.load()方法，只有在main里能用，在子程序中使用会报错
if __name__ == "__main__":
    model = torch.load("D:\\WorkSpace\\Python\\YOLO\\Models\\YOLOv1_epoch10.pkl")
    print(model)
