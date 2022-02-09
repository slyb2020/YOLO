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

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # python的进度条模块


class YOLOv1ControlPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.log = log
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
        hhbox.Add(self.spendTimeTXT, 0, wx.LEFT, 5)
        vbox.Add(hhbox)
        self.SetSizer(vbox)
        self.Bind(wx.EVT_BUTTON, self.OnStartDetection)

    def OnStartDetection(self, event):
        modelName = modelsDir + self.pretrainedModelCOMBO.GetValue()
        model = torch.load(modelName)  # 加载训练好的模型,但模型的定义一定要在main.py文件中
        transformer = transforms.Compose([transforms.ToTensor()])
        img = cv2.imread("./bitmaps/image.jpg")
        input = transformer(img)
        input = torch.unsqueeze(input, dim=0)
        print(input.shape)
        # 这块差一个resize()
        input = input.cpu()
        pred = model(input)  # pred的尺寸是(1,30,7,7)
        print("pred=", pred, pred.shape)


# 发现一个问题，使用torch.load()方法，只有在main里能用，在子程序中使用会报错
if __name__ == "__main__":
    model = torch.load("D:\\WorkSpace\\Python\\YOLO\\Models\\YOLOv1_epoch10.pkl")
    print(model)
