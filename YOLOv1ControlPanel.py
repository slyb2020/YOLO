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
        self.pretrainedModelCOMBO = wx.ComboBox(self, value=self.pretrainedModelList[-1], size=(200, -1),
                                                choices=self.pretrainedModelList)
        hhbox.Add(self.pretrainedModelCOMBO, 1, wx.RIGHT, 10)
        vbox.Add(hhbox)
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
        vbox.Add((-1, 5))
        hhbox = wx.BoxSizer()
        hhbox.Add((10, -1))
        self.enablePicBTN = wx.ToggleButton(self, label="图片", size=(-1, 30))
        hhbox.Add(self.enablePicBTN, 1, wx.RIGHT, 5)
        self.enableLabelBTN = wx.ToggleButton(self, label="标签", size=(-1, 30))
        hhbox.Add(self.enableLabelBTN, 1, wx.RIGHT, 5)
        self.enableBoxBTN = wx.ToggleButton(self, label="检测结果", size=(-1, 30))
        hhbox.Add(self.enableBoxBTN, 1, wx.RIGHT, 5)
        vbox.Add(hhbox, 0, wx.EXPAND)
        vbox.Add((-1, 3))
        vbox.Add(wx.StaticLine(self, style=wx.HORIZONTAL), 0, wx.EXPAND)
        vbox.Add((-1, 3))
        hhbox = wx.BoxSizer()
        hhbox.Add((10, -1))
        if self.currentPosition:
            self.label = wx.StaticText(self, label="第%d行，第%d列BOX数据：" % (
                self.currentPosition[0] + 1, self.currentPosition[1] + 1))
        else:
            self.label = wx.StaticText(self, label="第 行，第 列BOX数据：")
        hhbox.Add(self.label, 0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        self.resultDisplayPanel = ResultDisplayPanel(self, self.log)
        vbox.Add(self.resultDisplayPanel, 1, wx.EXPAND)
        self.SetSizer(vbox)
        # self.startDetectionBTN.Bind(wx.EVT_BUTTON, self.OnStartDetection)
        self.enablePicBTN.Bind(wx.EVT_TOGGLEBUTTON, self.OnEnablePicBTN)
        self.enableLabelBTN.Bind(wx.EVT_TOGGLEBUTTON, self.OnEnableLabelBTN)
        self.enableBoxBTN.Bind(wx.EVT_TOGGLEBUTTON, self.OnEnableBoxBTN)

    def OnEnableLabelBTN(self, event):
        self.parent.middlePanel.enableLabel = not self.parent.middlePanel.enableLabel
        self.parent.middlePanel.Refresh()

    def OnEnableBoxBTN(self, event):
        self.parent.middlePanel.enableBox = not self.parent.middlePanel.enableBox
        self.parent.middlePanel.Refresh()

    def OnEnablePicBTN(self, event):
        self.parent.middlePanel.enablePicture = not self.parent.middlePanel.enablePicture
        self.parent.middlePanel.Refresh()



# 发现一个问题，使用torch.load()方法，只有在main里能用，在子程序中使用会报错
if __name__ == "__main__":
    model = torch.load("D:\\WorkSpace\\Python\\YOLO\\Models\\YOLOv1_epoch10.pkl")
    print(model)
