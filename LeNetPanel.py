import copy
import os
import wx
import images
from DatasetTreePanel import DatasetTreePanel
from ListDataPanel import ListDataPanel
import xml.etree.ElementTree as ET
import os
import cv2
from ID_DEFINE import *
from DatasetLabelProcess import *
from PictureShowPanel import YOLOPictureShowPanel
from YOLOv1ControlPanel import YOLOv1ControlPanel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # python的进度条模块
import torch
import torchvision.transforms as transforms
import time
import numpy as np
from YOLOv1Algorithm import *
import wx.lib.scrolledpanel as scrolled
# from MNIST_Dataset import testDataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image


LeNetModelList = ["原始LeNet模型","改进LeNet模型","LeeNet模型"]

testDataset = MNIST(root="D:\\WorkSpace\\DataSet", train=False)

class ErrorPicPanel(scrolled.ScrolledPanel):
    def __init__(self, parent):
        scrolled.ScrolledPanel.__init__(self, parent, -1)
        # x, y = self.GetClientSize()
        x, y = 45,45
        hbox = wx.BoxSizer()
        for i in range(1):
            btn = wx.Button(self, size=(y,y))
            img, label = testDataset[i]
            img = np.array(img)
            self.img = np.zeros((28,28,3),dtype=np.int8)
            self.img[:,:,0]=img
            self.img[:,:,1]=img
            self.img[:,:,2]=img
            # self.img = cv2.imread("D:\\WorkSpace\\DataSet\\DogCat\\train\\cat.38.jpg")
            # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.width, self.height = self.img.shape[1], self.img.shape[0]
            bmp = wx.Image(self.width, self.height, self.img).Scale(width=y, height=y,
                                            quality=wx.IMAGE_QUALITY_BOX_AVERAGE).ConvertToBitmap()
            btn.SetBitmap(bmp)
            hbox.Add(btn,0)
        self.SetSizer(hbox)
        self.SetAutoLayout(1)
        self.SetupScrolling()


class LeNetMNISTPanel(wx.Panel):
    def __init__(self, parent,  log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.leftPanel = wx.Panel(self, size=(220,-1), style=wx.BORDER_THEME)
        self.middlePanel = wx.Panel(self, size=(300,-1))
        self.rightPanel = wx.Panel(self, size=(800,-1), style=wx.BORDER_THEME)
        hbox = wx.BoxSizer()
        hbox.Add(self.leftPanel, 0, wx.EXPAND)
        hbox.Add(self.middlePanel, 0, wx.EXPAND)
        hbox.Add(self.rightPanel, 1, wx.EXPAND)
        self.SetSizer(hbox)
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add((-1,5))
        hhbox = wx.BoxSizer()
        hhbox.Add((10,-1))
        hhbox.Add(wx.StaticText(self.leftPanel, label="模型结构:", size=(70,-1)),0,wx.TOP,5)
        self.modelStructureCombo = wx.ComboBox(self.leftPanel, value="原始LeNet模型", choices=LeNetModelList, size=(120,-1))
        hhbox.Add(self.modelStructureCombo, 0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        vbox.Add((-1,10))
        hhbox = wx.BoxSizer()
        hhbox.Add((10,-1))
        hhbox.Add(wx.StaticText(self.leftPanel, label="预训练模型:", size=(70,-1)),0,wx.TOP,5)
        self.preModelCombo = wx.ComboBox(self.leftPanel, value="原始LeNet模型", choices=LeNetModelList, size=(120,-1))
        hhbox.Add(self.preModelCombo, 0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        self.leftPanel.SetSizer(vbox)

        hhbox=wx.BoxSizer()
        self.modelTXT = wx.TextCtrl(self.middlePanel, size=(100,100), style=wx.TE_MULTILINE|wx.TE_READONLY)
        hhbox.Add(self.modelTXT, 1, wx.ALL|wx.EXPAND)
        self.middlePanel.SetSizer(hhbox)

        vvbox=wx.BoxSizer(wx.VERTICAL)
        for i in range(10):
            hhbox = wx.BoxSizer()
            button = wx.Button(self.rightPanel,label="%d"%i,size=(50,10))
            hhbox.Add(button, 0,wx.EXPAND)
            panel = ErrorPicPanel(self.rightPanel)
            hhbox.Add(panel,1, wx.EXPAND)
            vvbox.Add(hhbox,1,wx.EXPAND)
        self.rightPanel.SetSizer(vvbox)

class LeNetPanel(wx.Panel):
    def __init__(self, parent,  log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
                                    # wx.BK_DEFAULT
                                    # wx.BK_TOP
                                    wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    # | wx.NB_MULTILINE
                                    )
        il = wx.ImageList(16, 16)
        il.Add(images._rt_smiley.GetBitmap())
        self.total_page_num = 0
        self.notebook.AssignImageList(il)
        idx2 = il.Add(images.GridBG.GetBitmap())
        idx3 = il.Add(images.Smiles.GetBitmap())
        idx4 = il.Add(images._rt_undo.GetBitmap())
        idx5 = il.Add(images._rt_save.GetBitmap())
        idx6 = il.Add(images._rt_redo.GetBitmap())
        hbox = wx.BoxSizer()
        self.leNetIntroductionPanel = wx.Panel(self.notebook,style=wx.BORDER_THEME)
        self.notebook.AddPage(self.leNetIntroductionPanel, "LeNet神经网络模型介绍")
        self.leNetNMISTlPanel = LeNetMNISTPanel(self.notebook, self.log)
        self.notebook.AddPage(self.leNetNMISTlPanel, "LeNet在MNIST上的应用")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(hbox)
    #     self.Bind(wx.EVT_BUTTON, self.OnPictureButton)
    #
    # def OnPictureButton(self, event):
    #     id = event.GetId()
    #     # if id in self.trainDatasetPanel.buttonIdList:
    #     #     self.index = self.trainDatasetPanel.buttonIdList.index(id)
    #     #     self.leftPanel.Refresh()

