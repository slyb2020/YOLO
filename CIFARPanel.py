import os
from PIL import Image
import torch.utils.data
import torchvision.transforms
import wx
import images
from DatasetTreePanel import DatasetTreePanel
from ListDataPanel import ListDataPanel
from PictureShowPanel import PictureShowPanel
import xml.etree.ElementTree as ET
import os
import cv2
from ID_DEFINE import *
from DatasetLabelProcess import *
import wx.lib.scrolledpanel as scrolled
from torchvision.datasets import CIFAR10,CIFAR100
import numpy as np


class DatasetButtonShowPanel(scrolled.ScrolledPanel):
    def __init__(self, parent, dataset, log):
        scrolled.ScrolledPanel.__init__(self, parent, -1)
        self.parent = parent
        self.log = log
        self.height = 50
        self.width = 50
        wsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        self.buttonIdList = []
        self.buttonFilenameList = []
        for _ in range(198):
            image, label = dataset.__next__()
            image = np.array(image)
            a = image.shape[0]
            b = image.shape[1]
            id = wx.NewId()
            button = wx.Button(self, id, size=(self.width, self.height))
            bmp = wx.Image(b, a, image).Scale(self.width-1 , self.height-1, quality=wx.IMAGE_QUALITY_BOX_AVERAGE).ConvertToBitmap()
            button.SetBitmap(bmp)
            button.SetToolTip(str(label))
            self.buttonIdList.append(id)
            wsizer.Add(button, 0)
        self.SetSizer(wsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()

class CIFAR10Panel(wx.Panel):
    def __init__(self, parent, master, log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.master = master
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
                                    # wx.BK_DEFAULT
                                    # wx.BK_TOP
                                    wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    | wx.NB_MULTILINE
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
        self.trainDataset = iter(CIFAR10(self.master.datasetProperty[-1]+'CIFAR10',train=True))
        self.testDataset = iter(CIFAR10(self.master.datasetProperty[-1]+'CIFAR10',train=False))
        self.trainDatasetPanel = DatasetButtonShowPanel(self.notebook, self.trainDataset, self.log)
        self.notebook.AddPage(self.trainDatasetPanel, "训练数据集")
        self.testDatasetPanel = DatasetButtonShowPanel(self.notebook, self.testDataset, self.log)
        self.notebook.AddPage(self.testDatasetPanel, "测试数据集")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(hbox)


class CIFAR100Panel(wx.Panel):
    def __init__(self, parent, master, log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.master = master
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
                                    # wx.BK_DEFAULT
                                    # wx.BK_TOP
                                    wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    | wx.NB_MULTILINE
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
        for i in range(2):
            self.notebook.AddPage(wx.Panel(self.notebook),"分类%d"%(i))
            # self.datasetIntroductionPanel = wx.Panel(self.notebook)
            # self.notebook.AddPage(self.datasetIntroductionPanel, "数据集介绍")
            # self.caltech101Panel = Caltech101Panel(self.notebook, self.master, self.log)
            # self.notebook.AddPage(self.trainDatasetPanel, "训练集")
            # self.caltech256Panel = Caltech256Panel(self.notebook, self.master, self.log)
            # self.notebook.AddPage(self.testSetPanel, "测试集")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(hbox)


class CIFARPanel(wx.Panel):
    def __init__(self, parent, master, log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.master = master
        # self.trainDataset = iter(Caltech101(self.master.datasetProperty[-1]))
        # # self.trainLoader = iter(torch.utils.data.DataLoader(self.trainDataset, batch_size=198))
        # self.testDataset = iter(Caltech101(self.master.datasetProperty[-1]))
        # # self.testLoader = iter(torch.utils.data.DataLoader(self.testDataset, batch_size=198))
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
                                    # wx.BK_DEFAULT
                                    wx.BK_TOP
                                    # wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    | wx.NB_MULTILINE
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
        self.datasetIntroductionPanel = wx.Panel(self.notebook)
        self.notebook.AddPage(self.datasetIntroductionPanel, "CIFAR数据集介绍")
        self.cifar10Panel = CIFAR10Panel(self.notebook, self.master, self.log)
        self.notebook.AddPage(self.cifar10Panel, "CIFAR10")
        self.cifar100Panel = CIFAR100Panel(self.notebook, self.master, self.log)
        self.notebook.AddPage(self.cifar100Panel, "CIFAR100")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(hbox)
        self.Bind(wx.EVT_BUTTON, self.OnPictureButton)
        if self.master.datasetProperty[0] == "主集":
            self.notebook.SetSelection(0)
        elif self.master.datasetProperty[0] == "子集":
            if self.master.datasetProperty[2] == "CIFAR10":
                self.notebook.SetSelection(1)
            else:
                self.notebook.SetSelection(2)

    def OnPictureButton(self, event):
        id = event.GetId()
        # if id in self.trainDatasetPanel.buttonIdList:
        #     self.index = self.trainDatasetPanel.buttonIdList.index(id)
        #     self.leftPanel.Refresh()

