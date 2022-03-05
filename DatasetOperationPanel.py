import os
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
from MNISTPanel import MNISTPanel
from CaltechPanel import CaltechPanel
from CIFARPanel import CIFARPanel

class DatasetOperationPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log = log
        self.exDatasetName = None
        self.datasetProperty = None
        self.typeList = ['检测数据集', '识别数据集', '分割数据集']
        self.editDatasetBTN = wx.Button(self, -1, label="编辑数据集", size=(180, 35))
        self.datasetTreePanel = DatasetTreePanel(self, self.log, size=(180, 300))
        self.datasetTempPanel = wx.Panel(self)
        self.datasetShowPanel = wx.Panel(self.datasetTempPanel)
        self.datasetShowPanel.SetBackgroundColour(wx.Colour(122,112,23))
        sizer = wx.BoxSizer()
        sizer.Add(self.datasetShowPanel, 1, wx.EXPAND)
        self.datasetTempPanel.SetSizer(sizer)
        hbox = wx.BoxSizer()
        vvbox = wx.BoxSizer(wx.VERTICAL)
        vvbox.Add(self.editDatasetBTN, 0)
        vvbox.Add(self.datasetTreePanel, 1)
        for i in self.typeList:
            checkBox = wx.CheckBox(self, label=i, size=(-1, 25))
            checkBox.SetValue(True)
            vvbox.Add(checkBox, 0)
        hbox.Add(vvbox, 0, wx.EXPAND)
        hbox.Add(self.datasetTempPanel, 1, wx.EXPAND)
        self.SetSizer(hbox)
        self.Layout()
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnUpdateDatasetTree)

    def OnUpdateDatasetTree(self, event):
        item = self.datasetTreePanel.tree.GetFocusedItem()
        itemdata = self.datasetTreePanel.tree.GetItemData(item)
        itemtext = self.datasetTreePanel.tree.GetItemText(item)
        self.datasetProperty = itemdata.split(',')
        if len(self.datasetProperty)>1:  #如果鼠标每点击tree的那个‘+’
            if self.datasetProperty[1] != self.exDatasetName:   #如果更换了数据集
                self.exDatasetName = self.datasetProperty[1]
                self.datasetShowPanel.Destroy()
                self.datasetTempPanel.DestroyChildren()
                if self.datasetProperty[1] == "MNIST":
                    self.datasetShowPanel = MNISTPanel(self.datasetTempPanel, self, self.log)
                elif self.datasetProperty[1] == "Caltech":
                    self.datasetShowPanel = CaltechPanel(self.datasetTempPanel, self, self.log)
                elif self.datasetProperty[1] == "CIFAR":
                    self.datasetShowPanel = CIFARPanel(self.datasetTempPanel, self, self.log)
                else:
                    self.datasetShowPanel = wx.Panel(self.datasetTempPanel)
                sizer = wx.BoxSizer()
                sizer.Add(self.datasetShowPanel, 1, wx.EXPAND)
                self.datasetTempPanel.SetSizer(sizer)
            else: #如果是在相同数据集的不同子集之间切换
                if self.datasetProperty[1] == "MNIST":
                    if self.datasetProperty[2] == "训练数据集":
                        self.datasetShowPanel.notebook.SetSelection(1)
                    elif self.datasetProperty[2] == "测试数据集":
                        self.datasetShowPanel.notebook.SetSelection(2)
                    else:
                        self.datasetShowPanel.notebook.SetSelection(0)
                elif self.datasetProperty[1] == "Caltech":
                    if self.datasetProperty[2] == "Caltech101":
                        self.datasetShowPanel.notebook.SetSelection(1)
                    elif self.datasetProperty[2] == "Caltech256":
                        self.datasetShowPanel.notebook.SetSelection(2)
                    else:
                        self.datasetShowPanel.notebook.SetSelection(0)
                elif self.datasetProperty[1] == "CIFAR":
                    if self.datasetProperty[2] == "CIFAR10":
                        self.datasetShowPanel.notebook.SetSelection(1)
                    elif self.datasetProperty[2] == "CIFAR100":
                        self.datasetShowPanel.notebook.SetSelection(2)
                    else:
                        self.datasetShowPanel.notebook.SetSelection(0)
        self.datasetTempPanel.Layout()
6