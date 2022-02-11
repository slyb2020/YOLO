import copy
import os
import wx
import images
from DatasetTree import DatasetTree
from ListDataPanel import ListDataPanel
from PictureShowPanel import PictureShowPanel
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


class YOLOv1Panel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log = log
        self.datasetDir = None
        self.editDatasetBTN = wx.Button(self, -1, label="编辑数据集", size=(180, 35))
        self.datasetTree = DatasetTree(self, self.log, size=(180, 300), wantedList=['DETECTION'])
        self.listDataPanel = ListDataPanel(self, self.log, [])
        self.middlePanel = YOLOPictureShowPanel(self, self.log, size=(630, -1), gap=True)
        self.rightPanel = YOLOv1ControlPanel(self, self.log, size=(300, -1))
        hbox = wx.BoxSizer()
        vvbox = wx.BoxSizer(wx.VERTICAL)
        vvbox.Add(self.editDatasetBTN, 0)
        vvbox.Add(self.datasetTree, 1)
        hbox.Add(vvbox, 0, wx.EXPAND)
        hbox.Add(self.listDataPanel, 0, wx.EXPAND)
        hbox.Add(self.middlePanel, 1, wx.EXPAND)
        hbox.Add(self.rightPanel, 0, wx.EXPAND)
        self.SetSizer(hbox)
        self.rightPanel.startDetectionBTN.Bind(wx.EVT_BUTTON, self.OnStartDetection)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnUpdateDatasetTree)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnDatasetListSelectionChanged, self.listDataPanel.list)
        self.Bind(wx.EVT_BUTTON, self.OnButton)
        self.rightPanel.enablePicBTN.Bind(wx.EVT_TOGGLEBUTTON, self.OnEnablePicBTN)
        self.rightPanel.enableLabelBTN.Bind(wx.EVT_TOGGLEBUTTON, self.OnEnableLabelBTN)
        self.rightPanel.enableBoxBTN.Bind(wx.EVT_TOGGLEBUTTON, self.OnEnableBoxBTN)

    def OnEnableLabelBTN(self, event):
        self.middlePanel.enableLabel = not self.middlePanel.enableLabel
        self.middlePanel.Refresh()

    def OnEnableBoxBTN(self, event):
        self.middlePanel.enableBox = not self.middlePanel.enableBox
        self.middlePanel.Refresh()

    def OnEnablePicBTN(self, event):
        self.middlePanel.enablePicture = not self.middlePanel.enablePicture
        self.middlePanel.Refresh()


    def OnStartDetection(self, event):
        modelName = modelsDir + self.rightPanel.pretrainedModelCOMBO.GetValue()
        model = torch.load(modelName)  # 加载训练好的模型,但模型的定义一定要在main.py文件中
        img = cv2.imread(self.middlePanel.filename)
        # transforms.Resize()方法不需要channel维度，但是增广的是最后2个维度，所以需要先ToTensor，然后再Resize
        transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((448, 448))])
        input = transformer(img)
        input = torch.unsqueeze(input, dim=0)
        input = input.cpu()
        startTime = time.time()
        self.pred = model(input)  # pred的尺寸是(1,30,7,7)
        endTime = time.time()
        self.rightPanel.spendTimeTXT.SetValue("%.4f毫秒" % ((endTime - startTime) * 1000))
        self.pred = np.squeeze(self.pred)
        self.pred = self.pred.permute((1, 2, 0))
        self.middlePanel.bbox = labels2bbox(self.pred)
        self.middlePanel.BBOX = NMS(self.middlePanel.bbox)
        self.middlePanel.Refresh()

    def OnButton(self, event):
        data = []
        objId = event.GetId()
        for row, buttonList in enumerate(ButtonIdArray):
            if objId in buttonList:
                col = buttonList.index(objId)
                self.rightPanel.currentPosition = (row, col)
                data.append(list(self.middlePanel.bbox[2 * (row * 7 + col)].detach().numpy()))
                data.append(list(self.middlePanel.bbox[2 * (row * 7 + col)+1].detach().numpy()))
                self.rightPanel.resultDisplayPanel.ReCreate(data)
                break
        self.rightPanel.label.SetLabel("第%d行，第%d列BOX数据：" % (row + 1, col + 1))
        event.Skip()

    def OnUpdateDatasetTree(self, event):
        item = self.datasetTree.tree.GetFocusedItem()
        itemdata = self.datasetTree.tree.GetItemData(item)
        itemtext = self.datasetTree.tree.GetItemText(item)
        if '孙集' in itemdata:
            self.datasetDir = itemdata[2:]
            data = os.listdir(self.datasetDir)
            self.listDataPanel.ReCreate(data)

    def OnDatasetListSelectionChanged(self, event):
        currentItem = event.Index
        self.middlePanel.filename = self.datasetDir + "\\" + self.listDataPanel.list.GetItemText(currentItem)
        annotationFilename = self.datasetDir + "\\..\\" + "Annotations\\" + self.listDataPanel.list.GetItemText(
            currentItem)
        annotationFilename = annotationFilename[:-3] + "xml"
        self.middlePanel.labelList = GetAnnotationInfo(annotationFilename)
        self.OnStartDetection(None)
        self.middlePanel.Refresh()
