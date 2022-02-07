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


class YOLOv1Panel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log = log
        self.datasetDir = None
        self.editDatasetBTN = wx.Button(self, -1, label="编辑数据集", size=(180, 35))
        self.datasetTree = DatasetTree(self, self.log, size=(180, 300), wantedList=['DETECTION'])
        self.listDataPanel = ListDataPanel(self, self.log, [])
        self.middlePanel = PictureShowPanel(self, self.log, size=(630, -1))
        self.rightPanel = wx.Panel(self, -1, size=(300, -1), style=wx.BORDER_THEME)
        hbox = wx.BoxSizer()
        vvbox = wx.BoxSizer(wx.VERTICAL)
        vvbox.Add(self.editDatasetBTN, 0)
        vvbox.Add(self.datasetTree, 1)
        hbox.Add(vvbox, 0, wx.EXPAND)
        hbox.Add(self.listDataPanel, 0, wx.EXPAND)
        hbox.Add(self.middlePanel, 1, wx.EXPAND)
        hbox.Add(self.rightPanel, 0, wx.EXPAND)
        self.SetSizer(hbox)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnUpdateDatasetTree)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnDatasetListSelectionChanged, self.listDataPanel.list)

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
        self.middlePanel.Refresh()
