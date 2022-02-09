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
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnUpdateDatasetTree)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnDatasetListSelectionChanged, self.listDataPanel.list)
        self.Bind(wx.EVT_BUTTON, self.OnButton)

    def OnButton(self, event):
        objId = event.GetId()
        for row, buttonList in enumerate(ButtonIdArray):
            if objId in buttonList:
                col = buttonList.index(objId)
                print(row, col)
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
        self.middlePanel.Refresh()
