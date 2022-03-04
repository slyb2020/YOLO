#!/usr/bin/env python
# encoding: utf-8
import wx
from math import *
from ID_DEFINE import *
import string
import images
import xml.etree.ElementTree as ET
import os
import cv2


def GetDatasetInfo():  # 解析DatasetInformation.xml文件，获取本地所有数据集的相关信息
    """
    把图像imageId的xml文件转换为目标检测的label文件(txt),
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化
    :param imageId:
    :return:
    """
    infoFilename = "DatasetInformation"
    result = []
    inputFilename = open(infoFilename + '.xml', encoding='utf-8')
    # outputFilename = open('./labels/%s.txt' % (imageId), 'w')
    tree = ET.parse(inputFilename)
    root = tree.getroot()
    for obj in root.iter('Dataset'):
        subList = []
        datasetName = obj.find('name').text
        datasetLayer = obj.find('layer').text
        datasetType = obj.find('type').text
        datasetDir = obj.find('dir').text
        if datasetLayer != "0":
            for subSet in obj.iter('SubDataset'):
                grandList = []
                subDatasetName = subSet.find('name').text
                subDatasetLayer = subSet.find('layer').text
                subDatasetType = subSet.find('type').text
                subDatasetDir = subSet.find('dir').text
                if subDatasetLayer != "0":
                    for grandSet in subSet.iter('GrandDataset'):
                        grandDatasetName = grandSet.find('name').text
                        grandDatasetType = grandSet.find('type').text
                        grandDatasetDir = grandSet.find('dir').text
                        grandList.append([grandDatasetName, grandDatasetType, grandDatasetDir])
                subList.append([subDatasetName, subDatasetLayer, subDatasetType, subDatasetDir, grandList])
        result.append([datasetName,datasetLayer, datasetType, datasetDir, subList])
    return result


class MyTreeCtrl(wx.TreeCtrl):
    def __init__(self, parent, id, pos, size, style, log):
        wx.TreeCtrl.__init__(self, parent, id, pos, size, style)
        self.log = log

    def OnCompareItems(self, item1, item2):
        t1 = self.GetItemText(item1)
        t2 = self.GetItemText(item2)
        self.log.WriteText('compare: ' + t1 + ' <> ' + t2 + '\n')
        if t1 < t2: return -1
        if t1 == t2: return 0
        return 1


class DatasetTreePanel(wx.Panel):
    def __init__(self, parent, log, size, wantedList=['DETECTION', 'RECOGNITION']):
        # Use the WANTS_CHARS style so the panel doesn't eat the Return key.
        wx.Panel.__init__(self, parent, -1, size, style=wx.BORDER_THEME)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.log = log
        self.wantedList = wantedList
        tID = wx.NewIdRef()
        # datasetList = [
        #     ["ImageNet", [["2017", []], ["2018", []]]],
        #     ["FasionMNIST", [["2007", []], ["2012", []]]],
        #     ["MNIST", [["2007", []], ["2012", []]]],
        #     ["猫狗大战", []],
        # ]
        datasetList = GetDatasetInfo()

        self.tree = MyTreeCtrl(self, tID, wx.DefaultPosition, size,
                               wx.TR_HAS_BUTTONS
                               | wx.TR_EDIT_LABELS
                               # | wx.TR_MULTIPLE
                               # | wx.TR_HIDE_ROOT
                               , self.log)
        isz = (16, 16)
        il = wx.ImageList(isz[0], isz[1])
        fldridx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, isz))
        fldropenidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_OTHER, isz))
        fileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        smileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        # smileidx    = il.Add(images.Smiles.GetBitmap())
        self.tree.SetImageList(il)
        self.il = il
        self.ReCreateTree()
        self.Bind(wx.EVT_TREE_BEGIN_LABEL_EDIT, self.OnBeginEdit, self.tree)
        self.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit, self.tree)

    def OnSize(self, event):
        w, h = self.GetClientSize()
        self.tree.SetSize(0, 0, w, h)

    def OnBeginEdit(self, event):
        event.Veto()

    def OnEndEdit(self, event):
        event.Veto()

    def ReCreateTree(self):
        datasetList = GetDatasetInfo()
        self.tree.Destroy()
        tID = wx.NewIdRef()
        self.tree = MyTreeCtrl(self, tID, wx.DefaultPosition, (200, 900),
                               wx.TR_HAS_BUTTONS
                               | wx.TR_EDIT_LABELS
                               # | wx.TR_MULTIPLE
                               # | wx.TR_HIDE_ROOT
                               , self.log)
        isz = (16, 16)
        il = wx.ImageList(isz[0], isz[1])
        fldridx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, isz))
        fldropenidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_OTHER, isz))
        fileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        smileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        self.tree.SetImageList(il)
        self.il = il
        self.root = self.tree.AddRoot("数据集")
        self.tree.SetItemData(self.root, "根")
        self.tree.SetItemImage(self.root, fldridx, wx.TreeItemIcon_Normal)
        self.tree.SetItemImage(self.root, fldropenidx, wx.TreeItemIcon_Expanded)
        for i in datasetList:
            child = self.tree.AppendItem(self.root, i[0])
            self.tree.SetItemData(child,"主集," + str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + ',' + str(i[3]))
            if i[1] != '0':
                self.tree.SetItemImage(child, fldridx, wx.TreeItemIcon_Normal)
                self.tree.SetItemImage(child, fldropenidx, wx.TreeItemIcon_Expanded)
                for j in i[4]:
                    last = self.tree.AppendItem(child, j[0])
                    self.tree.SetItemData(last, "子集," + i[0] + ',' + j[0] + ',' + j[1] + ',' + j[2] + ',' + j[3])
                    if j[1] != '0':
                        self.tree.SetItemImage(last, fldridx, wx.TreeItemIcon_Normal)
                        self.tree.SetItemImage(last, fldropenidx, wx.TreeItemIcon_Expanded)
                        for k in j[4]:
                            if k[3] in self.wantedList:
                                item = self.tree.AppendItem(last, k[0])
                                self.tree.SetItemData(item, "孙集," + i[0] + ',' + j[0] + ',' + k[0] + ',' + k[1] + ',' + k[2])
                                self.tree.SetItemImage(item, fileidx, wx.TreeItemIcon_Normal)
                                self.tree.SetItemImage(item, smileidx, wx.TreeItemIcon_Selected)
                    else:
                        self.tree.SetItemImage(last, fileidx, wx.TreeItemIcon_Normal)
                        self.tree.SetItemImage(last, smileidx, wx.TreeItemIcon_Expanded)
            else:
                self.tree.SetItemImage(child, fileidx, wx.TreeItemIcon_Normal)
                self.tree.SetItemImage(child, smileidx, wx.TreeItemIcon_Expanded)
        self.tree.CollapseAll()
        self.tree.Expand(self.root)


if __name__ == "__main__":
    result = GetDatasetInfo()
    print("result=", result)
