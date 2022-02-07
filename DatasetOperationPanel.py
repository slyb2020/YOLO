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

def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def GetAnnotationInfo(annotationFilename):
    """把图像image_id的xml文件转换为目标检测的label文件(txt)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化"""
    tree = ET.parse(annotationFilename)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    result = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        classId = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)
        result.append([classId,bb])
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return result


def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    filenames = os.listdir(DATASET_PATH + 'Annotations')
    for file in filenames:
        convert_annotation(file)

class DatasetOperationPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log=log
        self.datasetDir = None
        self.editDatasetBTN = wx.Button(self, -1, label="编辑数据集", size=(180, 35))
        self.datasetTree = DatasetTree(self, self.log, size=(180, 300))
        self.listDataPanel = ListDataPanel(self, self.log, [])
        self.middlePanel = PictureShowPanel(self, self.log, size=(630,-1))
        self.rightPanel = wx.Panel(self, -1, size=(300, -1), style=wx.BORDER_THEME)
        hbox = wx.BoxSizer()
        vvbox = wx.BoxSizer(wx.VERTICAL)
        vvbox.Add(self.editDatasetBTN, 0)
        vvbox.Add(self.datasetTree, 1)
        hbox.Add(vvbox, 0,wx.EXPAND)
        hbox.Add(self.listDataPanel, 0, wx.EXPAND)
        hbox.Add(self.middlePanel, 1, wx.EXPAND)
        hbox.Add(self.rightPanel, 0, wx.EXPAND)
        self.SetSizer(hbox)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnUpdateDatasetTree)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnDatasetListSelectionChanged, self.listDataPanel.list)

    def OnUpdateDatasetTree(self,event):
        item=self.datasetTree.tree.GetFocusedItem()
        itemdata=self.datasetTree.tree.GetItemData(item)
        itemtext=self.datasetTree.tree.GetItemText(item)
        if '孙集' in itemdata:
            self.datasetDir = itemdata[2:]
            data = os.listdir(self.datasetDir)
            self.listDataPanel.ReCreate(data)

    def OnDatasetListSelectionChanged(self,event):
        currentItem = event.Index
        self.middlePanel.filename = self.datasetDir+ "\\" + self.listDataPanel.list.GetItemText(currentItem)
        annotationFilename = self.datasetDir + "\\..\\" + "Annotations\\" + self.listDataPanel.list.GetItemText(currentItem)
        annotationFilename = annotationFilename[:-3] + "xml"
        self.middlePanel.labelList = GetAnnotationInfo(annotationFilename)
        self.middlePanel.Refresh()



