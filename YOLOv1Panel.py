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
from YOLOv1Pytorch import *

def draw_bbox(img, bbox):
    """
    根据bbox的信息在图像上绘制bounding box
    :param img: 绘制bbox的图像
    :param bbox: 是(n,6)的尺寸，其中第0列代表bbox的分类序号，1~4为bbox坐标信息(xyxy)(均归一化了)，5是bbox的专属类别置信度
    """
    img = img.copy()
    h, w = img.shape[0:2]
    n = bbox.size()[0]
    for i in range(n):
        p1 = (w * bbox[i, 1], h * bbox[i, 2])
        p2 = (w * bbox[i, 3], h * bbox[i, 4])
        cls_name = CLASSES[int(bbox[i, 0])]
        confidence = bbox[i, 5]
        # p1 = p1.numpy()
        p1 = (int(p1[0].detach().numpy()), int(p1[1].detach().numpy()))
        p2 = (400, 400)
        cv2.rectangle(img, p1, p2, color=COLOR[int(bbox[i, 0])])
        cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # cv2.rectangle(img, (10,10), (100,100), (255,0,0), 2)
    cv2.imshow("bbox", img)
    cv2.waitKey(0)


def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    """bbox数据格式是(n,25),前4个是(x1,y1,x2,y2)的坐标信息，第5个是置信度，后20个是类别概率
    :param conf_thresh: cls-specific confidence score的阈值
    :param iou_thresh: NMS算法中iou的阈值
    """
    n = bbox.size()[0]
    bbox_prob = bbox[:, 5:].clone()  # 类别预测的条件概率
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)  # 预测置信度
    bbox_cls_spec_conf = bbox_confi * bbox_prob  # 置信度*类别条件概率=cls-specific confidence score整合了是否有物体及是什么物体的两种信息
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0  # 将低于阈值的bbox忽略
    for c in range(20):
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices
        for i in range(98):
            if bbox_cls_spec_conf[rank[i], c] != 0:
                for j in range(i + 1, 98):
                    if bbox_cls_spec_conf[rank[j], c] != 0:
                        iou = calculate_iou(bbox[rank[i], 0:4], bbox[rank[j], 0:4])
                        if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j], c] = 0
    bbox = bbox[torch.max(bbox_cls_spec_conf, dim=1).values > 0]  # 将20个类别中最大的cls-specific confidence score为0的bbox都排除
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf, dim=1).values > 0]
    res = torch.ones((bbox.size()[0], 6))
    res[:, 1:5] = bbox[:, 0:4]  # 储存最后的bbox坐标信息
    res[:, 0] = torch.argmax(bbox[:, 5:], dim=1).int()  # 储存bbox对应的类别信息
    res[:, 5] = torch.max(bbox_cls_spec_conf, dim=1).values  # 储存bbox对应的class-specific confidence scores
    return res


# 注意检查一下输入数据的格式，到底是xywh还是xyxy
def labels2bbox(matrix):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果
    """
    if matrix.size()[0:2] != (7, 7):
        raise ValueError("Error: Wrong labels size:", matrix.size())
    bbox = torch.zeros((98, 25))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):  # i是网格的行方向(y方向)
        for j in range(7):  # j是网格的列方向(x方向)
            bbox[2 * (i * 7 + j), 0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                       (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i, j, 4]
            bbox[2 * (i * 7 + j), 5:] = matrix[i, j, 10:]
            bbox[2 * (i * 7 + j) + 1, 0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                           (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j) + 1, 4] = matrix[i, j, 9]
            bbox[2 * (i * 7 + j) + 1, 5:] = matrix[i, j, 10:]
    return bbox
    # return NMS(bbox)  # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox


def ChangeImageSize(img, inputSize):
    h, w = img.shape[0:2]
    # 输入YOLOv1网络的图像尺寸为448x448
    # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
    # 然后再将Padding后的正方形图像缩放成448x448
    padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
    if h > w:
        padw = (h - w) // 2
        img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
    elif w > h:
        padh = (w - h) // 2
        img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.resize(img, (inputSize, inputSize))


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
                data.append(list(self.middlePanel.bbox[row * 7 + col].detach().numpy()))
                data.append(list(self.middlePanel.bbox[2 * (row * 7 + col)].detach().numpy()))
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
