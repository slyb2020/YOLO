import copy
import cv2
import torch
import wx
from ID_DEFINE import *
from PIL import Image
import numpy as np


class PictureShowPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.log = log
        self.filename = None
        self.labelList = None
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, evt):
        if self.filename:
            dc = wx.PaintDC(self)
            self.img = cv2.imread(self.filename)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.width, self.height = self.img.shape[1], self.img.shape[0]
            for label in self.labelList:
                pt1 = (int(label[1][0] * self.width - label[1][2] * self.width / 2),
                       int(label[1][1] * self.height - label[1][3] * self.height / 2))
                pt2 = (int(label[1][0] * self.width + label[1][2] * self.width / 2),
                       int(label[1][1] * self.height + label[1][3] * self.height / 2))
                cv2.putText(self.img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                cv2.rectangle(self.img, pt1, pt2, (0, 0, 255, 2))
            x, y = self.GetClientSize()
            bmp = wx.Image(self.width, self.height, self.img).Scale(width=x, height=y,
                                                                    quality=wx.IMAGE_QUALITY_BOX_AVERAGE).ConvertToBitmap()
            dc.DrawBitmap(bmp, 0, 0, True)
        evt.Skip()


class YOLOPictureShowPanel(wx.Panel):
    def __init__(self, parent, log, size, gap=False):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.log = log
        self.filename = None
        self.labelList = None
        self.gap = 1 if gap else 0
        self.bbox = None
        self.enablePicture = True
        self.enableLabel = True
        self.enableBox = True

        gbox = wx.GridSizer(7, 0, 0)
        self.buttonArray = []
        for i in range(7):
            row = []
            for j in range(7):
                button = wx.Button(self, id=ButtonIdArray[i][j])
                row.append(button)
                gbox.Add(button, 1, wx.EXPAND)
            self.buttonArray.append(row)
        self.SetSizer(gbox)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, evt):
        if self.filename:
            # dc = wx.PaintDC(self)
            self.img = cv2.imread(self.filename)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.width, self.height = self.img.shape[1], self.img.shape[0]
            shape = self.img.shape
            if not self.enablePicture:
                self.img = np.array(torch.zeros(size=shape, dtype=torch.uint8) + 255)
            if self.enableLabel:
                for label in self.labelList:
                    pt1 = (int(label[1][0] * self.width - label[1][2] * self.width / 2),
                           int(label[1][1] * self.height - label[1][3] * self.height / 2))
                    pt2 = (int(label[1][0] * self.width + label[1][2] * self.width / 2),
                           int(label[1][1] * self.height + label[1][3] * self.height / 2))
                    cv2.putText(self.img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                    cv2.rectangle(self.img, pt1, pt2, (0, 0, 255, 2))
            if self.enableBox:
                if self.BBOX != None:
                    h, w = self.img.shape[0:2]
                    n = self.BBOX.size()[0]
                    for i in range(n):
                        p1 = (self.width * self.BBOX[i, 1], self.height * self.BBOX[i, 2])
                        p2 = (self.width * self.BBOX[i, 3], self.height * self.BBOX[i, 4])
                        cls_name = CLASSES[int(self.BBOX[i, 0])]
                        confidence = self.BBOX[i, 5]
                        # p1 = p1.numpy()
                        p1 = (int(p1[0].detach().numpy()), int(p1[1].detach().numpy()))
                        p2 = (400, 400)
                        cv2.rectangle(self.img, p1, p2, color=COLOR[int(self.bbox[i, 0])])
                        cv2.putText(self.img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            x, y = self.img.shape[1] / len(self.buttonArray[0]), self.img.shape[0] / len(self.buttonArray)
            w, h = self.buttonArray[0][0].GetSize()
            for row, rowButtonList in enumerate(self.buttonArray):
                for col, button in enumerate(rowButtonList):
                    img = self.img[int(row * y):int(row * y + y), int(col * x):int(col * x + x), :]
                    img = np.array(img)  # np.array型数据经切片后，似乎不再是np.array类型了，所以需要强制转换一下，这里不转换会报错
                    a = img.shape[1]
                    b = img.shape[0]
                    bmp = wx.Image(a, b, img).Scale(width=w - self.gap, height=h - self.gap,
                                                    quality=wx.IMAGE_QUALITY_NORMAL).ConvertToBitmap()
                    button.SetBitmap(bmp)
        evt.Skip()


if __name__ == "__main__":
    imgArray = splitImage("./bitmaps/advancedsplash.png", 2, 2)
    print(imgArray[0].shape)
    cv2.imshow("图1", imgArray[0])
    cv2.waitKey(0)
