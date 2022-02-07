import cv2
import wx
from ID_DEFINE import *
class PictureShowPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.log=log
        self.filename = None
        self.labelList = None
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, evt):
        if self.filename:
            dc = wx.PaintDC(self)
            self.img = cv2.imread(self.filename)
            self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
            self.width,self.height = self.img.shape[1],self.img.shape[0]
            for label in self.labelList:
                pt1 = (int(label[1][0] * self.width - label[1][2] * self.width / 2), int(label[1][1] * self.height - label[1][3] * self.height /2))
                pt2 = (int(label[1][0] * self.width + label[1][2] * self.width / 2), int(label[1][1] * self.height + label[1][3] * self.height /2))
                cv2.putText(self.img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                cv2.rectangle(self.img, pt1, pt2, (0, 0, 255, 2))
            x,y = self.GetClientSize()
            bmp = wx.Image(self.width, self.height, self.img).Scale(width=x, height=y,
                                                      quality=wx.IMAGE_QUALITY_BOX_AVERAGE).ConvertToBitmap()
            dc.DrawBitmap(bmp, 0, 0, True)
        evt.Skip()

