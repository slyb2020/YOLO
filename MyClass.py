#!/usr/bin/env python
# encoding: utf-8
"""
@author: slyb
@license: (C) Copyright 2017-2020, 天津定智科技有限公司.
@contact: slyb@tju.edu.cn
@file: MyClass.py.py
@time: 2019/6/16 14:05
@desc:
"""
from MyLog import MyLogCtrl
import wx.lib.agw.pybusyinfo as PBI
import time
import wx.lib.agw.aquabutton as AB
import wx.lib.agw.gradientbutton as GB
# from OrderInfoEditDialog import *
# from ScheduleDemoDialog import *
import wx
import wx.adv
import wx.lib.agw.foldpanelbar as fpb
import wx.lib.gizmos as gizmos  # Formerly wx.gizmos in Classic
from six import BytesIO
import images
from ID_DEFINE import *
import math
import random
import os
import sys
import images
import wx.lib.agw.hypertreelist as HTL
import random

import wx.lib.agw.flatmenu as FM
from wx.lib.agw.artmanager import ArtManager, RendererBase, DCSaver
from wx.lib.agw.fmresources import ControlFocus, ControlPressed
from wx.lib.agw.fmresources import FM_OPT_SHOW_CUSTOMIZE, FM_OPT_SHOW_TOOLBAR, FM_OPT_MINIBAR
import datetime
from DatasetOperationPanel import DatasetOperationPanel
from YOLOv1Panel import YOLOv1Panel
from LeNetPanel import LeNetPanel

dirName = os.path.dirname(os.path.abspath(__file__))
bitmapDir = os.path.join(dirName, 'bitmaps')
sys.path.append(os.path.split(dirName)[0])


def switchRGBtoBGR(colour):
    return wx.Colour(colour.Blue(), colour.Green(), colour.Red())


def CreateBackgroundBitmap():
    mem_dc = wx.MemoryDC()
    bmp = wx.Bitmap(200, 300)
    mem_dc.SelectObject(bmp)
    mem_dc.Clear()
    # colour the menu face with background colour
    top = wx.Colour("blue")
    bottom = wx.Colour("light blue")
    filRect = wx.Rect(0, 0, 200, 300)
    mem_dc.GradientFillConcentric(filRect, top, bottom, wx.Point(100, 150))
    mem_dc.SelectObject(wx.NullBitmap)
    return bmp


class FM_MyRenderer(FM.FMRenderer):
    def __init__(self):
        FM.FMRenderer.__init__(self)

    def DrawMenuButton(self, dc, rect, state):
        self.DrawButton(dc, rect, state)

    def DrawMenuBarButton(self, dc, rect, state):
        self.DrawButton(dc, rect, state)

    def DrawButton(self, dc, rect, state, colour=None):
        if state == ControlFocus:
            penColour = switchRGBtoBGR(ArtManager.Get().FrameColour())
            brushColour = switchRGBtoBGR(ArtManager.Get().BackgroundColour())
        elif state == ControlPressed:
            penColour = switchRGBtoBGR(ArtManager.Get().FrameColour())
            brushColour = switchRGBtoBGR(ArtManager.Get().HighlightBackgroundColour())
        else:  # ControlNormal, ControlDisabled, default
            penColour = switchRGBtoBGR(ArtManager.Get().FrameColour())
            brushColour = switchRGBtoBGR(ArtManager.Get().BackgroundColour())
        dc.SetPen(wx.Pen(penColour))
        dc.SetBrush(wx.Brush(brushColour))
        dc.DrawRoundedRectangle(rect.x, rect.y, rect.width, rect.height, 4)

    def DrawMenuBarBackground(self, dc, rect):
        vertical = ArtManager.Get().GetMBVerticalGradient()
        dcsaver = DCSaver(dc)
        # fill with gradient
        startColour = self.menuBarFaceColour
        endColour = ArtManager.Get().LightColour(startColour, 90)
        dc.SetPen(wx.Pen(endColour))
        dc.SetBrush(wx.Brush(endColour))
        dc.DrawRectangle(rect)

    def DrawToolBarBg(self, dc, rect):
        if not ArtManager.Get().GetRaiseToolbar():
            return
        # fill with gradient
        startColour = self.menuBarFaceColour()
        dc.SetPen(wx.Pen(startColour))
        dc.SetBrush(wx.Brush(startColour))
        dc.DrawRectangle(0, 0, rect.GetWidth(), rect.GetHeight())


ArtIDs = ["None",
          "wx.ART_ADD_BOOKMARK",
          "wx.ART_DEL_BOOKMARK",
          "wx.ART_HELP_SIDE_PANEL",
          "wx.ART_HELP_SETTINGS",
          "wx.ART_HELP_BOOK",
          "wx.ART_HELP_FOLDER",
          "wx.ART_HELP_PAGE",
          "wx.ART_GO_BACK",
          "wx.ART_GO_FORWARD",
          "wx.ART_GO_UP",
          "wx.ART_GO_DOWN",
          "wx.ART_GO_TO_PARENT",
          "wx.ART_GO_HOME",
          "wx.ART_FILE_OPEN",
          "wx.ART_PRINT",
          "wx.ART_HELP",
          "wx.ART_TIP",
          "wx.ART_REPORT_VIEW",
          "wx.ART_LIST_VIEW",
          "wx.ART_NEW_DIR",
          "wx.ART_HARDDISK",
          "wx.ART_FLOPPY",
          "wx.ART_CDROM",
          "wx.ART_REMOVABLE",
          "wx.ART_FOLDER",
          "wx.ART_FOLDER_OPEN",
          "wx.ART_GO_DIR_UP",
          "wx.ART_EXECUTABLE_FILE",
          "wx.ART_NORMAL_FILE",
          "wx.ART_TICK_MARK",
          "wx.ART_CROSS_MARK",
          "wx.ART_ERROR",
          "wx.ART_QUESTION",
          "wx.ART_WARNING",
          "wx.ART_INFORMATION",
          "wx.ART_MISSING_IMAGE",
          "SmileBitmap"
          ]


##########################################################################
def GetCollapsedIconData():
    return \
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\
\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\
\x00\x01\x8eIDAT8\x8d\xa5\x93-n\xe4@\x10\x85?g\x03\n6lh)\xc4\xd2\x12\xc3\x81\
\xd6\xa2I\x90\x154\xb9\x81\x8f1G\xc8\x11\x16\x86\xcd\xa0\x99F\xb3A\x91\xa1\
\xc9J&\x96L"5lX\xcc\x0bl\xf7v\xb2\x7fZ\xa5\x98\xebU\xbdz\xf5\\\x9deW\x9f\xf8\
H\\\xbfO|{y\x9dT\x15P\x04\x01\x01UPUD\x84\xdb/7YZ\x9f\xa5\n\xce\x97aRU\x8a\
\xdc`\xacA\x00\x04P\xf0!0\xf6\x81\xa0\xf0p\xff9\xfb\x85\xe0|\x19&T)K\x8b\x18\
\xf9\xa3\xe4\xbe\xf3\x8c^#\xc9\xd5\n\xa8*\xc5?\x9a\x01\x8a\xd2b\r\x1cN\xc3\
\x14\t\xce\x97a\xb2F0Ks\xd58\xaa\xc6\xc5\xa6\xf7\xdfya\xe7\xbdR\x13M2\xf9\
\xf9qKQ\x1fi\xf6-\x00~T\xfac\x1dq#\x82,\xe5q\x05\x91D\xba@\xefj\xba1\xf0\xdc\
zzW\xcff&\xb8,\x89\xa8@Q\xd6\xaaf\xdfRm,\xee\xb1BDxr#\xae\xf5|\xddo\xd6\xe2H\
\x18\x15\x84\xa0q@]\xe54\x8d\xa3\xedf\x05M\xe3\xd8Uy\xc4\x15\x8d\xf5\xd7\x8b\
~\x82\x0fh\x0e"\xb0\xad,\xee\xb8c\xbb\x18\xe7\x8e;6\xa5\x89\x04\xde\xff\x1c\
\x16\xef\xe0p\xfa>\x19\x11\xca\x8d\x8d\xe0\x93\x1b\x01\xd8m\xf3(;x\xa5\xef=\
\xb7w\xf3\x1d$\x7f\xc1\xe0\xbd\xa7\xeb\xa0(,"Kc\x12\xc1+\xfd\xe8\tI\xee\xed)\
\xbf\xbcN\xc1{D\x04k\x05#\x12\xfd\xf2a\xde[\x81\x87\xbb\xdf\x9cr\x1a\x87\xd3\
0)\xba>\x83\xd5\xb97o\xe0\xaf\x04\xff\x13?\x00\xd2\xfb\xa9`z\xac\x80w\x00\
\x00\x00\x00IEND\xaeB`\x82'


def GetCollapsedIconBitmap():
    return wx.Bitmap(GetCollapsedIconImage())


def GetCollapsedIconImage():
    stream = BytesIO(GetCollapsedIconData())
    return wx.Image(stream)


# ----------------------------------------------------------------------
def GetExpandedIconData():
    return \
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\
\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\
\x00\x01\x9fIDAT8\x8d\x95\x93\xa1\x8e\xdc0\x14EO\xb2\xc4\xd0\xd2\x12\xb7(mI\
\xa4%V\xd1lQT4[4-\x9a\xfe\xc1\xc2|\xc6\xc2~BY\x83:A3E\xd3\xa0*\xa4\xd2\x90H!\
\x95\x0c\r\r\x1fK\x81g\xb2\x99\x84\xb4\x0fY\xd6\xbb\xc7\xf7>=\'Iz\xc3\xbcv\
\xfbn\xb8\x9c\x15 \xe7\xf3\xc7\x0fw\xc9\xbc7\x99\x03\x0e\xfbn0\x99F+\x85R\
\x80RH\x10\x82\x08\xde\x05\x1ef\x90+\xc0\xe1\xd8\ryn\xd0Z-\\A\xb4\xd2\xf7\
\x9e\xfbwoF\xc8\x088\x1c\xbbae\xb3\xe8y&\x9a\xdf\xf5\xbd\xe7\xfem\x84\xa4\
\x97\xccYf\x16\x8d\xdb\xb2a]\xfeX\x18\xc9s\xc3\xe1\x18\xe7\x94\x12cb\xcc\xb5\
\xfa\xb1l8\xf5\x01\xe7\x84\xc7\xb2Y@\xb2\xcc0\x02\xb4\x9a\x88%\xbe\xdc\xb4\
\x9e\xb6Zs\xaa74\xadg[6\x88<\xb7]\xc6\x14\x1dL\x86\xe6\x83\xa0\x81\xba\xda\
\x10\x02x/\xd4\xd5\x06\r\x840!\x9c\x1fM\x92\xf4\x86\x9f\xbf\xfe\x0c\xd6\x9ae\
\xd6u\x8d \xf4\xf5\x165\x9b\x8f\x04\xe1\xc5\xcb\xdb$\x05\x90\xa97@\x04lQas\
\xcd*7\x14\xdb\x9aY\xcb\xb8\\\xe9E\x10|\xbc\xf2^\xb0E\x85\xc95_\x9f\n\xaa/\
\x05\x10\x81\xce\xc9\xa8\xf6><G\xd8\xed\xbbA)X\xd9\x0c\x01\x9a\xc6Q\x14\xd9h\
[\x04\xda\xd6c\xadFkE\xf0\xc2\xab\xd7\xb7\xc9\x08\x00\xf8\xf6\xbd\x1b\x8cQ\
\xd8|\xb9\x0f\xd3\x9a\x8a\xc7\x08\x00\x9f?\xdd%\xde\x07\xda\x93\xc3{\x19C\
\x8a\x9c\x03\x0b8\x17\xe8\x9d\xbf\x02.>\x13\xc0n\xff{PJ\xc5\xfdP\x11""<\xbc\
\xff\x87\xdf\xf8\xbf\xf5\x17FF\xaf\x8f\x8b\xd3\xe6K\x00\x00\x00\x00IEND\xaeB\
`\x82'


def GetExpandedIconBitmap():
    return wx.Bitmap(GetExpandedIconImage())


def GetExpandedIconImage():
    stream = BytesIO(GetExpandedIconData())
    return wx.Image(stream)


# ----------------------------------------------------------------------
def GetMondrianData():
    return \
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 \x08\x06\x00\
\x00\x00szz\xf4\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\x00\x00qID\
ATX\x85\xed\xd6;\n\x800\x10E\xd1{\xc5\x8d\xb9r\x97\x16\x0b\xad$\x8a\x82:\x16\
o\xda\x84pB2\x1f\x81Fa\x8c\x9c\x08\x04Z{\xcf\xa72\xbcv\xfa\xc5\x08 \x80r\x80\
\xfc\xa2\x0e\x1c\xe4\xba\xfaX\x1d\xd0\xde]S\x07\x02\xd8>\xe1wa-`\x9fQ\xe9\
\x86\x01\x04\x10\x00\\(Dk\x1b-\x04\xdc\x1d\x07\x14\x98;\x0bS\x7f\x7f\xf9\x13\
\x04\x10@\xf9X\xbe\x00\xc9 \x14K\xc1<={\x00\x00\x00\x00IEND\xaeB`\x82'


def GetMondrianBitmap():
    return wx.Bitmap(GetMondrianImage())


def GetMondrianImage():
    stream = BytesIO(GetMondrianData())
    return wx.Image(stream)


def GetMondrianIcon():
    icon = wx.Icon()
    icon.CopyFromBitmap(GetMondrianBitmap())
    return icon


class MainPanel(wx.Panel):
    def __init__(self, parent, id=wx.ID_ANY, title="", pos=wx.DefaultPosition,
                 size=(1024, 768), style=wx.TAB_TRAVERSAL):
        wx.Panel.__init__(self, parent, id, pos, size, style)
        self.parent = parent

        il = wx.ImageList(16, 16)
        self.idx1 = il.Add(images._rt_smiley.GetBitmap())
        self.idx2 = il.Add(images.GridBG.GetBitmap())
        self.idx3 = il.Add(images.Smiles.GetBitmap())
        self.idx4 = il.Add(images._rt_undo.GetBitmap())
        self.idx5 = il.Add(images._rt_save.GetBitmap())
        self.idx6 = il.Add(images._rt_redo.GetBitmap())

        self._leftWindow1 = wx.adv.SashLayoutWindow(self, ID_WINDOW_LEFT, wx.DefaultPosition,
                                                    wx.Size(200, 1000), wx.NO_BORDER |
                                                    wx.adv.SW_3D | wx.CLIP_CHILDREN)
        self._leftWindow1.SetDefaultSize(wx.Size(220, 1000))
        self._leftWindow1.SetOrientation(wx.adv.LAYOUT_VERTICAL)
        self._leftWindow1.SetAlignment(wx.adv.LAYOUT_LEFT)
        self._leftWindow1.SetSashVisible(wx.adv.SASH_RIGHT, True)
        self._leftWindow1.SetExtraBorderSize(10)
        self._pnl = 0
        # will occupy the space not used by the Layout Algorithm
        self.CreateBottomWindow()
        self.log = MyLogCtrl(self.bottomWindow, -1, "")
        self.work_zone_Panel = WorkZonePanel(self, self, self.log)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.ReCreateFoldPanel(0)
        self.Bind(wx.adv.EVT_SASH_DRAGGED_RANGE, self.OnSashDrag, id=ID_WINDOW_LEFT,
                  id2=ID_WINDOW_BOTTOM)  # BOTTOM和LEFT顺序不能换，要想更改哪个先分，只需更改上面窗口定义的顺序

    def CreateBottomWindow(self):
        self.bottomWindow = wx.adv.SashLayoutWindow(self, ID_WINDOW_BOTTOM, style=wx.NO_BORDER | wx.adv.SW_3D)
        self.bottomWindow.SetDefaultSize((1000, 200))
        self.bottomWindow.SetOrientation(wx.adv.LAYOUT_HORIZONTAL)
        self.bottomWindow.SetAlignment(wx.adv.LAYOUT_BOTTOM)
        # win.SetBackgroundColour(wx.Colour(0, 0, 255))
        self.bottomWindow.SetSashVisible(wx.adv.SASH_TOP, True)
        self.bottomWindow.SetExtraBorderSize(5)
        self.Bind(wx.EVT_BUTTON, self.OnButton)

    def OnButton(self, event):
        id = event.GetId()
        if id == ID_DATASET_BTN:
            self.work_zone_Panel.notebook.SetSelection(0)
        if id == ID_YOLOv1_BTN:
            self.work_zone_Panel.notebook.SetSelection(1)
        if id == ID_LeNet_BTN:
            self.work_zone_Panel.notebook.SetSelection(2)

    def OnSize(self, event):
        wx.adv.LayoutAlgorithm().LayoutWindow(self, self.work_zone_Panel)
        event.Skip()

    def OnSashDrag(self, event):
        if event.GetDragStatus() == wx.adv.SASH_STATUS_OUT_OF_RANGE:
            return
        eID = event.GetId()
        if eID == ID_WINDOW_LEFT:
            self._leftWindow1.SetDefaultSize((event.GetDragRect().width, 1000))
        elif eID == ID_WINDOW_BOTTOM:
            self.bottomWindow.SetDefaultSize((1000, event.GetDragRect().height))
        wx.adv.LayoutAlgorithm().LayoutWindow(self, self.work_zone_Panel)
        self.work_zone_Panel.Refresh()

    def ReCreateFoldPanel(self, fpb_flags, state=0):
        # delete earlier panel
        self._leftWindow1.DestroyChildren()
        self._pnl = fpb.FoldPanelBar(self._leftWindow1, -1, wx.DefaultPosition,
                                     wx.Size(-1, -1), agwStyle=fpb_flags)
        Images = wx.ImageList(16, 16)
        Images.Add(GetExpandedIconBitmap())
        Images.Add(GetCollapsedIconBitmap())

        item = self._pnl.AddFoldPanel("工具面板", collapsed=False,
                                      foldIcons=Images)
        panel = wx.Panel(item, -1, size=(300, 300))
        bitmap = wx.Bitmap("bitmaps/aquabutton.png",
                           wx.BITMAP_TYPE_PNG)
        self.datasetOperationrBTN = AB.AquaButton(panel, ID_DATASET_BTN, bitmap, "数据集操作", size=(100, 50))
        self.datasetOperationrBTN.SetForegroundColour(wx.BLACK)
        self.datasetOperationrBTN.Enable(True)
        self.YOLOv1BTN = AB.AquaButton(panel, ID_YOLOv1_BTN, bitmap, "YOLOv1", size=(100, 50))
        self.YOLOv1BTN.SetForegroundColour(wx.BLACK)
        self.LeNetBTN = AB.AquaButton(panel, ID_LeNet_BTN, bitmap, "LeNet", size=(100, 50))
        self.LeNetBTN.SetForegroundColour(wx.BLACK)
        static = wx.StaticLine(panel, -1)
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.datasetOperationrBTN, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        vbox.Add(self.YOLOv1BTN, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        vbox.Add(self.LeNetBTN, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        vbox.Add(static, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        panel.SetSizer(vbox)
        self._pnl.AddFoldPanelWindow(item, panel, fpb.FPB_ALIGN_WIDTH, 5, 0)
        self._leftWindow1.SizeWindows()


class WorkZonePanel(wx.Panel):
    def __init__(self, parent, master, log):
        wx.Panel.__init__(self, parent, -1)
        self.master = master
        self.log = log
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
        wx.BK_DEFAULT
                                    # wx.BK_TOP
                                    # wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    # | wx.NB_MULTILINE
                                    )
        il = wx.ImageList(16, 16)
        idx1 = il.Add(images._rt_smiley.GetBitmap())
        self.total_page_num = 0
        self.notebook.AssignImageList(il)
        idx2 = il.Add(images.GridBG.GetBitmap())
        idx3 = il.Add(images.Smiles.GetBitmap())
        idx4 = il.Add(images._rt_undo.GetBitmap())
        idx5 = il.Add(images._rt_save.GetBitmap())
        idx6 = il.Add(images._rt_redo.GetBitmap())
        hbox = wx.BoxSizer()
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.datasetOperationPNL = DatasetOperationPanel(self.notebook, self.log)
        self.notebook.AddPage(self.datasetOperationPNL, "数据集操作面板")
        self.yolov1PNL = YOLOv1Panel(self.notebook, self.log)
        self.notebook.AddPage(self.yolov1PNL, "YOLOv1面板")
        self.leNetPNL = LeNetPanel(self.notebook, self.log)
        self.notebook.AddPage(self.leNetPNL, "LeNet面板")
        self.SetSizer(hbox)
