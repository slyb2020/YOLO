import cv2
import wx
from ID_DEFINE import *
import os
import torch
import torchvision.models as tvmodel
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as Loader
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # python的进度条模块
import wx.lib.scrolledpanel as scrolled
from ListDataPanel import ResultDisplayPanel
import wx.grid as gridlib


class SimpleGrid(gridlib.Grid):
    def __init__(self, parent, log):
        gridlib.Grid.__init__(self, parent, -1)
        self.log = log
        self.moveTo = None
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.CreateGrid(20, 4)
        self.SetRowLabelSize(0)
        self.SetColLabelSize(0)
        self.SetColSize(0, 60)
        self.SetColSize(1, 100)
        self.SetColSize(2, 60)
        self.SetColSize(3, 100)
        # self.SetRowSize(4, 45)
        self.SetCellValue(0, 0, "第一预测")
        self.SetCellValue(0, 2, "第二预测")
        self.SetCellValue(1, 0, "左上坐标")
        self.SetCellValue(1, 2, "左上坐标")
        self.SetCellValue(2, 0, "右下坐标")
        self.SetCellValue(2, 2, "右下坐标")
        for i in range(10):
            self.SetCellValue( i + 3,0, CLASSES2[2*i])
            self.SetCellValue( i + 3,2, CLASSES2[2*i+1])

    def ReCreate(self, data):
        # self.SetCellFont(0, 0, wx.Font(12, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
        self.SetCellValue(0, 1, str(data[0][4]))
        self.SetCellValue(0, 3, str(data[1][4]))
        self.SetCellValue(1, 1, "(%.3f,%.3f)" % (data[0][0],data[0][1]))
        self.SetCellValue(1, 3, "(%.3f,%.3f)" % (data[1][0],data[1][1]))
        self.SetCellValue(2, 1, "(%.3f,%.3f)" % (data[0][2],data[0][3]))
        self.SetCellValue(2, 3, "(%.3f,%.3f)" % (data[1][2],data[1][3]))
        for i in range(10):
            self.SetCellValue( i + 3, 1, "%.5f" % data[0][i * 2 + 5])
            self.SetCellValue( i + 3, 3, "%.5f" % data[0][i * 2 + 6])




        # self.SetCellTextColour(1, 1, wx.RED)
        # self.SetCellBackgroundColour(2, 2, wx.CYAN)
        # self.SetReadOnly(3, 3, True)
        #
        # self.SetCellEditor(5, 0, gridlib.GridCellNumberEditor(1, 1000))
        # self.SetCellValue(5, 0, "123")
        # self.SetCellEditor(6, 0, gridlib.GridCellFloatEditor())
        # self.SetCellValue(6, 0, "123.34")
        # self.SetCellEditor(7, 0, gridlib.GridCellNumberEditor())
        #
        # self.SetCellValue(6, 3, "You can veto editing this cell")
        #
        # # self.SetRowLabelSize(0)
        # # self.SetColLabelSize(0)
        #
        # # attribute objects let you keep a set of formatting values
        # # in one spot, and reuse them if needed
        # attr = gridlib.GridCellAttr()
        # attr.SetTextColour(wx.BLACK)
        # attr.SetBackgroundColour(wx.RED)
        # attr.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        #
        # # you can set cell attributes for the whole row (or column)
        # self.SetRowAttr(5, attr)
        #
        # # self.SetColLabelValue(0, "Custom")
        # # self.SetColLabelValue(1, "column")
        # # self.SetColLabelValue(2, "labels")
        # #
        # # self.SetColLabelAlignment(wx.ALIGN_LEFT, wx.ALIGN_BOTTOM)
        #
        # # self.SetDefaultCellOverflow(False)
        # # r = gridlib.GridCellAutoWrapStringRenderer()
        # # self.SetCellRenderer(9, 1, r)
        #
        # # overflow cells
        # self.SetCellValue(9, 1,
        #                   "This default cell will overflow into neighboring cells, but not if you turn overflow off.")
        # self.SetCellSize(11, 1, 3, 3)
        # self.SetCellAlignment(11, 1, wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
        # self.SetCellValue(11, 1, "This cell is set to span 3 rows and 3 columns")
        #
        # editor = gridlib.GridCellTextEditor()
        # editor.SetParameters('10')
        # # self.SetCellEditor(0, 4, editor)
        # # self.SetCellValue(0, 4, "Limited text")
        #
        # renderer = gridlib.GridCellAutoWrapStringRenderer()
        # self.SetCellRenderer(15, 0, renderer)
        # self.SetCellValue(15, 0, "The text in this cell will be rendered with word-wrapping")

        # test all the events
        # self.Bind(gridlib.EVT_GRID_CELL_LEFT_CLICK, self.OnCellLeftClick)
        # self.Bind(gridlib.EVT_GRID_CELL_RIGHT_CLICK, self.OnCellRightClick)
        # self.Bind(gridlib.EVT_GRID_CELL_LEFT_DCLICK, self.OnCellLeftDClick)
        # self.Bind(gridlib.EVT_GRID_CELL_RIGHT_DCLICK, self.OnCellRightDClick)
        #
        # self.Bind(gridlib.EVT_GRID_LABEL_LEFT_CLICK, self.OnLabelLeftClick)
        # self.Bind(gridlib.EVT_GRID_LABEL_RIGHT_CLICK, self.OnLabelRightClick)
        # self.Bind(gridlib.EVT_GRID_LABEL_LEFT_DCLICK, self.OnLabelLeftDClick)
        # self.Bind(gridlib.EVT_GRID_LABEL_RIGHT_DCLICK, self.OnLabelRightDClick)
        #
        # self.Bind(gridlib.EVT_GRID_COL_SORT, self.OnGridColSort)
        #
        # self.Bind(gridlib.EVT_GRID_ROW_SIZE, self.OnRowSize)
        # self.Bind(gridlib.EVT_GRID_COL_SIZE, self.OnColSize)
        #
        # self.Bind(gridlib.EVT_GRID_RANGE_SELECT, self.OnRangeSelect)
        # self.Bind(gridlib.EVT_GRID_CELL_CHANGED, self.OnCellChange)
        # self.Bind(gridlib.EVT_GRID_SELECT_CELL, self.OnSelectCell)
        #
        # self.Bind(gridlib.EVT_GRID_EDITOR_SHOWN, self.OnEditorShown)
        # self.Bind(gridlib.EVT_GRID_EDITOR_HIDDEN, self.OnEditorHidden)
        # self.Bind(gridlib.EVT_GRID_EDITOR_CREATED, self.OnEditorCreated)

    def OnCellLeftClick(self, evt):
        self.log.write("OnCellLeftClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnCellRightClick(self, evt):
        self.log.write("OnCellRightClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnCellLeftDClick(self, evt):
        self.log.write("OnCellLeftDClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnCellRightDClick(self, evt):
        self.log.write("OnCellRightDClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnLabelLeftClick(self, evt):
        self.log.write("OnLabelLeftClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnLabelRightClick(self, evt):
        self.log.write("OnLabelRightClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnLabelLeftDClick(self, evt):
        self.log.write("OnLabelLeftDClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnLabelRightDClick(self, evt):
        self.log.write("OnLabelRightDClick: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnGridColSort(self, evt):
        self.log.write("OnGridColSort: %s %s" % (evt.GetCol(), self.GetSortingColumn()))
        self.SetSortingColumn(evt.GetCol())

    def OnRowSize(self, evt):
        self.log.write("OnRowSize: row %d, %s\n" %
                       (evt.GetRowOrCol(), evt.GetPosition()))
        evt.Skip()

    def OnColSize(self, evt):
        self.log.write("OnColSize: col %d, %s\n" %
                       (evt.GetRowOrCol(), evt.GetPosition()))
        evt.Skip()

    def OnRangeSelect(self, evt):
        if evt.Selecting():
            msg = 'Selected'
        else:
            msg = 'Deselected'
        self.log.write("OnRangeSelect: %s  top-left %s, bottom-right %s\n" %
                       (msg, evt.GetTopLeftCoords(), evt.GetBottomRightCoords()))
        evt.Skip()

    def OnCellChange(self, evt):
        self.log.write("OnCellChange: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))

        # Show how to stay in a cell that has bad data.  We can't just
        # call SetGridCursor here since we are nested inside one so it
        # won't have any effect.  Instead, set coordinates to move to in
        # idle time.
        value = self.GetCellValue(evt.GetRow(), evt.GetCol())

        if value == 'no good':
            self.moveTo = evt.GetRow(), evt.GetCol()

    def OnIdle(self, evt):
        if self.moveTo is not None:
            self.SetGridCursor(self.moveTo[0], self.moveTo[1])
            self.moveTo = None

        evt.Skip()

    def OnSelectCell(self, evt):
        if evt.Selecting():
            msg = 'Selected'
        else:
            msg = 'Deselected'
        self.log.write("OnSelectCell: %s (%d,%d) %s\n" %
                       (msg, evt.GetRow(), evt.GetCol(), evt.GetPosition()))

        # Another way to stay in a cell that has a bad value...
        row = self.GetGridCursorRow()
        col = self.GetGridCursorCol()

        if self.IsCellEditControlEnabled():
            self.HideCellEditControl()
            self.DisableCellEditControl()

        value = self.GetCellValue(row, col)

        if value == 'no good 2':
            return  # cancels the cell selection

        evt.Skip()

    def OnEditorShown(self, evt):
        if evt.GetRow() == 6 and evt.GetCol() == 3 and \
                wx.MessageBox("Are you sure you wish to edit this cell?",
                              "Checking", wx.YES_NO) == wx.NO:
            evt.Veto()
            return

        self.log.write("OnEditorShown: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnEditorHidden(self, evt):
        if evt.GetRow() == 6 and evt.GetCol() == 3 and \
                wx.MessageBox("Are you sure you wish to  finish editing this cell?",
                              "Checking", wx.YES_NO) == wx.NO:
            evt.Veto()
            return

        self.log.write("OnEditorHidden: (%d,%d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetPosition()))
        evt.Skip()

    def OnEditorCreated(self, evt):
        self.log.write("OnEditorCreated: (%d, %d) %s\n" %
                       (evt.GetRow(), evt.GetCol(), evt.GetControl()))


class YOLOv1ControlPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.parent = parent
        self.log = log
        self.currentPosition = None
        self.SetBackgroundColour(wx.Colour(240, 240, 240))
        self.pretrainedModelList = os.listdir(modelsDir)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hhbox = wx.BoxSizer()
        vbox.Add((-1, 5))
        hhbox.Add((10, -1))
        hhbox.Add(wx.StaticText(self, label="选择预训练模型："), wx.TOP, 5)
        self.pretrainedModelCOMBO = wx.ComboBox(self, value=self.pretrainedModelList[-1], size=(200, -1),
                                                choices=self.pretrainedModelList)
        hhbox.Add(self.pretrainedModelCOMBO, 1, wx.RIGHT, 10)
        vbox.Add(hhbox)
        hhbox = wx.BoxSizer()
        hhbox.Add((8, -1))
        self.startDetectionBTN = wx.Button(self, label="开始检测", size=(100, 30))
        hhbox.Add(self.startDetectionBTN, 0)
        hhbox.Add((15, -1))
        hhbox.Add(wx.StaticText(self, label="检测用时："), wx.TOP, 5)
        self.spendTimeTXT = wx.TextCtrl(self, size=(100, -1))
        self.spendTimeTXT.Enable(False)
        hhbox.Add(self.spendTimeTXT, 0, wx.LEFT | wx.TOP, 3)
        vbox.Add(hhbox)
        vbox.Add((-1, 5))
        hhbox = wx.BoxSizer()
        hhbox.Add((10, -1))
        self.enablePicBTN = wx.ToggleButton(self, label="图片", size=(-1, 30))
        hhbox.Add(self.enablePicBTN, 1, wx.RIGHT, 5)
        self.enableLabelBTN = wx.ToggleButton(self, label="标签", size=(-1, 30))
        hhbox.Add(self.enableLabelBTN, 1, wx.RIGHT, 5)
        self.enableBoxBTN = wx.ToggleButton(self, label="检测结果", size=(-1, 30))
        hhbox.Add(self.enableBoxBTN, 1, wx.RIGHT, 5)
        vbox.Add(hhbox, 0, wx.EXPAND)
        vbox.Add((-1, 3))
        vbox.Add(wx.StaticLine(self, style=wx.HORIZONTAL), 0, wx.EXPAND)
        vbox.Add((-1, 3))
        hhbox = wx.BoxSizer()
        hhbox.Add((10, -1))
        if self.currentPosition:
            self.label = wx.StaticText(self, label="第%d行，第%d列BOX数据：" % (
                self.currentPosition[0] + 1, self.currentPosition[1] + 1))
        else:
            self.label = wx.StaticText(self, label="第 行，第 列BOX数据：")
        hhbox.Add(self.label, 0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        # self.resultDisplayPanel = ResultDisplayPanel(self, self.log)
        self.resultDisplayPanel = SimpleGrid(self, self.log)
        vbox.Add(self.resultDisplayPanel, 1, wx.EXPAND)
        self.SetSizer(vbox)


# 发现一个问题，使用torch.load()方法，只有在main里能用，在子程序中使用会报错
if __name__ == "__main__":
    model = torch.load("D:\\WorkSpace\\Python\\YOLO\\Models\\YOLOv1_epoch10.pkl")
    print(model)
