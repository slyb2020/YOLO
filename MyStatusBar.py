#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import wx
from MyGauge import *
import time

RELATIVEWIDTHS = False


class MyStatusBar(wx.StatusBar):
    def __init__(self, parent):
        wx.StatusBar.__init__(self, parent, -1, style=wx.STB_SIZEGRIP)
        self.parent = parent
        self.timer_count = 0
        # This status bar has three fields
        self.SetFieldsCount(4)
        if RELATIVEWIDTHS:
            # Sets the three fields to be relative widths to each other.
            self.SetStatusWidths([-2, -1, -2, -2])
        else:
            self.SetStatusWidths([200, -2, 400, 140])
        self.DBOpen_Error = 0
        self.DBOpen_Error_Alarm_Enable = 1
        self.All_Alarm_Enable = 1
        self.sizeChanged = False
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.SetStatusText("基于YOLO的目标检测系统正在运行......", 1)
        self.SetStatusText("当前状态：未登录", 2)
        # self.gauge = MyProgressGauge(self,size=(55, 15))
        self.gauge = wx.Gauge(self, 100)
        self.Reposition()
        self.timer = wx.PyTimer(self.Notify)
        self.timer.Start(100)

    def __del__(self):
        self.timer.Stop()

    def Notify(self):
        self.timer_count += 1
        t = time.localtime(time.time())
        st = time.strftime("%Y-%m-%d %H:%M:%S", t)
        try:
            self.gauge.Pulse()
            self.SetStatusText(st, 3)
        except:
            pass

    def OnSize(self, evt):
        evt.Skip()
        self.Reposition()  # for normal size events
        # Set a flag so the idle time handler will also do the repositioning.
        # It is done this way to get around a buglet where GetFieldRect is not
        # accurate during the EVT_SIZE resulting from a frame maximize.
        self.sizeChanged = True

    def OnIdle(self, evt):
        if self.sizeChanged:
            self.Reposition()

    # reposition the checkbox
    def Reposition(self):
        # rect = self.GetFieldRect(2)
        # rect.x += 1
        # rect.y += 1
        # self.cb.SetRect(rect)
        rect = self.GetFieldRect(0)
        rect = (5, 4, 200, 18)
        self.gauge.SetRect(rect)
        self.sizeChanged = False
