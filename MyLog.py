#!/usr/bin/env python
# encoding: utf-8
'''
@author: slyb
@license: (C) Copyright 2017-2020, 天津定智科技有限公司.
@contact: slyb@tju.edu.cn
@file: MyLog.py
@time: 2019/11/30 11:34
@desc:
'''
import wx
import time


class MyLogCtrl(wx.TextCtrl):  # 系统日志显示控件
    def __init__(self, parent, id=-1, title="", position=wx.Point(0, 0), size=wx.Size(150, 90),
                 style=wx.NO_BORDER | wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2):
        self.parent = parent
        wx.TextCtrl.__init__(self, parent, id, title, position, size, style)
        self.WriteText("系统开始运行。。。\r\n")
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDown)

    def OnLeftDown(self, evt):
        pass

    def SaveLogFile(self):
        t = time.localtime(time.time())
        filename = time.strftime("%Y%m%d%H.log", t)
        file = open(filename, 'w+')
        content = self.GetValue().encode('UTF-8')
        file.write(content)
        file.close()
        self.SetValue("")

    def WriteText(self, text, enable=True, font=wx.NORMAL_FONT, colour=wx.BLACK, bk_colour=wx.NullColour):
        import time
        if enable:
            if colour != wx.BLACK:
                wx.Bell()
            try:
                t = time.localtime(time.time())
                st = time.strftime("%Y-%m-%d %H:%M:%S  ", t)
                text = st + text
                # wx.TextCtrl.SetFont(self, font)
                # wx.TextCtrl.SetForegroundColour(self, colour)
                # wx.TextCtrl.SetBackgroundColour(self,backgroundcolour)
                start = self.GetLastPosition()
                wx.TextCtrl.WriteText(self, text)
                self.SetStyle(start, self.GetLastPosition(), wx.TextAttr(colour, bk_colour))
                self.ShowPosition(start)
            except:
                pass
