#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
from MyClass import *
from ID_DEFINE import *
import torch.nn as nn
from YOLOv1Pytorch import *
from YOLOv1Pytorch2 import *


VERSION_STRING = "20220206A"

class FlatMenuFrame(wx.Frame):
    def __init__(self, parent):
        # 如果要初始运行时最大化可以或上wx.MAXIMIZE
        wx.Frame.__init__(self, parent, size=(1500, 900), style=wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE)
        self.SetIcon(images.Mondrian.GetIcon())
        self.SetTitle("基于YOLO的目标检测系统   Version——0.%s" % VERSION_STRING)
        self._popUpMenu = None
        self.check_in_flag = False
        self.timer_count = 0
        self.mouse_position = wx.Point()
        self.operator_ID = ''
        self.operator_name = ''
        self.operator_role = 0
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        # Create a main panel and place some controls on it
        from MyClass import MainPanel
        if self.check_in_flag:
            self.mainPANEL = MainPanel(self, wx.ID_ANY)
        else:
            self.mainPANEL = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        from MyStatusBar import MyStatusBar
        self.statusbar = MyStatusBar(self)
        self.SetStatusBar(self.statusbar)
        self.CreateMenu()
        self.ConnectEvents()
        mainSizer.Add(self._mb, 0, wx.EXPAND)
        mainSizer.Add(self.mainPANEL, 1, wx.EXPAND)
        self.SetSizer(mainSizer)
        mainSizer.Layout()
        ArtManager.Get().SetMBVerticalGradient(True)
        ArtManager.Get().SetRaiseToolbar(False)
        self._mb.Refresh()
        self.CenterOnScreen()
        self._mb.GetRendererManager().SetTheme(FM.StyleVista)
        pswd_found = True
        if pswd_found:
            self.check_in_flag = True
            self.operator_name = ""
            self.statusbar.SetStatusText("当前状态：%s 已登录  " % self.operator_name, 2)
            self.UpdateMainUI()

    def UpdateMainUI(self):
        self._mb.Destroy()
        self.mainPANEL.Destroy()
        self.CreateMenu()
        if self.check_in_flag:
            self.mainPANEL = MainPanel(self, wx.ID_ANY)
        else:
            self.mainPANEL = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(self._mb, 0, wx.EXPAND)
        mainSizer.Add(self.mainPANEL, 1, wx.EXPAND)
        self.SetSizer(mainSizer)
        self.Layout()

    def CreateMenu(self):
        # Create the menubar
        self._mb = FM.FlatMenuBar(self, wx.ID_ANY, 48, 5, options=FM_OPT_SHOW_TOOLBAR)
        fileMenu = FM.FlatMenu()
        fileMenuOut = FM.FlatMenu()
        setupMenu = FM.FlatMenu()
        helpMenu = FM.FlatMenu()
        subMenuExit = FM.FlatMenu()
        self.newMyTheme = self._mb.GetRendererManager().AddRenderer(FM_MyRenderer())
        new_file_bmp = wx.Bitmap("bitmaps/filenew.png", wx.BITMAP_TYPE_PNG)
        view1Bmp = wx.Bitmap("bitmaps/sunling3.png", wx.BITMAP_TYPE_PNG)
        view2Bmp = wx.Bitmap("bitmaps/lbadd.png", wx.BITMAP_TYPE_PNG)
        view3Bmp = wx.Bitmap("bitmaps/lbcharge.png", wx.BITMAP_TYPE_PNG)
        view4Bmp = wx.Bitmap("bitmaps/filesave.png", wx.BITMAP_TYPE_PNG)
        contractBmp = wx.Bitmap("bitmaps/33.png", wx.BITMAP_TYPE_PNG)
        order1Bmp = wx.Bitmap("bitmaps/locked.png", wx.BITMAP_TYPE_PNG)
        order2Bmp = wx.Bitmap("bitmaps/opened.png", wx.BITMAP_TYPE_PNG)
        order3Bmp = wx.Bitmap("bitmaps/order3.png", wx.BITMAP_TYPE_PNG)

        # Set an icon to the exit/help/transparency menu item
        exitImg = wx.Bitmap("bitmaps/exit-16.png", wx.BITMAP_TYPE_PNG)
        helpImg = wx.Bitmap("bitmaps/help-16.png", wx.BITMAP_TYPE_PNG)
        ghostBmp = wx.Bitmap("bitmaps/field-16.png", wx.BITMAP_TYPE_PNG)

        # Create the menu items
        item = FM.FlatMenuItem(fileMenu, MENU_CHECK_IN, "&R 登录系统...\tCtrl+R", "登录系统", wx.ITEM_NORMAL)
        fileMenuOut.AppendItem(item)
        item = FM.FlatMenuItem(fileMenu, MENU_CHECK_OUT, "&R 注销...\tCtrl+R", "登录系统", wx.ITEM_NORMAL)
        fileMenu.AppendItem(item)

        if self.check_in_flag:
            self._mb.AddTool(MENU_NEW_FILE, u"新建订单", view1Bmp)
            self._mb.AddTool(MENU_CHECK_OUT, u"注销...", view2Bmp)
        self._mb.AddSeparator()  # Separator

        self._mb.AddRadioTool(wx.ID_ANY, "View Details", view3Bmp)
        self._mb.AddRadioTool(wx.ID_ANY, "View Details", view4Bmp)
        self._mb.AddRadioTool(wx.ID_ANY, "View Multicolumn", contractBmp)
        self._mb.AddRadioTool(wx.ID_ANY, "View Multicolumn", order1Bmp)
        self._mb.AddRadioTool(wx.ID_ANY, "View Multicolumn", order2Bmp)
        self._mb.AddRadioTool(wx.ID_ANY, "View Multicolumn", order3Bmp)

        # Add non-toolbar item
        item = FM.FlatMenuItem(subMenuExit, wx.ID_EXIT, "E&xit\tAlt+X", "Exit demo", wx.ITEM_NORMAL, None, exitImg)
        subMenuExit.AppendItem(item)
        fileMenu.AppendSeparator()
        item = FM.FlatMenuItem(subMenuExit, wx.ID_EXIT, "E&xit\tAlt+Q", "Exit demo", wx.ITEM_NORMAL, None, exitImg)
        fileMenu.AppendItem(item)
        fileMenuOut.AppendItem(item)

        item = FM.FlatMenuItem(helpMenu, MENU_HELP, "&A关于\tCtrl+H", "关于...", wx.ITEM_NORMAL, None, helpImg)
        helpMenu.AppendItem(item)

        fileMenu.SetBackgroundBitmap(CreateBackgroundBitmap())

        # Add menu to the menu bar
        if self.check_in_flag:
            self._mb.Append(fileMenu, "&F 文件")
            self._mb.Append(setupMenu, "&O 系统参数设置")
            self._mb.Append(helpMenu, "&H 帮助")
        else:
            self._mb.Append(fileMenuOut, "&F 文件")
            self._mb.Append(helpMenu, "&H 帮助")

    def ConnectEvents(self):
        # Attach menu events to some handlers
        self.Bind(FM.EVT_FLAT_MENU_SELECTED, self.OnQuit, id=wx.ID_EXIT)
        self.Bind(FM.EVT_FLAT_MENU_SELECTED, self.OnAbout, id=MENU_HELP)

    def UpdateMenuState(self):
        self._mb.FindMenuItem(MENU_CHECK_IN).Enable(not self.check_in_flag)
        self._mb.FindMenuItem(MENU_CHECK_OUT).Enable(self.check_in_flag)

    def OnSize(self, event):
        self._mgr.Update()
        self.Layout()

    def OnQuit(self, event):
        self.Destroy()

    def OnAbout(self, event):
        msg = "YOLOv1系统\n\n" + \
              "天津大学精仪四室 版权所有 2021——2029\n\n" + \
              "\n" + \
              "如发现问题请联系:\n\n" + \
              "slyb@tju.edu.cn\n\n" + \
              "版本： 0." + VERSION_STRING
        dlg = wx.MessageDialog(self, msg, "关于",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()


def main():
    app = wx.App()
    win = FlatMenuFrame(None)
    win.Show()
    win.Center(wx.BOTH)
    app.MainLoop()


if __name__ == '__main__':
    __name__ = 'Main'
    main()
