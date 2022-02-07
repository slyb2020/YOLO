#!/usr/bin/env python
# encoding: utf-8
'''
@author: slyb
@license: (C) Copyright 2017-2020, 天津定智科技有限公司.
@contact: slyb@tju.edu.cn
@file: EditStaffInfoPanel.py
@time: 2019/9/2 9:18
@desc:
'''
import sys
import wx
import wx.lib.mixins.listctrl as listmix
import images
class ListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):
    def __init__(self, parent, ID, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.ListCtrlAutoWidthMixin.__init__(self)
class ListDataPanel(wx.Panel, listmix.ColumnSorterMixin):
    def __init__(self, parent, log,data):
        wx.Panel.__init__(self, parent, -1, style=wx.BORDER_THEME)
        self.log = log
        self.data=data
        tID = wx.NewIdRef()
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.il = wx.ImageList(16, 16)
        self.idx1 = self.il.Add(images.Smiles.GetBitmap())
        self.sm_up = self.il.Add(images.SmallUpArrow.GetBitmap())
        self.sm_dn = self.il.Add(images.SmallDnArrow.GetBitmap())
        self.list = ListCtrl(self, tID,
                                 style=wx.LC_REPORT
                                 #| wx.BORDER_SUNKEN
                                 | wx.BORDER_NONE
                                 | wx.LC_EDIT_LABELS
                                 #| wx.LC_SORT_ASCENDING    # disabling initial auto sort gives a
                                 # | wx.LC_NO_HEADER         # better illustration of col-click sorting
                                 | wx.LC_VRULES
                                 | wx.LC_HRULES
                                 # | wx.LC_SINGLE_SEL
                                 )
        self.list.SetImageList(self.il, wx.IMAGE_LIST_SMALL)
        sizer.Add(self.list, 1, wx.EXPAND)
        self.PopulateList(self.data)
        #self.SortListItems(0, True)
        self.SetSizer(sizer)
        self.SetAutoLayout(True)
        # self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnItemSelected, self.list)
    def ReCreate(self, data):
        self.data=data
        self.list.ClearAll()
        self.PopulateList(data)
        self.list.Refresh()
    def OnUseNative(self, event):
        wx.SystemOptions.SetOption("mac.listctrl.always_use_generic", not event.IsChecked())
        wx.GetApp().GetTopWindow().LoadDemo("ListCtrl")
    def PopulateList(self,data):
        info = wx.ListItem()
        info.Align = wx.LIST_FORMAT_LEFT
        info.Text = "文件名"
        self.list.InsertColumn(0, info)
        # info.Align = wx.LIST_FORMAT_LEFT
        # info.Text = "属性"
        # self.list.InsertColumn(1, info)
        # info.Align = wx.LIST_FORMAT_LEFT
        # info.Text = "大小"
        # self.list.InsertColumn(2, info)
        for i in data:
            index = self.list.InsertItem(self.list.GetItemCount(), i,self.idx1)
            item=self.list.GetItem(index,1)
            # index=self.data.index(i)
            self.list.SetItem(index, 0, i)
            # self.list.SetItem(index, 1, i[1])
            # self.list.SetItem(index, 2, i[2])
            # self.list.SetItemData(index,i[0])
        self.list.SetColumnWidth(0,150)
        # self.list.SetColumnWidth(1, 70)
        # self.list.SetColumnWidth(2, 50)
        self.currentItem = 0
    def getColumnText(self, index, col):
        item = self.list.GetItem(index, col)
        return item.GetText()
    # def OnItemSelected(self, event):
    #     event.Skip()
    #     # event.Skip()