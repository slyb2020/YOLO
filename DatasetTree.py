#!/usr/bin/env python
# encoding: utf-8
'''
@author: slyb
@license: (C) Copyright 2017-2020, 天津定智科技有限公司.
@contact: slyb@tju.edu.cn
@file: DatasetTree.py
@time: 2019/9/1 15:54
@desc:
'''
import wx
from math import *
from ID_DEFINE import *
import string
import images


class MyTreeCtrl(wx.TreeCtrl):
    def __init__(self, parent, id, pos, size, style, log):
        wx.TreeCtrl.__init__(self, parent, id, pos, size, style)
        self.log = log

    def OnCompareItems(self, item1, item2):
        t1 = self.GetItemText(item1)
        t2 = self.GetItemText(item2)
        self.log.WriteText('compare: ' + t1 + ' <> ' + t2 + '\n')
        if t1 < t2: return -1
        if t1 == t2: return 0
        return 1


class DatasetTree(wx.Panel):
    def __init__(self, parent, log, size):
        # Use the WANTS_CHARS style so the panel doesn't eat the Return key.
        wx.Panel.__init__(self, parent, -1, size, style=wx.WANTS_CHARS)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.log = log
        tID = wx.NewIdRef()
        datasetList = [
            ["ImageNet", [["2017", []], ["2018", []]]],
            ["FasionMNIST", [["2007", []], ["2012", []]]],
            ["MNIST", [["2007", []], ["2012", []]]],
            ["猫狗大战", []],
        ]
        self.tree = MyTreeCtrl(self, tID, wx.DefaultPosition, size,
                               wx.TR_HAS_BUTTONS
                               | wx.TR_EDIT_LABELS
                               # | wx.TR_MULTIPLE
                               # | wx.TR_HIDE_ROOT
                               , self.log)
        isz = (16, 16)
        il = wx.ImageList(isz[0], isz[1])
        fldridx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, isz))
        fldropenidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_OTHER, isz))
        fileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        smileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        # smileidx    = il.Add(images.Smiles.GetBitmap())
        self.tree.SetImageList(il)
        self.il = il
        self.root = self.tree.AddRoot("数据集")
        self.tree.SetItemData(self.root, "根")
        self.tree.SetItemImage(self.root, fldridx, wx.TreeItemIcon_Normal)
        self.tree.SetItemImage(self.root, fldropenidx, wx.TreeItemIcon_Expanded)
        for i in datasetList:
            child = self.tree.AppendItem(self.root, i[0])
            self.tree.SetItemData(child, "集")
            self.tree.SetItemImage(child, fldridx, wx.TreeItemIcon_Normal)
            self.tree.SetItemImage(child, fldropenidx, wx.TreeItemIcon_Expanded)
            for j in i[1]:
                last = self.tree.AppendItem(child, j[0])
                self.tree.SetItemData(last, "子集")
                self.tree.SetItemImage(last, fldridx, wx.TreeItemIcon_Normal)
                self.tree.SetItemImage(last, fldropenidx, wx.TreeItemIcon_Expanded)
                for k in j[1]:
                    item = self.tree.AppendItem(last, k[0])
                    self.tree.SetItemData(item, "孙集")
                    self.tree.SetItemImage(item, fileidx, wx.TreeItemIcon_Normal)
                    self.tree.SetItemImage(item, smileidx, wx.TreeItemIcon_Selected)
        self.tree.ExpandAll()
        self.Bind(wx.EVT_TREE_BEGIN_LABEL_EDIT, self.OnBeginEdit, self.tree)
        self.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit, self.tree)

    def OnSize(self, event):
        w, h = self.GetClientSize()
        self.tree.SetSize(0, 0, w, h)

    def OnBeginEdit(self, event):
        event.Veto()

    def OnEndEdit(self, event):
        event.Veto()

    def ReCreateTree(self):
        self.tree.Destroy()
        tID = wx.NewIdRef()
        self.tree = MyTreeCtrl(self, tID, wx.DefaultPosition, (200, 900),
                               wx.TR_HAS_BUTTONS
                               | wx.TR_EDIT_LABELS
                               # | wx.TR_MULTIPLE
                               # | wx.TR_HIDE_ROOT
                               , self.log)
        isz = (16, 16)
        il = wx.ImageList(isz[0], isz[1])
        fldridx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, isz))
        fldropenidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_OTHER, isz))
        fileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        smileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FLOPPY, wx.ART_OTHER, isz))
        # smileidx    = il.Add(images.Smiles.GetBitmap())
        self.tree.SetImageList(il)
        self.il = il
        self.root = self.tree.AddRoot("数据集")
        self.tree.SetItemData(self.root, "根")
        self.tree.SetItemImage(self.root, fldridx, wx.TreeItemIcon_Normal)
        self.tree.SetItemImage(self.root, fldropenidx, wx.TreeItemIcon_Expanded)
        for i in self.master.department_list:
            child = self.tree.AppendItem(self.root, i[0])
            self.tree.SetItemData(child, "集")
            self.tree.SetItemImage(child, fldridx, wx.TreeItemIcon_Normal)
            self.tree.SetItemImage(child, fldropenidx, wx.TreeItemIcon_Expanded)
            for j in i[1]:
                last = self.tree.AppendItem(child, j[0])
                self.tree.SetItemData(last, "子集")
                self.tree.SetItemImage(last, fldridx, wx.TreeItemIcon_Normal)
                self.tree.SetItemImage(last, fldropenidx, wx.TreeItemIcon_Expanded)
                for k in j[1]:
                    item = self.tree.AppendItem(last, k[0])
                    self.tree.SetItemData(item, "孙集")
                    self.tree.SetItemImage(item, fileidx, wx.TreeItemIcon_Normal)
                    self.tree.SetItemImage(item, smileidx, wx.TreeItemIcon_Selected)
        self.tree.ExpandAll()
        # self.tree.Refresh()
