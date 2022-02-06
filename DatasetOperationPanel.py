import wx
import images
from DatasetTree import DatasetTree
from ListDataPanel import ListDataPanel

class DatasetOperationPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log=log
        self.editDatasetBTN = wx.Button(self, -1, label="编辑数据集", size=(200, 35))
        self.datasetTree = DatasetTree(self, self.log, size=(200, 300))
        self.listDataPanel = ListDataPanel(self, self.log, [])
        self.middlePanel = wx.Panel(self, -1, size=(490,-1))
        self.middlePanel.SetBackgroundColour(wx.Colour(213,112,112))
        hbox = wx.BoxSizer()
        vvbox = wx.BoxSizer(wx.VERTICAL)
        vvbox.Add(self.editDatasetBTN, 0)
        vvbox.Add(self.datasetTree, 1)
        hbox.Add(vvbox, 0,wx.EXPAND)
        hbox.Add(self.listDataPanel, 0, wx.EXPAND)
        hbox.Add(self.middlePanel, 0, wx.EXPAND)
        self.SetSizer(hbox)
