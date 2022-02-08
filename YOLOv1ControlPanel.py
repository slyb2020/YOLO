import wx


class YOLOv1ControlPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size, style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_THEME)
        self.log = log
        self.SetBackgroundColour(wx.Colour(240, 240, 240))
        vbox = wx.BoxSizer(wx.VERTICAL)
        hhbox = wx.BoxSizer()
        vbox.Add((-1, 5))
        hhbox.Add((10, -1))
        hhbox.Add(wx.StaticText(self, label="选定预训练模型："), wx.TOP, 5)
        self.pretrainedModelCOMBO = wx.ComboBox(self, value='1', size=(200, -1), choices=['1', '2', '3'])
        hhbox.Add(self.pretrainedModelCOMBO, 1, wx.RIGHT, 10)
        vbox.Add(hhbox)
        self.SetSizer(vbox)
