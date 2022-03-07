import os

import wx

import images
from DatasetLabelProcess import *
from YOLOv1Algorithm import *
import wx.lib.scrolledpanel as scrolled
# from MNIST_Dataset import testDataset
from torchvision.datasets import MNIST
from errorDataFinder import ErrorTest
from ID_DEFINE import *


testDataset = MNIST(root="D:\\WorkSpace\\DataSet", train=False)

class ErrorPicPanel(scrolled.ScrolledPanel):
    def __init__(self, parent, data=[]):
        scrolled.ScrolledPanel.__init__(self, parent, -1)
        # x, y = self.GetClientSize()
        self.data = data
        self.ReCreate()
    def ReCreate(self):
        self.DestroyChildren()
        x, y = 45,45
        hbox = wx.BoxSizer()
        if len(self.data) > 0:
            for index, label in self.data:
                btn = wx.Button(self, size=(y,y))
                btn.SetToolTip(str(label))
                img, label = testDataset[index]
                img = np.array(img)
                self.img = np.zeros((28,28,3),dtype=np.int8)
                self.img[:,:,0]=img
                self.img[:,:,1]=img
                self.img[:,:,2]=img
                self.width, self.height = self.img.shape[1], self.img.shape[0]
                bmp = wx.Image(self.width, self.height, self.img).Scale(width=y, height=y,
                                                quality=wx.IMAGE_QUALITY_BOX_AVERAGE).ConvertToBitmap()
                btn.SetBitmap(bmp)
                hbox.Add(btn,0)
        self.SetSizer(hbox)
        self.SetAutoLayout(1)
        self.SetupScrolling()


class LeNetMNISTPanel(wx.Panel):
    def __init__(self, parent,  log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.leftPanel = wx.Panel(self, size=(340,-1), style=wx.BORDER_THEME)
        self.middlePanel = wx.Panel(self, size=(300,-1))
        self.rightPanel = wx.Panel(self, size=(800,-1), style=wx.BORDER_THEME)
        hbox = wx.BoxSizer()
        hbox.Add(self.leftPanel, 0, wx.EXPAND)
        hbox.Add(self.middlePanel, 0, wx.EXPAND)
        hbox.Add(self.rightPanel, 1, wx.EXPAND)
        self.SetSizer(hbox)
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add((-1,5))
        hhbox = wx.BoxSizer()
        hhbox.Add((10,-1))
        hhbox.Add(wx.StaticText(self.leftPanel, label="模型结构:", size=(70,-1)),0,wx.TOP,5)
        self.modelStructureCombo = wx.ComboBox(self.leftPanel, value="LeeNet", choices=LeNetModelList, size=(240,-1))
        hhbox.Add(self.modelStructureCombo, 0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        vbox.Add((-1,10))
        hhbox = wx.BoxSizer()
        hhbox.Add((10,-1))
        modelName = self.modelStructureCombo.GetValue()
        preList = os.listdir("model/"+modelName+'/')
        hhbox.Add(wx.StaticText(self.leftPanel, label="预训练模型:", size=(70,-1)),0,wx.TOP,5)
        self.preModelCombo = wx.ComboBox(self.leftPanel, value=preList[-1], choices=preList, size=(240,-1))
        hhbox.Add(self.preModelCombo, 0)
        vbox.Add(hhbox, 0, wx.EXPAND)
        vbox.Add(wx.Panel(self.leftPanel),1)
        self.runBTN = wx.Button(self.leftPanel, label="运行", size=(200, 35))
        self.runBTN.Bind(wx.EVT_BUTTON, self.RunErrorTest)
        vbox.Add(self.runBTN, 0, wx.EXPAND|wx.ALL, 2)
        self.leftPanel.SetSizer(vbox)

        hhbox=wx.BoxSizer()
        self.modelTXT = wx.TextCtrl(self.middlePanel, size=(100,100), style=wx.TE_MULTILINE|wx.TE_READONLY)
        hhbox.Add(self.modelTXT, 1, wx.ALL|wx.EXPAND)
        self.middlePanel.SetSizer(hhbox)

        vvbox=wx.BoxSizer(wx.VERTICAL)
        self.buttonIDList=[]
        self.buttonList=[]
        self.panelList=[]
        for i in range(10):
            hhbox = wx.BoxSizer()
            id = wx.NewId()
            self.buttonIDList.append(id)
            button = wx.Button(self.rightPanel, id, label="%d"%i, size=(50, 10))
            self.buttonList.append(button)
            hhbox.Add(button, 0,wx.EXPAND)
            panel = ErrorPicPanel(self.rightPanel)
            self.panelList.append(panel)
            hhbox.Add(panel,1, wx.EXPAND)
            vvbox.Add(hhbox,1,wx.EXPAND)
        self.rightPanel.SetSizer(vvbox)
        self.Bind(wx.EVT_BUTTON, self.OnButton)

    def OnButton(self, event):
        objectId = event.GetId()
        if objectId in self.buttonIDList:
            index = self.buttonIDList.index(objectId)

    def RunErrorTest(self, event):
        modelStructure = self.modelStructureCombo.GetValue()
        modelStructure = LeNetModelList.index(modelStructure)
        preModel = self.preModelCombo.GetValue()
        errorList = ErrorTest(modelStructure, preModel)
        for i in range(10):
            self.panelList[i].data=[]
        for index, label, predict in errorList:
            self.panelList[predict].data.append([index,label])
        for i in range(10):
            self.panelList[i].ReCreate()

class LeNetPanel(wx.Panel):
    def __init__(self, parent,  log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
                                    # wx.BK_DEFAULT
                                    # wx.BK_TOP
                                    wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    # | wx.NB_MULTILINE
                                    )
        il = wx.ImageList(16, 16)
        il.Add(images._rt_smiley.GetBitmap())
        self.total_page_num = 0
        self.notebook.AssignImageList(il)
        idx2 = il.Add(images.GridBG.GetBitmap())
        idx3 = il.Add(images.Smiles.GetBitmap())
        idx4 = il.Add(images._rt_undo.GetBitmap())
        idx5 = il.Add(images._rt_save.GetBitmap())
        idx6 = il.Add(images._rt_redo.GetBitmap())
        hbox = wx.BoxSizer()
        self.leNetIntroductionPanel = wx.Panel(self.notebook,style=wx.BORDER_THEME)
        self.notebook.AddPage(self.leNetIntroductionPanel, "LeNet神经网络模型介绍")
        self.leNetNMISTlPanel = LeNetMNISTPanel(self.notebook, self.log)
        self.notebook.AddPage(self.leNetNMISTlPanel, "LeNet在MNIST上的应用")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(hbox)
    #     self.Bind(wx.EVT_BUTTON, self.OnPictureButton)
    #
    # def OnPictureButton(self, event):
    #     id = event.GetId()
    #     # if id in self.trainDatasetPanel.buttonIdList:
    #     #     self.index = self.trainDatasetPanel.buttonIdList.index(id)
    #     #     self.leftPanel.Refresh()

