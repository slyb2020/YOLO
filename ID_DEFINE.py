#!/usr/bin/env python
# encoding: utf-8
"""
@author: slyb
@license: (C) Copyright 2017-2020, 天津定智科技有限公司.
@contact: slyb@tju.edu.cn
@file: ID_DEFINE.py.py
@time: 2019/6/16 15:23
@desc:
"""

import wx
import os

MENU_CHECK_IN = wx.NewIdRef()
MENU_CHECK_OUT = wx.NewIdRef()
MENU_STYLE_DEFAULT = wx.NewIdRef()
MENU_STYLE_XP = wx.NewIdRef()
MENU_STYLE_2007 = wx.NewIdRef()
MENU_STYLE_VISTA = wx.NewIdRef()
MENU_STYLE_MY = wx.NewIdRef()
MENU_USE_CUSTOM = wx.NewIdRef()
MENU_LCD_MONITOR = wx.NewIdRef()
MENU_HELP = wx.NewIdRef()
MENU_DISABLE_MENU_ITEM = wx.NewIdRef()
MENU_REMOVE_MENU = wx.NewIdRef()
MENU_TRANSPARENCY = wx.NewIdRef()
MENU_NEW_FILE = 10005
MENU_SAVE = 10006
MENU_OPEN_FILE = 10007
MENU_NEW_FOLDER = 10008
MENU_COPY = 10009
MENU_CUT = 10010
MENU_PASTE = 10011
ID_WINDOW_LEFT = wx.NewId()
ID_WINDOW_BOTTOM = wx.NewId()
ID_DATASET_BTN = wx.NewId()
ID_YOLOv1_BTN = wx.NewId()
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'airplane', 'bicycle', 'boat', 'bus', 'car', 'motobike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

CLASSES2 = ['人', '鸟', '猫', '牛', '狗', '马', '羊',
           '飞机', '自行车', '船', '公交车', '小汽车', '摩托车', '火车',
           '瓶子', '椅子', '餐桌', 'potted plant', '沙发', '显示器']

COLOR = [(255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
         (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
         (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
         (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (0, 0, 0), ]  # 用来标识20个类别的bbox颜色，可自行设定

dirName = os.path.dirname(os.path.abspath(__file__))
modelsDir = os.path.join(dirName, "Models\\")
ButtonIdArray = [
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
    [wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId(),wx.NewId()],
]