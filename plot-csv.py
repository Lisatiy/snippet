#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-04-23 下午22:08
# @Author  : Shufei Li
# @Site    : http://github.com/Lisatiy
# @File    : plot-csv.py
# @IDE: PyCharm Community Edition
"""
Present the figure from csv file, which is the loss in tensorboard
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    # 第一个是字符型的描述内容
    return list(map(int, x[1:])), list(map(float, y[1:]))


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()

x4, y4 = readcsv("./data/fpn-im2col-lstm-g05a050-tag-loss.csv")
plt.plot(x4, y4, color='orange', label='g05a050')

x2, y2 = readcsv("./data/fpn-im2col-lstm-g10a050-tag-loss.csv")
plt.plot(x2, y2, color='red', label='g10a050')

x, y = readcsv("./data/fpn-im2col-lstm-g10a025-tag-loss.csv")
plt.plot(x, y, 'g', label='g10a025')

x1, y1 = readcsv("./data/fpn-im2col-lstm-g20a025-tag-loss.csv")
plt.plot(x1, y1, color='black', label='g20a025')

x4, y4 = readcsv("./data/fpn-im2col-lstm-g50a025-tag-loss.csv")
plt.plot(x4, y4, color='blue', label='g50a025')

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.ylim(0, 5)
plt.xlim(0, 100)
plt.xlabel('Steps', fontsize=8)
plt.ylabel('Score', fontsize=8)
plt.legend(fontsize=12)
plt.show()

magit commit.
