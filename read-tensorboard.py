#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-04-23 下午22:09
# @Author  : Shufei Li
# @Site    : http://github.com/Lisatiy
# @File    : plot-csv.py
# @IDE: PyCharm Community Edition
"""
Read data from tensorboard
"""

from tensorboard.backend.event_processing import event_accumulator

# 加载日志数据
ea = event_accumulator.EventAccumulator('./data/events.out.tfevents.1587131085.pc-C246-WU4')
ea.Reload()
print(ea.scalars.Keys())

val_psnr = ea.scalars.Items('loss')
print(len(val_psnr))
print([(i.step, i.value) for i in val_psnr])
