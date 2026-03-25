#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : wd
# @time    : 2025/3/15 14:35
# @function: train

import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('datasets/uav_dataset/UAV-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='datasets/uav_dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16, # batchsize 
                workers=4, #
                # device='0,1', 
                # resume='', # last.pt path
                patience=0, 
                project='runs/train',
                name='exp',
                )