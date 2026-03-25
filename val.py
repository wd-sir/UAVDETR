#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : wd
# @time    : 2025/3/16 14:01
# @function: valid

import warnings
warnings.filterwarnings('ignore')
import os
from ultralytics import RTDETR

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = 'runs/train/exp/weights/best.pt'
    model = RTDETR(model_path) 
    result = model.val(data='datasets/uav_dataset/data.yaml',
                      split='test',
                      imgsz=640,
                      batch=16,
                      project='runs/val',
                      name='exp',
                      )