#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : wd
# @time    : 2025/3/16 15:10
# @function: detect

import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('uav.pt')  # select your model.pt path
    model.predict(
        source=r'D:\project\PycharmProject\ultralytics-main\datasets\uav_dataset\test\images',  
        data=r'D:\project\PycharmProject\ultralytics-main\datasets\uav_dataset\data.yaml',
        split='test',  
        conf=0.25,
        project='runs/predict',
        name='uav_dataset/uavdetr',
        save=True,
    )
