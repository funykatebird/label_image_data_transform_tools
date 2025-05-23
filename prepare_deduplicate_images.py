#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-04-06 10:05
# @update    : 2025-04-06 10:05
# @Author  : funy
# @File    : prepare_extract_image_features.py
# @Des    : prepare_extract_image_features.py
# @Email  :  @16.com
# @Software: PyCharm

# https://www.jb51.net/article/167966.htm
# https://blog.csdn.net/qq_41234663/article/details/129896837

# -*- coding: utf-8 -*-

import os.path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image
import pandas as pd

source_path ="D:\\20210706E\\2024-python\\pythonProjectCodeTest\\images"
features_dir = './features'
# img_path = "D:\\20210706E\\2024-python\\pythonProjectCodeTest\\images\\1.jpg"
# file_name = img_path.split('/')[-1]
# file_name = img_path.split('\\')[-1]
# feature_path = os.path.join(features_dir, file_name + '.txt')
# print(feature_path)

transform1 = transforms.Compose([
    # transforms.Scale(256),
transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)

col_number = 2048
# resnet18 = models.resnet18(pretrained = True)/
# resnet50_feature_extractor = models.resnet50(pretrained=True)
# resnet50_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet50_feature_extractor.fc = nn.Linear(2048, col_number)
# resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
# torch.nn.init.eye(resnet50_feature_extractor.fc.weight)

for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False
# resnet152 = models.resnet152(pretrained = True)
# densenet201 = models.densenet201(pretrained = True)

file_list = os.listdir(source_path)

all_feature_list = []
for i,image in enumerate(file_list):
    print("extract {}/{} {} image embedding".format(i+1,len(file_list),image))
    file_name = image.split('\\')[-1]
    feature_path = os.path.join(features_dir, file_name + '.txt')
    # print(feature_path)
    img = Image.open(os.path.join(source_path,image)).convert('RGB')
    # img = Image.open(img_path)
    img1 = transform1(img)
    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    # y1 = resnet18(x)
    y = resnet50_feature_extractor(x)
    y = y.data.numpy()
    np.savetxt(feature_path, y, delimiter=',')
    # y3 = resnet152(x)
    # y4 = densenet201(x)

    # y_ = np.loadtxt(feature_path, delimiter=',').reshape(1, 2048)
    y = y.tolist()
    y[0].insert(0, image)
    # print(y)
    # print(type(y))
    all_feature_list.append(y[0])

col_name = ["feature_"+ str(i).zfill(4) for i in range(col_number)]
column_final  = ['image_name'] + col_name

# 提取图片特征
# df = pd.DataFrame(data =y, columns=column_final)
df = pd.DataFrame(data =all_feature_list, columns=column_final)
df.to_csv('test.csv',encoding='utf-8-sig')

# 图片去重
df2=df.drop_duplicates(subset=col_name,keep='first',inplace=False)
df2.to_csv('deduplicate.csv',encoding='utf-8-sig',index=True)