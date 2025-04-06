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
from idlelib.iomenu import encoding
from operator import index

import cv2
import torch
import torch.nn as nn
from bokeh.transform import transform
from fontTools.subset import subset
from openpyxl.styles.builtins import total
from sympy.printing.codeprinter import requires
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image
import pandas as pd

from prepare_deduplicate_images import file_list

source_path ="D:\\20210706E\\2024-python\\pythonProjectCodeTest\\images"
features_dir = './features'

video_path = "D:\\20210706E\\2024-python\\pythonProjectCodeTest\\Video_results\\"
pic_path = './deduplicate_images\\'
# img_path = "D:\\20210706E\\2024-python\\pythonProjectCodeTest\\images\\1.jpg"
# file_name = img_path.split('/')[-1]
# file_name = img_path.split('\\')[-1]
# feature_path = os.path.join(features_dir, file_name + '.txt')
# print(feature_path)


video_file_list = os.listdir(video_path)
video_file_list =[file for file in video_file_list if file.endswith('.mp4')]
print(video_file_list)

transform1 = transforms.Compose([
    # transforms.Scale(256),
transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)

# col_number = 64  # 控制去重粒度
col_number = 1024  # 控制去重粒度
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



all_feature_list = []

def extract_video_name (input_image):
    return input_image[:-10]

def video2pic2(video_name):
    video_name = video_name.split('.')[0]
    cnt =0
    dnt =0
    # if os.path.exists(pic_path +str(video_name)):
    #     pass
    # else:
    #     os.mkdir(pic_path+str(video_name))
    os.makedirs(pic_path+str(video_name),exist_ok=True)

    # cap = cv2.VideoCapture(video_path+str(video_name)+'.mp4')
    cap = cv2.VideoCapture(os.path.join(video_path, str(video_name) + '.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("video duration is: {0:.2f} ".format((total_frames/fps))+ 'seconds')
    image_feature_list = [] # 记录视频所有帧的特征

    while True:
        # get a frame
        ret, image =cap.read()
        if image is None:
            break
        # show a frame
        w =image.shape[1]
        h =image.shape[0]
        tmp ='00000' + str(dnt).zfill(5)
        ##tmp = str(dnt).zfill(5)
        ##cv2.imencode('.jpg',image)[1].tofile(pic_path+str(video_name)+'/'+str(dnt)+'.jpg')
        cv2.imencode('.jpg', image)[1].tofile(pic_path + str(video_name) + '/' + str(video_name)+'_'+tmp[-5:] + '.jpg')

        tmp2 = video_name +"_" + str(dnt).zfill(5)
        dnt = dnt + 1
        if (cnt %50) ==0:
            print(pic_path+str(video_name)+'/'+tmp2+'.jpg')
        cnt = cnt +1

        feature_list = [] # 记录当前帧特征
        pic =pic_path +str(video_name) +'/' +tmp2 + '.jpg'
        image_name = tmp2 +'.jpg'

        print("extract embedding -->{0}/{1} {2} {3}".format(dnt,total_frames,image_name,video_name))
        # 读入视频流
        img11 =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img11)

        # 读入图片序列
        # img=Image.open(pic).convert('RGB') # 可以处理png格式的图片
        img1 = transform1(img)
        x=Variable(torch.unsqueeze(img1,dim=0).float(),requires_grad = False)
        y = resnet50_feature_extractor(x)
        y = y.data.numpy()
        # np.savetxt(feature_path, y, delimiter=',')
        y = y.tolist()
        feature_list = y[0]
        # 插入图片名
        feature_list.insert(0, image_name)
        # 插入视频名
        feature_list.insert(0, video_name)
        image_feature_list.append(feature_list)
        ######################
        # cv2.imshow('frame', image)
        # if cv2.waitKey(1)& 0xff == ord('q'):
        #     break
        #############
        # print(y)
        # print(type(y))
        all_feature_list.append(y[0])
    cap.release()

    columns = ["column_"+str(i).zfill(4) for i in range(1024)]
    color_attribute = ['video'] + ['image'] + columns
    test = pd.DataFrame(data=image_feature_list,columns=color_attribute)
    save_feature_name = f'D:\\20210706E\\2024-python\\pythonProjectCodeTest\\Video_results\\{video_name}.csv'
    # test.to_csv(save_feature_name,encoding='utf-8-sig',index=True)
    df =test.drop_duplicates(subset=columns,keep='last',ignore_index=True)
    # print(len(test))
    # print(len(df))
    save_feature_name2 = f"./{video_name}_de.csv"
    # df.to_csv(save_feature_name2,encoding='utf-8-sig',index=True)
    df = df.copy()
    df.loc[:,'total_frames']= total_frames
    # print(df.head())
    return df


if __name__ =='__main__':
    all_data_features = None
    for video_name  in video_file_list:
        print(video_name)
        video_features_df = video2pic2(video_name)
        if all_data_features is None:
            all_data_features = video_features_df
        else:
            all_data_features = pd.concat([all_data_features,video_features_df])
    save_feature_names = 'all_de.csv'
    columns = ['column_' +str(i).zfill(4) for i in range(1024)]
    final_df = all_data_features.drop_duplicates(subset=columns,keep='first',ignore_index=True)
    save_feature_name4 ='all_video_names.csv'
    # final_df.to_csv(save_feature_name4,encoding='utf-8-sig',index = True)
    print(len(all_data_features))
    print(len(final_df))
    df_selected =final_df[['video','image','total_frames']]
    df_video_count=df_selected.groupby('video').count()
    df_video_count=df_video_count.rename(columns = {'image':'frame_count'})
    df_selected2 = final_df[['video','total_frames']]
    df_selected2 = df_selected2.drop_duplicates(subset=['video','total_frames'],keep='first',ignore_index=True)

    for i in range(len(df_video_count)):
        temp = 0
        temp =df_selected2[df_selected2['video']==df_video_count.index[i]]['total_frames']
        df_video_count.iloc[i,1]=temp
    print(df_video_count)

    df_video_count.to_csv('video_names.csv',encoding='utf-8-sig',index= True)




#
# file_list = os.listdir(source_path)
# for i,image in enumerate(file_list):
#     print("extract {}/{} {} image embedding".format(i+1,len(file_list),image))
#     file_name = image.split('\\')[-1]
#     feature_path = os.path.join(features_dir, file_name + '.txt')
#     # print(feature_path)
#     img = Image.open(os.path.join(source_path,image)).convert('RGB')
#     # img = Image.open(img_path)
#     img1 = transform1(img)
#     x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#     # y1 = resnet18(x)
#     y = resnet50_feature_extractor(x)
#     y = y.data.numpy()
#     np.savetxt(feature_path, y, delimiter=',')
#     # y3 = resnet152(x)
#     # y4 = densenet201(x)
#
#     # y_ = np.loadtxt(feature_path, delimiter=',').reshape(1, 2048)
#     y = y.tolist()
#     y[0].insert(0, image)
#     # print(y)
#     # print(type(y))
#     all_feature_list.append(y[0])
#
# col_name = ["feature_"+ str(i).zfill(4) for i in range(col_number)]
# column_final  = ['image_name'] + col_name
# # 提取图片特征
# # df = pd.DataFrame(data =y, columns=column_final)
# df = pd.DataFrame(data =all_feature_list, columns=column_final)
# df.to_csv('test.csv',encoding='utf-8-sig')
# # 图片去重
# df2=df.drop_duplicates(subset=col_name,keep='first',inplace=False)
# df2.to_csv('deduplicate.csv',encoding='utf-8-sig',index=True)