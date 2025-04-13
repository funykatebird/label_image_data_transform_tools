#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-04-13 12:50
# @update    : 2025-04-13 12:50
# @Author  : funy
# @File    : prepare_chinese_caption.py
# @Des    : prepare_chinese_caption.py
# @Email  :  @16.com
# @Software: PyCharm

# https://blog.csdn.net/qq_45378085/article/details/143364406
# https://github.com/gasharper/linux-fonts
# https://github.com/jiaxiaochu/font/tree/master

import cv2
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import requests
#
# ①opencv-python(只包含主模块)
# ②opencv-contrib-python(包含main和contrib模块)
# ③opencv-python-headless(与opencv-python相同，但没有GUI功能)
# ④opencv-contrib-python-headless(与opencv-contrib-python相同，但没有GUI功能)

def cv2AddChineseText(img, text, position, textColor=(0, 0, 255), textSize=15):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


input_image = "D:\\20210706E\\2024-python\\pythonProjectCodeTest\\images\\1.jpg"

img = cv2.imread(input_image)
# cv2.imread(image)
# import cv2

# img = cv2.imread('test.png')  # 读取彩色图像(BGR)
# cv2.putText(img, '@Elaine', (300, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)

# img2 = cv2AddChineseText(img, '中文字幕', (40, 40), textColor=(0, 0, 255), textSize=35)
# cv2.imshow('test', img2)  # 显示叠加图像

img_pil = Image.open(input_image).convert("RGB")
image = cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_BGR2RGB)
# 缩放比例
h, w =image.shape[:2]
scale_rate = max(h,w)//360
# padding_img = np.zeros((h+20*scale_rate,w,3),dtype=np.uint8) # 黑底
padding_img = np.ones((h+20*scale_rate,w,3),dtype=np.uint8) # 白底
padding_img[20*scale_rate:,:,:] = image
padding_img = Image.fromarray(padding_img)
font = ImageFont.truetype("simsun.ttc", 20*scale_rate)
draw = ImageDraw.Draw(padding_img)
image_caption = '中文标题'
# 立体显示
draw.text((0, 2),image_caption,font=font,fill=(255,255,255)) # 白字
draw.text((2, 2),image_caption,font=font,fill=(255,0,255)) # 黑字 BGR
img3 = np.array(padding_img)

cv2.imshow('test', img3)  # 显示叠加图像
cv2.waitKey()  # 等待按键命令



# camera=cv.VideoCapture(0)
# face_detect=cv.CascadeClassifier('D:/opencv/opencv-4.7.0-windows/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml')
# while True:
#     flag,frame=camera.read()
#     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     faces=face_detect.detectMultiScale(gray)
#     for x,y,w,h in faces:
#         cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
#         frame1=cv2AddChineseText(frame,"彭锁群", (x, y),(255, 0, 0), 30)
#         cv.imshow("我的照片", frame1)
#     key=cv.waitKey(1)
#     if key==ord("q"):
#         break
# camera.release()
# cv.destoryAllWindows()