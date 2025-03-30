# coding=utf-8
import cv2
import os

allims = '/data/dataset/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages'
out = '/data/dataset/datasets/label_image_tools/voc2coco-pattern'
trainval = '/data/dataset/datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Main/test.txt'
f = open(trainval)
for line in f:
    im_path = os.path.join(allims, line[:-1] + '.jpg')
    im = cv2.imread(im_path)
    out_path = os.path.join(out, line[:-1] + '.jpg')
    cv2.imwrite(out_path, im)
