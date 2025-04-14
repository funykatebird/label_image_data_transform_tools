#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-04-14 20:27
# @update    : 2025-04-14 20:27
# @Author  : funy
# @File    : prepare_extract_key_frames.py
# @Des    : prepare_extract_key_frames.py
# @Email  :  @16.com
# @Software: PyCharm


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:48:57 2018
keyframes extract tool
this key frame extract algorithm is based on interframe difference.
The principle is very simple
First, we load the video and compute the interframe difference between each frames
Then, we can choose one of these three methods to extract keyframes, which are
all based on the difference method:

1. use the difference order
    The first few frames with the largest average interframe difference
    are considered to be key frames.
2. use the difference threshold
    The frames which the average interframe difference are large than the
    threshold are considered to be key frames.
3. use local maximum
    The frames which the average interframe difference are local maximum are
    considered to be key frames.
    It should be noted that smoothing the average difference value before
    calculating the local maximum can effectively remove noise to avoid
    repeated extraction of frames of similar scenes.
After a few experiment, the third method has a better key frame extraction effect.
The original code comes from the link below, I optimized the code to reduce
unnecessary memory consumption.
https://blog.csdn.net/qq_21997625/article/details/81285096
@author: zyb_as
"""
#https://blog.csdn.net/cungudafa/article/details/104919405
# https://zhuanlan.zhihu.com/p/696043674
# https://www.jb51.net/article/284251.htm

import os
import sys
import time


from scipy.signal import argrelextrema  # 极值点
import cv2
import operator  # 内置操作符函数接口（后面排序用到）
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端'Agg'
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor#线程池


def smooth(x, window_len=13, window='hanning'):
    """使用具有所需大小的窗口使数据平滑。

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    该方法是基于一个标度窗口与信号的卷积。
    通过在两端引入信号的反射副本(具有窗口大小)来准备信号，
    使得在输出信号的开始和结束部分中将瞬态部分最小化。
    input:
        x: the input signal输入信号
        window_len: the dimension of the smoothing window平滑窗口的尺寸
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
            平坦的窗口将产生移动平均平滑
    output:
        the smoothed signal平滑信号

    example:
    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: 如果使用数组而不是字符串，则window参数可能是窗口本身
    """
    print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    # 提高ValueError，“平滑仅接受一维数组。”
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    # 提高ValueError，“输入向量必须大于窗口大小。”
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[2 * x[0] - x[window_len:1:-1],
    x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))

    if window == 'flat':  # moving average平移
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


class Frame:
    """class to hold information about each frame
    用于保存有关每个帧的信息
    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
    x = (b - a) / max(a, b)
    print(x)
    return x


def getEffectiveFrame(videopath, dir):
    # 如果文件目录不存在则创建目录
    if not os.path.exists(dir):
        os.makedirs(dir)
    (filepath, tempfilename) = os.path.split(videopath)  # 分离路径和文件名
    (filename, extension) = os.path.splitext(tempfilename)  # 区分文件的名字和后缀
    # Setting fixed threshold criteria设置固定阈值标准
    USE_THRESH = False
    # fixed threshold value固定阈值
    THRESH = 0.6
    # Setting fixed threshold criteria设置固定阈值标准
    USE_TOP_ORDER = False
    # Setting local maxima criteria设置局部最大值标准
    USE_LOCAL_MAXIMA = True
    # Number of top sorted frames排名最高的帧数
    NUM_TOP_FRAMES = 50
    # smoothing window size平滑窗口大小
    len_window = int(50)

    print("target video :" + videopath)
    print("frame save directory: " + dir)
    # load video and compute diff between frames加载视频并计算帧之间的差异
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    # success, frame = cap.read()
    success, frame1 = cap.read()
    ######################################
    scale = 0.25
    if success:
        height, width = frame1.shape[:2]
        size = (int(width*scale),int(height*scale))
        frame = cv2.resize(frame1,size,interpolation=cv2.INTER_NEAREST)
    ######################################
    i = 0
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)  # 获取差分图
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])  # 平均帧
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
        # success, frame = cap.read()
        success, frame1 = cap.read()
        ######################################
        if success:
            height, width = frame1.shape[:2]
            size = (int(width * scale), int(height * scale))
            frame = cv2.resize(frame1, size, interpolation=cv2.INTER_NEAREST)
        ######################################
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    if USE_TOP_ORDER:
        # sort the list in descending order以降序对列表进行排序
        frames.sort(key=operator.attrgetter("diff"), reverse=True)  # 排序operator.attrgetter
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)
    if USE_THRESH:
        print("Using Threshold")  # 使用阈值
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                keyframe_id_set.add(frames[i].id)
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")  # 使用局部极大值
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)  # 平滑
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]  # 找极值
        print(len(frame_indexes))
        print(len(sm_diff_array))
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)  # 记录极值帧数

        plt.figure(figsize=(40, 20))
        plt.locator_params("x", nbins=100)
        # stem 绘制离散函数，polt是连续函数
        # plt.stem(sm_diff_array, linefmt='-', markerfmt='o', basefmt='--', label='sm_diff_array')
        plt.plot(sm_diff_array, linewidth=1, markersize=1, label='sm_diff_array')
        for i,p in enumerate(sm_diff_array):
            if i in frame_indexes:
                plt.scatter(i, p,marker = 'o', c='r',s =60)
        # plt.plot(x, y2, 'bo-', linewidth=4, markersize=5, label='平方数')
        plt.savefig(dir + filename + '_plot.png')

    # save all keyframes as image将所有关键帧另存为图像
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    keyframes = []
    success, frame = cap.read()
    idx = 0
    while (success):
        if idx in keyframe_id_set:
            # name = filename + '_' + str(idx) + ".jpg"
            name = filename + '_' + str(idx).zfill(4) + ".jpg"
            cv2.imwrite(dir + name, frame)
            keyframe_id_set.remove(idx)
        idx = idx + 1
        success, frame = cap.read()
    cap.release()


if __name__ == "__main__":
    print(sys.executable)
    start = time.time()
    print("[INFO]Effective Frame.")

    # Video path of the source file源文件的视频路径
    videopath = r'D:\20210706E\2024-python\pythonProjectCodeTest\Video_results2\699891e3-3fa4-4bcf-86c9-a51af58673c1.mp4'
    # Directory to store the processed frames存储已处理帧的目录
    dir = 'D:\\20210706E\\2024-python\\pythonProjectCodeTest\\images\\'
    getEffectiveFrame(videopath, dir)

    # video_path = 'D:\\20210706E\\2024-python\\pythonProjectCodeTest\\Video_results2\\'
    # video_list =os.listdir(video_path)
    # print(video_list)
    # with ThreadPoolExecutor(max_workers=2) as pool:
    #     for video in video_list:
    #         pool.submit(getEffectiveFrame,os.path.join(video_path,video),dir)

    print("[INFO]Extract Result time: ", time.time() - start)
