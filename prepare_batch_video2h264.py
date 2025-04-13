#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-04-13 8:58
# @update    : 2025-04-13 8:58
# @Author  : funy
# @File    : prepare_batch_video2h264.py
# @Des    : prepare_batch_video2h264.py
# @Email  :  @16.com
# @Software: PyCharm
# https://www.jianshu.com/p/4e85728c7c4c

import os
# import fnmatch
import re
import subprocess

os.environ['PATH']= os.pathsep + r'D:\ProgramFiles\ffmpeg-7.1.1-full_build\bin'
# command = f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac {target_path}"

# gl_file_list = []
# gl_failed_list = []


def getFilesPath(path):
    gl_file_list = [] # 存储视频文件名称
    # 获得指定目录中的内容
    file_list = os.listdir(path)
    for file_name in file_list:
        new_path = os.path.join(path, file_name)
        if os.path.isdir(new_path):
            getFilesPath(new_path)
        elif os.path.isfile(new_path):
            # 文件处理
            # if fnmatch.fnmatch(new_path, '*.(mp4|avi)'): # 匹配mp4格式
            #     # 视频处理
            #     fileProcessing(new_path)
            result = re.match(r".+\.(mp4|avi|mpeg|mov|flv|mpg|f4v|rmvb|mkv|ogg|asf|3gp|m4a)$", new_path)
            if result:
                gl_file_list.append(new_path)
                # fileProcessing(new_path)
        else:
            print("It's not a directory or a file.")
    return gl_file_list


def fileProcessing(file_list, output_path):
    gl_failed_list = [] # 存放转换失败的视频文件名称
    print("start----------------")
    codePre = "ffmpeg -threads 2 -i "
    # codeMid = " -vcodec h264 "
    # codeMid = " -vcodec libx264 -acodec aac -preset fast -b:v 2000k "
    codeMid = " -vcodec libx264 -acodec aac "
    # ffmpeg -i input.mp4 -vcodec h264 output.mp4
    codeMid = " -vcodec h264 "
    for file_path in file_list:
        subname = file_path.split('.')
        print(subname)
        # output_path = subname[0] + "_new.mp4"   # 处理后的文件路径
        output_path = os.path.join(output_path, subname[0]+'_H264.mp4')
        command = codePre + file_path + codeMid + output_path
        file_name = os.path.basename(file_path).split('.')

        # result = os.system(command)
        # if(result != 0):
        #     gl_failed_list.append(file_path)
        #     print(file_name[0], "is failed-----", "result = ", result)
        # else:
        #     print("end------", file_name[0], "result = ", result)

        try:
            retcode = subprocess.call(command, shell=True)  # 调用了cmd
            if retcode == 0:
                print(file_name[0], "successed------")
            else:
                print(file_name[0], "is failed--------")
        except Exception as e:
            print("Error:", e)

    print("---------------End all-----------------")
    print("failed:", gl_failed_list)


if __name__ =='__main__':
    # file_path = r'/Users/xxx/Desktop/video'
    file_path = 'D:\\20210706E\\2024-python\\pythonProjectCodeTest\\Video_results2'
    output_path = file_path  # 结果目录
    gl_file_list = getFilesPath(file_path)
    fileProcessing(gl_file_list,output_path)