#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-03-30 20:40
# @update    : 2025-03-30 20:40
# @Author  : funy
# @File    : prepare_find_files.py
# @Des    : prepare_find_files.py
# @Email  :  @16.com
# @Software: PyCharm

import os
import sys
import shutil
from operator import index

import pandas as pd
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result  # 返回被装饰函数的执行结果
    return wrapper

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile))

# srcfile='/Users/xxx/git/project1/test.sh'
# dstfile='/Users/xxx/tmp/tmp/1/test.sh'
#
# mymovefile(srcfile,dstfile)


@timer
def find_files(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            find_files(cur_path, all_files)
        else:
            # if ".png" in cur_path:  ## 条件
            if ".py" in cur_path:  ## 条件
                all_files.append(cur_path)
    return all_files


# path = "../data/"
path = "./"
list_name = []
result_list = find_files(path, list_name)
print(result_list)
source_path =r"D:\20210706E\2024-python\pythonProjectCodeTest"
desc_path = r"D:\20210706E\2024-python\pythonProjectCodeTest\Video_results"

file_list =os.listdir(source_path)
n =1000
video_result_list = []
for i, file in enumerate(file_list):
    print(i, file)
    if file.endswith('.py'):
        des_name = '20250330'+ str(n) +'.mp4'
        n+=1
        # des_name = file.split('.')[0]
        os.makedirs(desc_path,exist_ok=True)
        mycopyfile(os.path.join(source_path,file), os.path.join(desc_path,des_name))
        video_result_list.append([os.path.join(source_path,file),des_name])
print(video_result_list)

df=pd.DataFrame(video_result_list,columns =['source_name','dest_name'])
df.to_csv('funy_video_name.csv',encoding='utf-8-sig',index=True)


