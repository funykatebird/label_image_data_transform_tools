#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-04-06 13:50
# @update    : 2025-04-06 13:50
# @Author  : funy
# @File    : prepare_wget_remote_files.py
# @Des    : prepare_wget_remote_files.py
# @Email  :  @16.com
# @Software: PyCharm

import os
import sys
import time
import csv
from idlelib.iomenu import encoding

import pandas as pd
import wget

import tqdm
import subprocess

from param import output

from prepare_clip4videos import end_time


def timer(func):
    def a(*args,**kwargs):
        start_time = time.time()
        func(*args,**kwargs)
        end_time = time.time()
        print("program run time is :{} seconds".format(str(round(end_time-start_time))))
        return  end_time -start_time

    return a


@timer
def download_remote_file(input_csv,output_path='./download_file',save_log = 'download_file_info.csv'):
    if sys.platform =='win32':
        print('this is windows platform')
    elif sys.platform =='darwin':
        print('this is darwin platform')
    elif sys.platform =='macos':
        print('this is macos platform')
    elif sys.platform =='linux':
        print('this is linux platform')
    else:
        print('this is unkonwn platform')
    os.makedirs(output_path,exist_ok=True)

    name_list =[]
    n =0
    data = pd.read_csv(input_csv, encoding='utf-8')
    for i in range(len(data)):
        file_url =data.iloc[i]['video_url']
        file_name = data.iloc[i]['video_name']
        print("{}/{}: {} -->{}".format(i,len(data),file_name,file_url))
        full_path = os.path.join(output_path,file_name)
        if os.path.exists(full_path):
            continue
        if sys.platform =='win32':
            wget.download(file_url,out=full_path)
        else:
            os.system('wget {url} -O {name}'.format(url=file_url,name = full_path))
        name_list.append([n,file_name,file_url,full_path])
        n +=1

    with open(save_log,'w',encoding='utf-8',newline='') as fp:
        csv_writer=csv.writer(fp)
        csv_writer.writerow(['id','file_name','file_url','full_path'])
        for i in name_list:
            csv_writer.writerow(i)
        fp.close()





