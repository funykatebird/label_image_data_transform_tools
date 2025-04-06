#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2025-03-30 22:08
# @update    : 2025-03-30 22:08
# @Author  : funy
# @File    : prepare_clip4videos.py
# @Des    : prepare_clip4videos.py
# @Email  :  @16.com
# @Software: PyCharm


import subprocess
import os
path = r'D:\20210706E\2024-python\pythonProjectCodeTest\Video_results'  # 待切割视频存储目录

video_list = os.listdir(path)
delta_X = 10   # 每10s切割
save_path = './save'
# mark = 0
# 获取视频的时长
def get_length(filename):
	result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
	return float(result.stdout)


for file_name in video_list:
	mark = 0
	min = int(get_length(os.path.join(path, file_name))) // 60    # file_name视频的分钟数
	second = int(get_length(os.path.join(path, file_name))) % 60    # file_name视频的秒数
	for i in range(min+1):
		if second >= delta_X:   # 至少保证一次切割
			start_time = 0
			end_time = start_time+delta_X
			for j in range((second//delta_X)+1):
			# for j in range((second // 10) + 1):
				min_temp = str(i)
				start = str(start_time)
				end = str(end_time)
				# crop video
				# 保证两位数
				# if len(str(min_temp)) == 1:
				# 	min_temp = '0'+str(min_temp)
				# if len(str(start_time)) == 1:
				# 	start = '0'+str(start_time)
				# if len(str(end_time)) == 1:
				# 	end = '0'+str(end_time)
				min_temp = str(min_temp).zfill(2)
				start = str(start_time).zfill(2)
				end = str(end_time).zfill(2)
				# 设置保存视频的名字
				if len(str(mark)) < 6:
					name = '0'*(6-len(str(mark))-1)+str(mark)
				else:
					name = str(mark)
				video_prefix = os.path.basename(file_name)
				clip_name = os.path.join(save_path, video_prefix + '_' + str(mark).zfill(4)) + '.mp4'
				command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -strict -2 {}'.format(os.path.join(path, file_name),
																					min_temp, start, min_temp, end,
																					clip_name)
				# command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -strict -2 {}'.format(os.path.join(path,file_name),
				# 								min_temp,start,min_temp,end,
				# 								os.path.join(save_path,'id_'+str(name))+'.mp4')
				mark += 1
				os.system(command)
				if i != min or (i == min and (end_time+delta_X) < second):
					start_time += delta_X
					end_time += delta_X
				elif (end_time+delta_X) <= second:
					start_time += delta_X
					end_time += delta_X
				elif (end_time+delta_X) > second:  # 最后不足delta_X的部分会被舍弃
					break # 不药要最后一段
					# 保留最后一段视频
					# start_time += delta_X
					# end_time = get_length(os.path.join(path, file_name))