# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File   : prepare_clip_shots.py
# Time   : 2025/3/9 22:06
# Author :
# DESC   : The scripts implement video clip shot, no graphical interface has been set up;
"""

import os
import subprocess
from typing import List
from concurrent.futures import ThreadPoolExecutor

from moviepy import VideoFileClip # 1.0.3
# from moviepy.editor import VideoFileClip
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base
from scenedetect import detect, AdaptiveDetector


os.environ['PATH']= os.pathsep + r'D:\ProgramFiles\ffmpeg-7.1.1-full_build\bin'

class Config(object):
    """ Load Config file"""

    # SQL 配置
    MODE = "MYSQL"  # 选择 sql_lite

    # SQL-LITE
    DB_NAME = "video.db"

    ENGINE = create_engine(f"sqlite:///{DB_NAME}")

    MIN_DUR = 0.6

    # 线程并发数量, 根据电脑的性能进行设定;
    THREADS = 20

    # FFMPEG = False
    FFMPEG = True


Base = declarative_base()


class VideoInfo(Base):
    """Video info."""
    __tablename__ = 'video_seg'  # 表名
    id = Column(Integer, primary_key=True)  # 主键
    video_name = Column(String(200), index=True, nullable=False)  # 普通约束,非空约束。
    shot_seg = Column(String(200), index=True, nullable=False)  # 剪切点
    shot_number = Column(String(100), index=True, nullable=False)  # 第几段;


class VideoClipShots(object):
    """ Core class;
    """

    def __init__(self, video_folder, target_folder, min_number: float = 0.6):
        """Initialize basic variable information;
        """
        self.video_folder = video_folder
        self.target_folder = target_folder
        self.min_number = min_number

    def load_videos(self):
        """Load video information from the directory;"""
        videos = [i for i in os.listdir(self.video_folder) if i.endswith("mp4")]
        return [os.path.join(self.video_folder, i) for i in videos]

    @staticmethod
    def merge_durations(durations: List) -> List:
        """合并时长;
        """
        if not durations:  # 如果列表为空，直接返回空列表
            return durations

        # 如果第一项小于阈值，尝试将其加到第二项
        while durations and durations[0] < Config.MIN_DUR:
            if len(durations) > 1:  # 确保列表中至少有两项
                durations[1] += durations[0]  # 将第一项加到第二项
                durations.pop(0)  # 删除第一项
            else:  # 如果列表中只有一项且小于阈值，无法处理，直接返回空列表
                return []
        # 处理剩余的元素
        result = [durations[0]]  # 初始化结果列表，包含第一项
        for item in durations[1:]:
            if item < Config.MIN_DUR:
                result[-1] += item  # 将小于阈值的元素加到上一项
            else:
                result.append(item)  # 将大于等于阈值的元素添加到结果列表
        return result

    def get_video_clip_info(self, video_path):
        """Obtain video editing information;
        """
        scene_list = detect(video_path, AdaptiveDetector(), start_in_scene=True, show_progress=True)
        print(scene_list)
        print(len(scene_list))
        print(type(scene_list[0]))
        print(scene_list[0])
        sub_li = []
        durations = [scene[0].get_seconds() for scene in scene_list]
        if len(durations) == 1:
            durations = durations + [scene[1].get_seconds() for scene in scene_list]
        result_durations = self.merge_durations(durations)
        print("合并完成", result_durations)
        if len(result_durations) == 1:
            return [{
                "start_time": round(0.0, 2),
                "end_time": round(result_durations[0], 2),
                "shot_number": 0
            }]
        for ind, scene in enumerate(result_durations):
            if ind + 1 == len(result_durations):
                break
            else:
                sub_li.append({
                    "start_time": round(scene, 2),
                    "end_time": round(result_durations[ind + 1], 2),
                    "shot_number": ind + 1,
                    # "shot_seg": scene[1].get_frames(),
                    # 可以留作 ffmpeg 的函数;
                    # "start_time_str": scene[0].get_timecode(),
                    # "end_time_str": scene[1].get_timecode(),
                })
        return sub_li

    @staticmethod
    def clip_shots(video_path, target_path, start_time, end_time):
        """Clip the video to target_path;"""
        # clip = VideoFileClip(video_path).subclip(start_time, end_time)
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        # 保存裁剪后的视频
        clip.write_videofile(target_path, codec="libx264")

    @staticmethod
    def clip_shots_ffmpeg(video_path, target_path, start_time, end_time):
        """Editing videos using ffmpeg, The premise is that ffmpeg must be installed on your computer"""
        command = f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac {target_path}"
        try:
            subprocess.run(command, shell=True)
        except subprocess.CalledProcessError as e:
            print("ffmpeg error", e)

    def data_storage(self):
        """Store data in a database;
        """
        pass

    def export_csv(self):
        """Read database and export to CSV;"""
        pass

    def run(self):
        for video_path in self.load_videos():
            sub_li = self.get_video_clip_info(video_path=video_path)
            print(sub_li)
            for ind, sub_dict in enumerate(sub_li):
                target_name = os.path.basename(video_path)[0:-4] + f"_{str(ind).zfill(5)}.mp4"
                target_path = os.path.join(self.target_folder, target_name)
                # 单线程执行;
                print(sub_dict)
                if Config.FFMPEG:
                    self.clip_shots_ffmpeg(video_path=video_path, target_path=target_path,
                                           start_time=sub_dict.get("start_time"),
                                           end_time=sub_dict.get("end_time"))
                else:
                    self.clip_shots(video_path=video_path, target_path=target_path,
                                    start_time=sub_dict.get("start_time"),
                                    end_time=sub_dict.get("end_time"))

    def thread_run(self):
        """ 使用多线程的方式启动;
        """
        pool = ThreadPoolExecutor(Config.THREADS)
        for video_path in self.load_videos():
            sub_li = self.get_video_clip_info(video_path=video_path)
            for ind, sub_dict in enumerate(sub_li):
                target_name = os.path.basename(video_path)[0:-4] + f"_{str(ind).zfill(5)}.mp4"
                target_path = os.path.join(self.target_folder, target_name)
                # 多线程执行
                if Config.FFMPEG:
                    pool.submit(self.clip_shots_ffmpeg, video_path, target_path, sub_dict.get("start_time"),
                                sub_dict.get("end_time"))
                else:
                    pool.submit(self.clip_shots, video_path, target_path, sub_dict.get("start_time"),
                                sub_dict.get("end_time"))


if __name__ == '__main__':
    # 同级目录下创建本地记录数据库
    Base.metadata.create_all(Config.ENGINE)
    # 开始执行;
    source_folder = r"D:\20210706E\2024-python\pythonProjectCodeTest\Video_results"
    target_folder = r"./result"
    obj = VideoClipShots(video_folder=source_folder, target_folder=target_folder, min_number=0.6)

    # 单线程
    obj.run()

    # 多线程
    # obj.thread_run()
