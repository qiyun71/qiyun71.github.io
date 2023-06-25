---
title: 基于Python的工具
date: 2023-06-20 12:58:25
tags:
    - Python Practise
categories: Python
---

Python写的一些小工具

<!-- more -->

# video2image
```
import cv2
import os

def split_video_to_images(video_path, output_folder, frame_interval=1):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    # 读取视频的帧
    frame_count = 0
    saved_frame_count = 0
    while True:
        # 读取一帧
        ret, frame = video_capture.read()
        # 如果视频帧读取失败，则退出循环
        if not ret:
            break
        # 判断是否保存该帧
        if frame_count % frame_interval == 0:
            # 生成输出文件名
            # output_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            output_filename = os.path.join(output_folder, f"{frame_count:03d}.png")  # like BlendedMVS
             # 保存帧为图像文件
            cv2.imwrite(output_filename, frame)
            saved_frame_count += 1
        # 增加帧计数
        frame_count += 1
    # 释放资源
    video_capture.release()
    print(f"视频分割完成，总共分割得到 {saved_frame_count} 张图片。")
# 示例用法
video_path = "wsz.mp4"
output_folder = "output_images"
frame_interval = 1  # 保存每隔一帧的图像
split_video_to_images(video_path, output_folder, frame_interval)
```

# get_mask_from_image

To get images dataset like BlendedMVS

```
# coding:utf-8
import cv2
import os
import numpy as np

def get_mask(dir):
    dir_mask = os.path.join(dir,'mask')
    # print(dir_mask)
    if not os.path.exists(dir_mask):
        os.mkdir(dir_mask)
    img_dir = os.path.join(dir,'image')
    img_list = os.listdir(img_dir)
    for img_file in img_list:
        (filename, extension) = os.path.splitext(img_file)
        if extension == '.jpg' :
            img_file_png = filename + '.png'
            img_file_png = os.path.join(img_dir, img_file_png)
            img_file_mask = dir + '/mask/' + filename + '.png'
            img = cv2.imread(os.path.join(img_dir,img_file))
            img_mask = np.zeros(img.shape, np.uint8)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_mask[i,j,:] = [255,255,255]
            cv2.imwrite(img_file_png, img)  
            cv2.imwrite(img_file_mask, img_mask)  
            os.remove(os.path.join(img_dir,img_file))  # 删除原文件
        elif extension == '.png' :
            img_file_mask = dir + '/mask/' + filename + '.png'
            img = cv2.imread(os.path.join(img_dir,img_file))
            img_mask = np.ones(img.shape, np.uint8)
            img_mask = img_mask * 255
            cv2.imwrite(img_file_mask, img_mask)
    print('get mask done!')
```
