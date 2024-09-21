---
title: Python Scripts
date: 2023-06-20 12:58:25
tags:
  - Python
  - Practise
categories: Project
---

Python写的一些小工具

<!-- more -->

# video2image
```python
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

```python
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

# 利用ffmpeg的图片转视频

NeRO的Relight后生成图片，需要生成一个可循环的视频

```python
import os
from PIL import Image

just_video = True
# 输入文件夹路径和输出文件夹路径
input_folder = "E:\\BaiduSyncdisk\\NeRF_Proj\\NeRO\\data\\relight\\bear-neon\\"
image_file_suffix = None

if not just_video:
    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_files.sort(key=lambda x:int(x.split('.')[0]))
    # print(image_files)
    image_num =  len(image_files)
    print('==>图片数量: ' + str(image_num))

    out_image_th = 0
    image_file_suffix = image_files[0].split('.')[-1]
    # 倒序复制图片
    for image_file in image_files:
        out_image_name = image_num * 2 - 1 - out_image_th
        out_image_name = str(out_image_name) + '.' + image_file_suffix
        print("==>正在处理图片：", image_file , '-->' , out_image_name)
        out_image_th += 1
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(input_folder, out_image_name)

        # 打开图片并倒序保存
        image = Image.open(input_path)
        image.save(output_path)

    print("==>处理完成")


import subprocess
# ffmpeg -r 15 -i %3d.jpg video.avi -vf  "scale=ih*16/9:ih:force_original_aspect_ratio=decrease,pad=ih*16/9:ih:(ow-iw)/2:(oh-ih)/2"
frame = 120
video_name = 'nero_relight.mp4'
if image_file_suffix is None:
    image_file_suffix = 'png'
cmds=[
    'ffmpeg', '-r' , str(frame) , '-i' , input_folder + '%d.'+image_file_suffix ,
    '-vf' , "scale=ih*16/9:ih:force_original_aspect_ratio=decrease,pad=ih*16/9:ih:(ow-iw)/2:(oh-ih)/2" , video_name
]
subprocess.run(cmds)
```
