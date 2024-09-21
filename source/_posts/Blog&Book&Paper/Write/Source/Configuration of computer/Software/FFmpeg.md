# ffmpeg

将多张图片合成视频output.mp4
`ffmpeg -r 10 -f image2 -i %03d.png output.mp4`
- -r在-i前：每秒输入10帧图片转为视频
  - -r在-i后：每秒输入25帧图片转为视频，视频为每秒10帧
- -f image2
- -i 输入图片文件名