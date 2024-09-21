# Install

## COLMAP环境变量导致 matlab一直报错

```
This application failed to start because no Ot platform plugin could beinitialized. Reinstalling the application may fix this problem.
Available platform plugins are: minimal, offscreen, webgl, windows.
```
### 解决办法，删除环境变量，每次启动终端使用set设置一下

set QT_QPA_PLATFORM_PLUGIN_PATH=D:\0Proj\Tools\COLMAP\lib\plugins
colmap gui