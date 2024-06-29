---
title: Learn-Matlab
date: 2024-03-31 13:38:47
tags:
  - 
categories: Learn
---

Matlab

<!-- more -->

# Plot

调整整体字体：`set(gca,'FontSize',16)`
- FontSize
- FontName
- FontWeight: ‘bold’ or 'normal'

Legend图例：lgd = legend;
- 'DisplayName' 给plot绘制时添加图例`plot(x,Y_updated_cdf,'Color',cl,'DisplayName',label);`
- 忽略某条plot的图例`'HandleVisibility','off'`，或者直接`legend('off')`

## plotmatrix

```matlab
% samples_ftheta_BD(300,4) --> ax(4,4)
[~,ax] = plotmatrix(samples_ftheta_BD);
ax(1,1).YLabel.String = '\mu_E';
ax(2,1).YLabel.String = '\mu_\rho';
ax(3,1).YLabel.String = '\sigma_E';
ax(4,1).YLabel.String = '\sigma_\rho';
ax(4,1).XLabel.String = '\mu_E';
ax(4,2).XLabel.String = '\mu_\rho';
ax(4,3).XLabel.String = '\sigma_E';
ax(4,4).XLabel.String = '\sigma_\rho';

set(ax,"FontName","Times New Roman",'FontSize',16,'FontWeight','bold')
```


## gplotmatrix

```matlab
% Ysample_result(300,4), Ysample_exp(1000,4) --> axx(4,4)
[~,axx] = gplotmatrix([Ysample_result; Ysample_exp],[],[ones(300,1); 2*ones(1000,1)],[],[],10);
h = findobj('Tag','legend'); % set legend Text
set(h, 'String', {'Updated', 'Target'})

axx(1,1).YLabel.String = 'f_1';
axx(2,1).YLabel.String = 'f_2';
axx(3,1).YLabel.String = 'f_3';
axx(4,1).YLabel.String = 'f_4';
axx(4,1).XLabel.String = 'f_1';
axx(4,2).XLabel.String = 'f_2';
axx(4,3).XLabel.String = 'f_3';
axx(4,4).XLabel.String = 'f_4';
set(axx,"FontName","Times New Roman",'FontSize',16,'FontWeight','bold')
```

# 编码方式

~~低版本修改编码方式 (防止查看高版本matlab 脚本文件时出现乱码)~~ 还是会乱码
[Matlab: 修改编码方式, 如GBK-＞UTF-8_matlab编码设置utf8-CSDN博客](https://blog.csdn.net/yu1581274988/article/details/127271923)


# Parallel

- [x] 安装VMware后，Matlab2023b版本无法使用parallel，启动很慢  win(win10的问题，升级win11系统后没出现过该问题)