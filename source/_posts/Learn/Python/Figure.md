---
title: Python Plot Figure
date: 2020-02-21 11:37:01
summary: Python plt 的学习过程，可以让数据可视化，还可以展示图像
toc: true
categories: Learn
tags:
  - Python
---

matplotlib | seaborn

`import matplotlib.pyplot as plt`
`import seaborn as sns`

<!--more-->

# 其他

## 布局

plt.tight_layout()
通常在绘制多个子图时使用，用于自动调整图形中的子图布局，以避免子图之间的重叠或太过拥挤

# Property

```python
color: 
rcolor = '#AA232E'
bcolor = '#3C53A6'

# plot parameters
plt.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})

scatter: omega1, omega2, label='Target',s=140,marker='x',c=c1,alpha=alpha,linewidths=5
- alpha=0.7
- c1 = 'r' or 'b'
  
label: fontsize=24,labelpad = 10,fontname='Times New Roman'

axs.tick_params(labelsize=22, pad=5)
axs.grid(linewidth=1.5, linestyle='--')

plt.legend(loc='upper center',fontsize=26,ncol=4, bbox_to_anchor=(1.25, 1.25))
plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.07, right=0.98, top=0.92, bottom=0.1)
```


# Seaborn

## distplot

```python
import seaborn as sns

# 核密度函数
sns.distplot(prep_data, hist=False, kde=True, kde_kws={'linewidth': 5}, label='Original', color="red")
sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linewidth': 5, 'linestyle':'--'}, label='Synthetic', color="blue")
```

# Matplotlib

## plt.clf 动态图片展示

```python
# 动态图片展示
plt.clf()

plt.pause(0.01)
plt.ioff()
```


## plt.plot

`format_string的内容`

![plot1.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/plot1.png)

![plot2.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/plot2.png)

![plot3.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/plot3.png)


`**kwargs`

\*\*kwards：
color 颜色
linestyle 线条样式
marker 标记风格
markerfacecolor 标记颜色
markersize 标记大小 等等

```py
# plt.plot(x,y,format_string,**kwargs)
# x轴数据，y轴数据，控制曲线格式的字符串format_string颜色字符，风格字符，和标记字符
plt.plot([1,2,3,6],[4,5,8,1],'r-s')
plt.show()
```

**展示**

![plot4.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/plot4.png)


## plt.imshow

**BGR 用 RGB 打开**

```py
import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('lena.jpg',cv2.IMREAD_COLOR)

#method1
b,g,r=cv2.split(img)
img2=cv2.merge([r,g,b])
plt.imshow(img2)
plt.show()

#method2
img3=img[:,:,::-1]
plt.imshow(img3)
plt.show()

#method3
img4=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img4)
plt.show()
```

## plt.figure 

定义画布大小，然后用 plot 画图

```py
# plt.figure() # 用来画图,自定义画布大小
fig1 = plt.figure(num='fig111111', figsize=(10, 3), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
# 名字,宽*高，dpi图像每英寸长度内的像素点数 一般75，
plt.plot([1,2,3],[2,2,3])
plt.show()
plt.close()
```

![figure1.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/figure1.png)


## plt.subplot 子图

将 figure 设置的画布大小分成几个部分，参数‘221’表示 2(row)x2(colu),即将画布分成 2x2，两行两列的 4 块区域，1 表示选择图形输出的区域在第一块，图形输出区域参数必须在“行 x 列”范围，此处必须在 1 和 2 之间选择——如果参数设置为 subplot(111)，则表示画布整个输出，不分割成小块区域，图形直接输出在整块画布上

```py
plt.subplot(222) 
plt.plot(y,xx)    #在2x2画布中第二块区域输出图形
plt.show()
plt.subplot(223)  #在2x2画布中第三块区域输出图形
plt.plot(y,xx)
plt.subplot(224)  # 在在2x2画布中第四块区域输出图形
plt.plot(y,xx)

# 子图
plt.add_subplot(221)
plt.add_subplot(222)
```

## plt.scatter


`matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None,alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)`

- x，y：表示的是大小为(n,)的数组，也就是我们即将绘制散点图的数据点
- s:是一个实数或者是一个数组大小为(n,)，这个是一个可选的参数
- c:表示的是颜色，也是一个可选项。默认是蓝色'b',表示的是标记的颜色，或者可以是一个表示颜色的字符，或者是一个长度为 n 的表示颜色的序列等等
- marker:表示的是标记的样式，默认的是'o'
- cmap:Colormap 实体或者是一个 colormap 的名字，cmap 仅仅当 c 是一个浮点数数组的时候才使用。如果没有申明就是 image.cmap
- norm:Normalize 实体来将数据亮度转化到 0-1 之间，也是只有 c 是一个浮点数的数组的时候才使用。如果没有申明，就是默认为 colors.Normalize
- vmin,vmax:实数，当 norm 存在的时候忽略。用来进行亮度数据的归一化
- alpha：实数，0-1 之间
- linewidths:也就是标记点的长度

> [参考教程](https://blog.csdn.net/m0_37393514/article/details/81298503)

## plt.hist

_可以将高斯函数这些画出来_

`n, bins, patches = plt.hist(arr, bins=10, normed=0, facecolor='black', edgecolor='black',alpha=1，histtype='bar')`
hist 的参数非常多，但常用的就这六个，只有第一个是必须的，后面四个可选

- arr: 需要计算直方图的一维数组
- bins: 直方图的柱数，可选项，默认为 10
- normed: 是否将得到的直方图向量归一化。默认为 0
- facecolor: 直方图颜色
- edgecolor: 直方图边框颜色
- alpha: 透明度
- histtype: 直方图类型，‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’
  返回值 ：
  n: 直方图向量，是否归一化由参数 normed 设定
  bins: 返回各个 bin 的区间范围
  patches: 返回每个 bin 里面包含的数据，是一个 list

```
mu, sigma = 0, .1
s = np.random.normal(loc=mu, scale=sigma, size=1000)
a,b,c = plt.hist(s, bins=3)
print("a: ",a)
print("b: ",b)
print("c: ",c)

plt.show()

结果：

a:  [ 85. 720. 195.]         #每个柱子的值
b:  [-0.36109509 -0.1357318   0.08963149  0.31499478]   #每个柱的区间范围
c:  <a list of 3 Patch objects>       #总共多少柱子
```

## Property

### plt.xlim

```py
plt.xlim(0,1000)  #  设置x轴刻度范围，从0~1000         #lim为极限，范围
plt.ylim(0,20)   # 设置y轴刻度的范围，从0~20
```

### plt.xticks

```py
fig2 = plt.figure(num='fig222222', figsize=(6, 3), dpi=75, facecolor='#FFFFFF', edgecolor='#FF0000')
plt.plot()
# np.linspace 创建等差数列
plt.xticks(np.linspace(0,1000,15,endpoint=True))
# 设置x轴刻度
plt.yticks(np.linspace(0,20,10,endpoint=True))
plt.show()
plt.close()
```

**展示**

![xticks.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/xticks.png)


### ax2.set_title('xxx')设置标题，画图

`plt.xlabel()` `plt.ylabel()`xy 轴标签

```py
#产生[1,2,3,...,9]的序列
x = np.arange(1,10)
y = x
fig = plt.figure()
ax1 = fig.add_subplot(221)


#设置标题
ax1.set_title('Scatter Plot1')
plt.xlabel('M')
plt.ylabel('N')
ax2 = fig.add_subplot(222)
ax2.set_title('Scatter Plot2clf')
#设置X轴标签
plt.xlabel('X')           #设置X/Y轴标签是在对应的figure后进行操作才对应到该figure
#设置Y轴标签
plt.ylabel('Y')
#画散点图
ax1.scatter(x,y,c = 'r',marker = 'o')          #可以看出画散点图是在对figure进行操作
ax2.scatter(x,y,c = 'b',marker = 'x')
#设置图标
plt.legend('show picture x1 ')
#显示所画的图
plt.show()
```

**展示**

![title1.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/title1.png)


