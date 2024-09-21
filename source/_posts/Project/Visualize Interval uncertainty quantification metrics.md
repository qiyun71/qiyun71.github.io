---
title: Visualize Interval uncertainty quantification metrics
date: 2024-08-28 18:27:11
tags: 
categories: Project
---

使用Pygame可视化区间UQ指标

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240828182827.png)


<!-- more -->

# 区间不确定性量化指标

作用：用来量化两个区间之间的差异，方便进行优化

| UQ metrics | Source Paper                                                                                                                                                                                                                |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISF by RPO | [The sub-interval similarity: A general uncertainty quantification metric for both stochastic and interval model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022004575?via%3Dihub)  |
| IOR        | [Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S096599781731164X?via%3Dihub)                  |
| IDD        | [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation International Journal of Computational Methods](https://worldscientific.com/doi/abs/10.1142/S0219876218501037) |
| ISL        | 1-RPO                                                                                                                                                                                                                       |



![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240828183505.png)

## ISF

$ISF=\frac{1}{1+\exp\{-RPO(A,B)\}}, RPO(A,B)=\frac{min\{\overline a ,\overline b\}-max\{\underline a,\underline b\}}{max\{L(A),L(B)\}}$ 

or $\left.\mathrm{RPO}(A,B)=\begin{cases}\frac{(\overline{a}-\underline{b})}{\max\{len(A),len(B)\}},&\quad\text{Case 1,2}\\\\\frac{(\overline{a}-\underline{a})}{\max\{len(A),len(B)\}},&\quad\text{Case 3}\\\\\frac{(\overline{b}-\underline{b})}{\max\{len(A),len(B)\}},&\quad\text{Case 4}\\\\\frac{(\overline{b}-\underline{a})}{\max\{len(A),len(B)\}},&\quad\text{Case 5,6}\end{cases}\right.$


## IOR

$\mathrm{IOR}(A|B)=\begin{cases}\frac{-[(\overline{a}-a)+(\overline{b}-\underline{b})]}{\overline{b}-\underline{b}},\text{ Case1and6}\\\frac{(\overline{a}-\underline{b})-[(\underline{b}-\underline{a})+(\overline{b}-\overline{a})]}{\overline{b}-\underline{b}},\text{ Case2}\\\frac{(\overline{a}-\underline{a})-[(\underline{a}-\underline{b})+(\overline{b}-\overline{a})]}{\overline{b}-\underline{b}},\text{ Case3}\\\frac{(\overline{b}-\underline{b})-[(\underline{b}-\underline{a})+(\overline{a}-\overline{b})]}{\overline{b}-\underline{b}},\text{ Case4}\\\frac{(\overline{b}-\underline{a})-[(\underline{a}-\underline{b})+(\overline{a}-\overline{b})]}{\overline{b}-\underline{b}},\text{ Case5}\end{cases}$ 

or $\mathrm{IOR}(A|B)=\frac{\mathrm{len}(A\cap B)-\mathrm{len}((A\cup B)-(A\cap B))}{\mathrm{len}(B)}$


## IDD

$\mathrm{IDD}(A\mid B)=\frac{\mathrm{len}((A\cup B)-(A\cap B))}{\mathrm{len}(B)}$

## ISL

$L_{is}(A,B)=1-RPO(A,B)=1-\frac{min\{\overline{a},\overline{b}\}-max\{\underline{a},\underline{b}\}}{max\{len(A),len(B)\}}$

# 代码

打包：
1. `pip install pyinstaller`
2. `pyinstaller -F -w (-i icofile) filename` -F单个文件 -w隐藏cmd -i 图标文件
3. 将编写Pygame用到的所有资源包放入dist

## Version 1

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240828182827.png)

```python
import pygame
import sys
import numpy as np

def Calculate_RPO(interval1, interval2):
    interval1_len = interval1.right - interval1.left
    interval2_len = interval2.right - interval2.left
    max_len = max(interval1_len, interval2_len)

    RPO = (min(interval1.right, interval2.right) - max(interval1.left, interval2.left)) / max_len
    return RPO

def interval_similarity_function(interval1, interval2):
    RPO = Calculate_RPO(interval1, interval2)
    return 1 / (1 + np.exp(-RPO))


def interval_similarity_loss(interval1, interval2):
    RPO = Calculate_RPO(interval1, interval2)
    return 1 - RPO

def interval_overlap_ratio(interval1, interval2):
    interval1_len = interval1.right - interval1.left
    interval2_len = interval2.right - interval2.left
    if interval1.right <= interval2.left or interval1.left >= interval2.right: # case 1 and 6
        ior = - ((interval1.right - interval1.left) + (interval2.right - interval2.left)) / interval2_len
    elif interval1.left <= interval2.left and interval1.right >= interval2.right: # case 4
        ior = - (interval1.right - interval1.left) / interval2_len
    elif interval1.left >= interval2.left and interval1.right <= interval2.right: # case 3
        ior = - (interval2.right - interval2.left) / interval2_len
    elif interval1.left <= interval2.left and interval1.right <= interval2.right: # case 2
        ior = ((interval1.right - interval2.left)-(interval2.left-interval1.left+interval2.right-interval1.right)) / interval2_len
    else: # case 5
        ior = ((interval2.right - interval1.left)-(interval1.left-interval2.left+interval1.right-interval2.right)) / interval2_len
    return ior

def interval_deviation_degree(interval1, interval2):
    interval1_len = interval1.right - interval1.left
    interval2_len = interval2.right - interval2.left
    if interval1.right <= interval2.left or interval1.left >= interval2.right:
        idd = - ((interval1.right - interval1.left) + (interval2.right - interval2.left)) / interval2_len
    elif interval1.left <= interval2.left and interval1.right >= interval2.right: # case 4
        idd = - (interval2.left-interval1.left+interval1.right-interval2.right) / interval2_len
    elif interval1.left >= interval2.left and interval1.right <= interval2.right: # case 3
        idd = - (interval1.left-interval2.left+interval2.right-interval1.right) / interval2_len
    elif interval1.left <= interval2.left and interval1.right <= interval2.right:
        idd = - (interval2.left-interval1.left+interval2.right-interval1.right) / interval2_len
    else:
        idd = - (interval1.left-interval2.left+interval1.right-interval2.right) / interval2_len
    return idd

    return 1 - ior

# 初始化Pygame
pygame.init()

# 设置屏幕尺寸
screen_width = 800
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('可视化量化指标')

# 定义颜色
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

font = pygame.font.SysFont(None, 24)

# 定义滑块初始属性
slider_height = 20
slider1 = pygame.Rect(200, (screen_height - slider_height) // 2 - 10, 100, slider_height+10)
slider2 = pygame.Rect(500, (screen_height - slider_height) // 2, 100, slider_height)

# 追踪当前正在调整的滑块及其调整模式
selected_slider = None
resizing = False

# 游戏主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # 处理鼠标按下事件
        if event.type == pygame.MOUSEBUTTONDOWN:
            if slider1.collidepoint(event.pos):
                if abs(slider1.right - event.pos[0]) < 10:
                    resizing = True
                selected_slider = slider1
            elif slider2.collidepoint(event.pos):
                if abs(slider2.right - event.pos[0]) < 10:
                    resizing = True
                selected_slider = slider2

        # 处理鼠标松开事件
        if event.type == pygame.MOUSEBUTTONUP:
            selected_slider = None
            resizing = False

        # 处理鼠标移动事件
        if event.type == pygame.MOUSEMOTION:
            if selected_slider is not None and not resizing:
                selected_slider.x = event.pos[0] - selected_slider.width // 2
            elif selected_slider is not None and resizing:
                selected_slider.width = max(10, event.pos[0] - selected_slider.x)

    # print(slider2.width)
    # 防止滑块超出屏幕边界
    slider1.x = max(0, min(slider1.x, screen_width - slider1.width))
    slider2.x = max(0, min(slider2.x, screen_width - slider2.width))

    # 绘制屏幕
    screen.fill(white)
    pygame.draw.line(screen, black, (0, screen_height // 2), (screen_width, screen_height // 2), 2)
    pygame.draw.rect(screen, red , slider1)
    pygame.draw.rect(screen, blue, slider2)

    slider_A_text = font.render("A", True, black)
    slider_B_text = font.render("B", True, black)
    screen.blit(slider_A_text, (slider1.left + slider1.width//2 - 5, screen_height//2 - 10))
    screen.blit(slider_B_text, (slider2.left + slider2.width//2 - 5, screen_height//2 - 10))

    # 显示滑块1的左右端点坐标
    slider1_left_text = font.render(f"{slider1.left}", True, red)
    slider1_right_text = font.render(f"{slider1.right}", True, red)
    screen.blit(slider1_left_text, (slider1.left, slider1.top - 25))
    screen.blit(slider1_right_text, (slider1.right - slider1_right_text.get_width(), slider1.top - 25))

    # 显示滑块2的左右端点坐标
    slider2_left_text = font.render(f"{slider2.left}", True, blue)
    slider2_right_text = font.render(f"{slider2.right}", True, blue)
    screen.blit(slider2_left_text, (slider2.left, slider2.bottom + 15))
    screen.blit(slider2_right_text, (slider2.right - slider2_right_text.get_width(), slider2.bottom + 15))

    rpo = Calculate_RPO(slider1, slider2)
    rpo_text = font.render(f"Relative Position Overlap (RPO): {rpo:.4f}", True, black)
    screen.blit(rpo_text, (10, 10))

    isf = interval_similarity_function(slider1, slider2)
    similarity_text = font.render(f"Interval similarity Function (ISF): {isf:.4f}", True, black)
    screen.blit(similarity_text, (10, 40))

    isl = interval_similarity_loss(slider1, slider2)
    similarity_loss_text = font.render(f"Interval similarity Loss (ISL): {isl:.4f}", True, black)
    screen.blit(similarity_loss_text, (10, 70))

    ior = interval_overlap_ratio(slider1, slider2)
    overlap_ratio_text = font.render(f"Interval Overlap Ratio (IOR): {ior:.4f}", True, black)
    screen.blit(overlap_ratio_text, (10, 100))

    idd = interval_deviation_degree(slider1, slider2)
    deviation_degree_text = font.render(f"Interval Deviation Degree (IDD): {idd:.4f}", True, black)
    screen.blit(deviation_degree_text, (10, 130))

    pygame.display.flip()

    # 设置帧率
    pygame.time.Clock().tick(60)
```

## Version 2

- [ ] 添加重置按钮
- [ ] 优化外观