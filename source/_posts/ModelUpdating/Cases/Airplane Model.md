---
title: Airplane Model
date: 2025-08-13 13:20:22
tags: 
categories: ModelUpdating/Cases
Year: 
Journal:
---
## Airplane Model

> “30 manufactured airplane models and the vibration test.” ([Bi 等, 2023, p. 22](zotero://select/library/items/5JEKED2M)) ([pdf](zotero://open-pdf/library/items/5Y239HYU?page=22&annotation=NJR5RWE6))


![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240307214120.png)
![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240307213826.png)

![1-s2.0-S0888327023006921-gr18_lrg.jpg (2672×1394)|666](https://ars.els-cdn.com/content/image/1-s2.0-S0888327023006921-gr18_lrg.jpg)

| 结构参数                                                | nominal value | 实验模态频率      | Interval with 30 times measurements |
| --------------------------------------------------- | ------------- | ----------- | ----------------------------------- |
| a half wingspan a/mm                                | 300           | $f_{1}$ /Hz | [18.57, 20.96]                      |
| a wingtip chord b/mm                                | 25            | $f_{2}$ /Hz | [38.24, 41.70]                      |
| a wing structural thickness T/mm                    | 1.2           | $f_{3}$ /Hz | [84.54, 95.14]                      |
| Young’s modulus of fuselage/wing join $E_{1}$ /GPa  | 70 铝合金        | $f_{4}$ /Hz | [98.99, 109.25]                     |
| Young’s modulus of fuselage/tail joint $E_{2}$ /GPa | 70            | $f_{5}$ /Hz | [135.43, 145.13]                    |

模态试验：(数据大小: 3 x (6273, 52, 65/67) ) from "Airplane Benchmark Example"
- 3 代表 FRFs 纵轴的种类：Displacement、Velocity、Acceleration FRFs
- 测量点 (DOF)：共 52 个——翼展上有 40 个、尾翼上 12 个点
- 试验产品：共 30 个不同 sizes 的飞机机翼 （共进行了 65/67 次实验，对某些 sizes 的飞机做了多次实验）
- FRFs 频率范围：10~500Hz 共有 6273 个 frequency sample points

然后从 FRFs 中提取 modal frequency

### Equation

$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ --> $f_1,f_2,f_3,f_4,f_5$ 前 5 阶固有频率
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$ 为 a 和 b 两分布的相关系数
- $E_{1} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join
- $E_{2} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/tail joint

### Data generation

***origin code by matlab***

yichang: $a \in [0.29,0.31]$
yikuan: $b \in [0.02,0.03]$

`data1 = 0.169839009642601 + a` ==> H, W ==> E
`zuo = 0.169839009642601 - a` ==> E ==> Y

`B = zuo - (zuo + 0.1001159)/3 = 2/3 * zuo + 1/3 * (-0.1001159)`
`C = zuo - (zuo + 0.1001159)/3*2 = 1/3 * zuo + 2/3 * (-0.1001159)`
`J = data1 - (data1 - 0.43984)/3 = 2/3 * data1 + 1/3 * 0.43984`
`K = data1 - (data1 - 0.43984)/3*2 = 1/3 * data1 + 2/3 * 0.43984`

翼长x方向-->：
 zuo == B == C == -0.1001159 = 0.169839009642601 = 0.43984 == K == J == data1

`data2 = 0.109211482107639 - b` ==> O

翼宽z方向-->：
0.109211482107639 =b= data2 =0.14-b= -0.03037185035646
data2 == H1 == H2 == ... == H30 == -0.03037185035646

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250822125454.png)


data2 == L1 == 0.100878
H1 == L2 == 0.099740
...
H30 == L31 == -0.015938

```matlab
data1=(yichang+0.169839009642601);
data2=(0.109211482107639-yikuan);
H =data1;
ZUO=0.169839009642601-(data1-0.169839009642601);

E =ZUO;
W = data1;
O = data2;
B=ZUO-(ZUO+0.1001159)/3;
C=ZUO-(ZUO+0.1001159)/3*2;
J=data1-(data1-0.43984)/3;
K=data1-(data1-0.43984)/3*2;

H1=data2-(data2+0.03037185035646)/31;
H2=data2-(data2+0.03037185035646)/31*2;
H3=data2-(data2+0.03037185035646)/31*3;
H4=data2-(data2+0.03037185035646)/31*4;
H5=data2-(data2+0.03037185035646)/31*5;
H6=data2-(data2+0.03037185035646)/31*6;
H7=data2-(data2+0.03037185035646)/31*7;
H8=data2-(data2+0.03037185035646)/31*8;
H9=data2-(data2+0.03037185035646)/31*9;
H10=data2-(data2+0.03037185035646)/31*10;
H11=data2-(data2+0.03037185035646)/31*11;
H12=data2-(data2+0.03037185035646)/31*12;
H13=data2-(data2+0.03037185035646)/31*13;
H14=data2-(data2+0.03037185035646)/31*14;
H15=data2-(data2+0.03037185035646)/31*15;
H16=data2-(data2+0.03037185035646)/31*16;
H17=data2-(data2+0.03037185035646)/31*17;
H18=data2-(data2+0.03037185035646)/31*18;
H19=data2-(data2+0.03037185035646)/31*19;
H20=data2-(data2+0.03037185035646)/31*20;
H21=data2-(data2+0.03037185035646)/31*21;
H22=data2-(data2+0.03037185035646)/31*22;
H23=data2-(data2+0.03037185035646)/31*23;
H24=data2-(data2+0.03037185035646)/31*24;
H25=data2-(data2+0.03037185035646)/31*25;
H26=data2-(data2+0.03037185035646)/31*26;
H27=data2-(data2+0.03037185035646)/31*27;
H28=data2-(data2+0.03037185035646)/31*28;
H29=data2-(data2+0.03037185035646)/31*29;
H30=data2-(data2+0.03037185035646)/31*30;

L1=O+(0.100878-O)/2;
L2=H1+(0.099740-H1)/2;
L3=H2+(0.098782-H2)/2;
L4=H3+(0.097537-H3)/2;
L5=H4+(0.083165-H4)/2;
L6=H5+(0.081690-H5)/2;
L7=H6+(0.080667-H6)/2;
L8=H7+(0.078307-H7)/2;
L9=H8+(0.076320-H8)/2;
L10=H9+(0.068358-H9)/2;
L11=H10+(0.064506-H10)/2;
L12=H11+(0.061679-H11)/2;
L13=H12+(0.058957-H12)/2;
L14=H13+(0.053311-H13)/2;
L15=H14+(0.049898-H14)/2;
L16=H15+(0.047397-H15)/2;
L17=H16+(0.043198-H16)/2;
L18=H17+(0.039502-H17)/2;
L19=H18+(0.036912-H18)/2;
L20=H19+(0.032112-H19)/2;
L21=H20+(0.028419-H20)/2;
L22=H21+(0.024728-H21)/2;
L23=H22+(0.021039-H22)/2;
L24=H23+(0.017349-H23)/2;
L25=H24+(0.013647-H24)/2;
L26=H25+(0.009889-H25)/2;
L27=H26+(0.005797-H26)/2;
L28=H27+(0.001748-H27)/2;
L29=H28+(-0.002594-H28)/2;
L30=H29+(-0.005315-H29)/2;
L31=H30+(-0.015938-H30)/2;
```

***new method***

❎方法1：剪切变换

左翼节点编号：`1104:1137 1139:1165 1185:1405`
左（x0,z0）: 242
左(x_r, z_r): 1104

右翼节点编号：`1427:1717`
右（x0,z0）：196
右(x_r, z_r): 1455

*matlab代码之所以那么长，就是不清楚机翼上节点的编号或者坐标是什么，然后就要去bdf文件里找每个节点对应的坐标，然后替换值。python有个库PyNastran它可以简单地根据节点编号，直接查询对应的坐标值，也就是说只要知道节点编号就行了。节点编号好查，可以用PyNastran给的GUI框选所有节点直接就能得到。*

调整a：
if z >= z_r: rectangle area
$x' = \frac{x-x_{0}}{a}a'+x_{0}$
$z'=z$

if z < z_r: triangle area
$x'=\frac{0.14-b-(z-z_{r})}{0.14-b}\left( x_{r}'-x_{r} \right)+x=\frac{0.14-b-(z-z_{r})}{0.14-b}\left( \frac{x_{r}-x_{0}}{a}a'+x_{0}-x_{r} \right)+x$
$z'=z$

调整b:
if z >= z_r: rectangle area
$x'=x$
$z' = \frac{z-z_{0}}{b}b'+z_{0}$

if z < z_r: triangle area
$x'=x$
$z'=\frac{x-x_{0}}{x_{r}-x_{0}}\left( z_{r}'-z_{r} \right)+z=\frac{x-x_{0}}{x_{r}-x_{0}}\left( \frac{z_{r}-z_{0}}{b}b'+z_{0}-z_{r} \right)+z$

```python
left_wing_node_index = [i for i in range(1104, 1138)] + [i for i in range(1139, 1166)] + [i for i in range(1185, 1406)]
left_node_00 = nodes_dict_copy[242]
left_node_11 = nodes_dict_copy[1104]
right_wing_node_index = [i for i in range(1427, 1718)]
right_node_00 = nodes_dict_copy[196]
right_node_11 = nodes_dict_copy[1455]

wing_node_index = left_wing_node_index + right_wing_node_index

for wn_i in wing_node_index:
    node = nodes_dict_copy[wn_i]
    x = node[0]
    y = node[1]
    z = node[2]

    if wn_i in left_wing_node_index:
        x_0 = left_node_00[0]
        z_0 = left_node_00[2]
        x_r = left_node_11[0]
        z_r = left_node_11[2]
    elif wn_i in right_wing_node_index:
        x_0 = right_node_00[0]
        z_0 = right_node_00[2]
        x_r = right_node_11[0]
        z_r = right_node_11[2]

    if z >= z_r:
        x_new = (x - x_0) / a * a_new + x_0
        z_new = (z - z_0) / b * b_new + z_0
    else:
        x_new = (0.14 - b - (z - z_r)) / (0.14 - b) * ((x_r - x_0) / a * a_new + x_0 - x_r) + x
        z_new = (x - x_0) / (x_r - x_0) * ((z_r - z_0) / b * b_new + z_0 - z_r) + z

    node_new = np.array([x_new, y, z_new])
    nodes_dict_copy[wn_i] = node_new
```

❎rectangle area的线性缩放+triangle的剪切变形，剪切变形可能会使得四边形单元的内角(>150°/<30°)，网格质量太差从而无法后处理求解计算

---

方法2：基于参数化映射的平滑变形

```python
# 1104:1137 1139:1165 1185:1405
left_wing_node_index = [i for i in range(1104, 1138)] + [i for i in range(1139, 1166)] + [i for i in range(1185, 1406)]
left_node_00 = nodes_dict_copy[242]
left_node_01 = nodes_dict_copy[241]
# left_node_rect = nodes_dict_copy[1104]
left_node_10 = nodes_dict_copy[1135]
left_node_11 = nodes_dict_copy[1104]

# 1427:1717
right_wing_node_index = [i for i in range(1427, 1718)]
right_node_00 = nodes_dict_copy[196]
right_node_01 = nodes_dict_copy[197]
# right_node_rect = nodes_dict_copy[1455]
right_node_10 = nodes_dict_copy[1427]
right_node_11 = nodes_dict_copy[1455]

# P01 -- P11
# P00 -- P10

left_corners_orig = [left_node_00, left_node_10, left_node_01, left_node_11]
right_corners_orig = [right_node_00, right_node_10, right_node_01, right_node_11]

left_corners_new = [left_node_00, 
                    np.array([(left_node_10[0] - left_node_00[0]) * a_new / a + left_node_00[0], left_node_10[1], left_node_10[2]]), 
                    left_node_01, 
                    np.array([(left_node_11[0] - left_node_00[0]) * a_new / a + left_node_00[0], left_node_11[1], (left_node_11[2] - left_node_00[2]) * b_new / b + left_node_00[2]])]

right_corners_new = [right_node_00,
                    np.array([(right_node_10[0] - right_node_00[0]) * a_new / a + right_node_00[0], right_node_10[1], right_node_10[2]]), 
                    right_node_01, 
                    np.array([(right_node_11[0] - right_node_00[0]) * a_new / a + right_node_00[0], right_node_11[1], (right_node_11[2] - right_node_00[2]) * b_new / b + right_node_00[2]])]
                    
                    
nodes_dict_deformed = perform_wing_deformation(nodes_dict_copy, left_wing_node_index, left_corners_orig, left_corners_new)
nodes_dict_deformed = perform_wing_deformation(nodes_dict_deformed, right_wing_node_index, right_corners_orig, right_corners_new)
```

Basis：
$x_{1}$ <=u=> $x_m$ <=1-u=> $x_{2}$ ，其中$0<u=\frac{x_{m}-x_{1}}{x_{2}-x_{1}}<1$
线性插值得到：
$\begin{align}x_{m}=(1-u)x_{1}+ux_{2}=\frac{x_{2}-x_{1}-x_{m}+x_{1}}{x_{2}-x_{1}}x_{1}+\frac{x_{m}-x_{1}}{x_{2}-x_{1}}x_{2}\end{align}$

推广到二维的双线性插值：$x_{m,n}=(1-u)(1-v)x_{0,0}+u(1-v)x_{1,0}+(1-u)vx_{0,1}+uvx_{1,1}$

首先根据origin的四个角点坐标(00 01 10 11)计算梯形(不规则四边形同理)机翼节点的uv坐标，然后通过四个new角点坐标，双线性插值得到机翼节点的新xyz坐标。

01 -- 11
00 -- 10

```python
def perform_wing_deformation(nodes_dict: dict, wing_node_ids: list,
                                    corners_orig: np.ndarray, corners_new: np.ndarray) -> dict:
    """
    对机翼节点执行基于“位移插值”的平滑变形，以保留局部几何特征。

    参数:
        nodes_dict (dict): 包含所有节点ID和其坐标的字典。
        wing_node_ids (list): 需要变形的机翼节点的ID列表。
        corners_orig (np.ndarray): 原始机翼的4个角点坐标 (P00, P10, P01, P11)。
        corners_new (np.ndarray): 变形后机翼的4个角点坐标 (P00', P10', P01', P11')。

    返回:
        dict: 包含变形后节点坐标的新字典。
    """
    nodes_dict_deformed = nodes_dict.copy()
    
    P00_orig, P10_orig, P01_orig, P11_orig = corners_orig
    P00_new, P10_new, P01_new, P11_new = corners_new

    delta_P00 = P00_new - P00_orig
    delta_P10 = P10_new - P10_orig
    delta_P01 = P01_new - P01_orig
    delta_P11 = P11_new - P11_orig
    
    node_uv_map = {}

    print("Step 1: Calculating (u,v) parametric coordinates for each node...")
    for node_id in wing_node_ids:
        node_coord_orig = nodes_dict[node_id]
        u, v = calculate_uv_for_node(node_coord_orig, P00_orig, P10_orig, P01_orig, P11_orig)
        node_uv_map[node_id] = (u, v)

    print("Step 2: Mapping nodes to new positions using displacement interpolation...")
    for node_id in wing_node_ids:
        u, v = node_uv_map[node_id]
        
        # 使用双线性插值计算当前节点的“位移向量”
        # displacement = (1-u)(1-v)delta_P00 + u(1-v)delta_P10 + (1-u)vdelta_P01 + uvdelta_P11
        term1 = (1 - u) * (1 - v) * delta_P00
        term2 = u * (1 - v) * delta_P10
        term3 = (1 - u) * v * delta_P01
        term4 = u * v * delta_P11
        
        node_displacement = term1 + term2 + term3 + term4
        
        original_coord = nodes_dict[node_id]
        nodes_dict_deformed[node_id] = original_coord + node_displacement
    
    print("Deformation complete.")
    return nodes_dict_deformed
    
def calculate_uv_for_node(point_coord: np.ndarray, 
                            P00: np.ndarray, P10: np.ndarray, 
                            P01: np.ndarray, P11: np.ndarray) -> tuple[float, float]:
    """
    参数:
        point_coord (np.ndarray): 需要计算(u,v)的点的XYZ坐标。
        P00 (np.ndarray): (u=0, v=0) 的坐标。
        P10 (np.ndarray): (u=1, v=0) 的坐标。
        P01 (np.ndarray): (u=0, v=1) 的坐标。
        P11 (np.ndarray): (u=1, v=1) 的坐标。

        P01 -- P11
        P00 -- P10
    返回:
        tuple[float, float]: 参数化坐标 (u, v)。
    """

    z_root = P00[2]
    z_tip = P01[2]
    
    # solve v (z-direction)
    span_length = z_tip - z_root
    if abs(span_length) < 1e-9: 
        v = 0.0
    else:
        v = (point_coord[2] - z_root) / span_length

    # solve u (x-direction)
    le_point = (1 - v) * P00 + v * P01
    te_point = (1 - v) * P10 + v * P11
    
    x_le = le_point[0]
    x_te = te_point[0]
    
    chord_length_at_v = x_te - x_le
    if abs(chord_length_at_v) < 1e-9: 
        u = 0.0
    else:
        u = (point_coord[0] - x_le) / chord_length_at_v

    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    return u, v
```

### Applications

#### Traditional model updating

[飞机模型修正方案](飞机模型修正方案.md)

- BP 神经网络——FE 代理模型
- SSA 优化算法(/PSO...)
- UQ 指标：区间、子区间、椭圆、子椭圆相似度

数据：a 的均值 0.3(m)固定，不同方差 sigA 来生成 50 组数，使用 1000 组 $[0.002,0.0066]$ (m)均匀生成的方差，一共生成了 50000 组 a，然后根据相似三角形，得到 b（b 与 a 呈近似负线性相关）

```matlab
MUA=.469840228557587;sigA=0.005748;
n=50; %a
up1=normrnd(MUA,sigA,n,1);
up2=0.109211482107639-0.4151629366*(0.337217-up1+ 0.192931562662125);
up1 = (up1-0.169839009642601)*1000;
up2 = (0.109211482107639-up2)*1000;
```

- 根据 50000 组 ab，利用 Nastran 生成 50000 组的 $f_{1}\sim f_6$，然后训练 BP(代理模型)
- 使用 BP 生成 exp 的 $f_{1} \sim f_6$ 数据，n=1000, $\mu_{exp}$ 和 $\sigma_{exp}$ 确定，大小为(6, n)
- **开始修正**：(对 a 进行修正，以固定 $\mu$，修正 $\sigma$ 为例)
  - 针对 a 参数设置一个 $\sigma$ 的先验区间，比如大概为 0.X，区间设置为(0, 1)
  - SSA 寻优
    - 随机确定一个 $\sigma$，与固定的 $\mu$ 一起，生成 ab 的数据：(2, n)
    - 利用 ab 数据：(2, n)，使用 BP 生成仿真数据 sim：(6, n)
    - 根据 UQ 指标将仿真数据 sim 与实验数据 exp 进行对比
    - **重复寻优直到结束**
  - 得到 $\sigma_{updated}$ 然后跟实验 $\sigma_{exp}$ 进行对比求得误差

修正 $\mu$ 同理

#### Response-consistent MLP for interval model calibration

论文中：[基于NN飞机算例修正思路](基于NN飞机算例修正思路.md)
$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ 7 个参数 --> $f_1,f_2,f_3,f_4,f_5$ 前 5 阶固有频率
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$ 为 a 和 b 两分布的相关系数
- $E_{1} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join
- $E_{2} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/tail joint

***数据生成结果 1(废除)***

杨标生成：**(ab 的方差不能生成得太大)** [飞机算例数据集生成](飞机算例数据集生成.md)
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [1.217,4.049]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [1.025,2.101]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$ 为 a 和 b 两分布的相关系数
- $E_{1} \in [0.4,1.0]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join

共生成了 1000 组 $\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,\rho$，每组根据正态分布生成 60 个 a、b 样本，生成 60 个相同的 $E_1$ 和 $\rho$

1000 组 $\mu_a,\sigma_a,\mu_b,\sigma_b$
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240320204219.png)

**暂定 m=1000，n=100**
$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ 7 个参数在各自的区间内均匀分布，随机抽取 m=1000 组
针对抽取的 m 中的每一组 $\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ ：需要生成 n=100 组 $a,b,E_1,E_2,$ 数据
- 生成两个相关系数为 $\rho$ 的标准正态分布 $X$ 和 $Y$ [方法参考](https://blog.csdn.net/kdazhe/article/details/104599229) n = 100
  - **用联合分布生成 a 和 b**
- 根据 $\mu_a,\sigma_a,\mu_b,\sigma_b$ ，得到 a 和 b 一般正态分布 $X_{a} \sim N(\mu_{a},\sigma^{2}_{a}),X_{b} \sim N(\mu_b,\sigma^{2}_b)$ n = 100
- $a,b,E_1,E_2$：其中 a 和 b 为相关的正态分布生成，$E_1,E_2$ 为 100 个重复的固定数
- 根据 100 组 $a,b,E_1,E_2$，使用 Nastran，生成 100 组 $f_1,f_2,f_3,f_4,f_5$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240312161326.png)

总共需要使用 Nastran 进行 $1,000 \times 100 = 100,000$ 次计算
按照每次计算花费 10s 计算，共需要 100 万 s = 278 h = 11.57 day

***数据生成结果 2(MU_Airplane中数据)***

- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$ 为 a 和 b 两分布的相关系数
- 机翼厚度 $T\in [1.1,1.2]$ mm
- 连接件刚度 $E_{1} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240611155141.png)

```
# exp设置
f1 tensor(16.9921) tensor(22.6861) 19, 21
f2 tensor(37.3678) tensor(46.0559) 38, 42
f3 tensor(77.1566) tensor(102.7040) 85, 95
f4 tensor(96.6111) tensor(117.1446) 100, 110
f5 tensor(123.9420) tensor(152.1165) 135, 145
```

***数据生成结果 3(RC-MLP)***

最终使用数据 600x1000：(还是有一点偏差，但是基本上训练集可以涵盖实验的数据，$f_{2}$在38Hz时也有数据只不过很少)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250813163804.png)

***Method 1(不合理)*** ❎

 **单次测量频率预测单组参数(反向代理模型——确定的)**
根据多组参数的值 $\{a,b,E_{1},E_{2},\rho\}_{i=1}^{1e6}$ 来计算参数的均值和方差 or 区间边界

***Method 2*** ❎未完成

**直接预测均值和方差 or 区间边界(反向代理模型——随机)**

神经网络训练：
- 输入：5x100 大小的数组（100 组 $f_1,f_2,f_3,f_4,f_5$）
- 输出/标签标签：7x1 的向量（$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ ）

训练思路：输入 5x100 大小的数组通过 FC(全连接层)计算得到中间向量，然后 reshape 成 3 通道图片，使用 CNN 处理图片，提取特征并解码为 7x1 的向量

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240312161313.png)

方法缺点：训练完成的 NN，必须要输入固定大小的数组

***Method 3***  interval model calibration via RC-MLP

输入响应区间，RC-MLP输出参数区间，并通过differentiable interval uncertainty propagation实现响应区间的监督

##### 问题讨论

对于小区间例如 $[302.2234,302.2323]$ 预测的效果不好 $[302.2415,302.2204]$

下界和上界会预测反，这是由于小区间的上界下界本来就相近，即使预测反或者预测得偏差一点，也不会有很大的 loss

```
## a_label
tensor([
302.4267, 302.2234, 302.6144, 308.2171, 309.9898, 309.9898, 309.9898,
......
293.0252, 304.0647, 304.0647], device='cuda:0')

tensor([
302.4413, 302.2323, 302.6641, 308.2196, 309.9922, 309.9953, 309.9919,
......
293.0608, 304.1299, 304.1080], device='cuda:0')
```

```
## a_pred
tensor([
302.4215, 302.2415, 302.6063, 308.2065, 309.9922, 309.9915, 309.9943,
......
293.0485, 304.0324, 304.0781], device='cuda:0')

tensor([
302.4599, 302.2204, 302.7219, 308.2034, 309.9624, 309.9611, 309.9670,
......
293.2524, 304.1639, 304.0648], device='cuda:0')
```


~~**解法**: 换成网络去预测区间中心和区间半径，但是可能会出现大数吃小数的问题~~
**解法**: 取消小区间的情况，将小区间当作误差来处理

- $a\in [290,310]$
- $b\in [20,30]$
- $T\in [1.1,1.2]$ mm

| a_lower | a_upper | b_lower | b_upper | T_lower  | T_upper  |
| ------- | ------- | ------- | ------- | -------- | -------- |
| 290,308 | 292,310 | 20,29.8 | 20.2,30 | 1.1,1.18 | 1.12,1.2 |

**取消小区间后还是有问题**：对 T(mm)的修正误差很大

真正问题：**发现是数据的问题**(ab 固定，T 上下改变，输出的频率几乎不变)，**重新生成数据** --> 解决