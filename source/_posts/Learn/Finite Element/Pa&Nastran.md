---
title: Pa&Nastran
date: 2024-12-25 10:21:16
tags:
  - 
categories: Learn/Finite Element
---

Patran前处理 & Nastran计算

<!-- more -->

## Modeling

Solidworks建立好的模型导出为`.x_t`格式

> [How To Convert Solid To Surface Body In SolidWorks - YouTube](https://www.youtube.com/watch?v=Fx0jX_7aJHM)

抽壳：将Solid 转换成Surface：
- **Delete Face**
- Offset

Patran中进行网格划分-->材料属性设置-->边界条件和负载设置，得到bdf文件
Nastran利用bdf文件进行进行求解


## Patran

Preferences --> Geometry 单位制 inches/m/mm，不同的单位下材料特性的尺度会有所不同

eg(钢板): E = 210GPa，$\rho = 7860 Kg/m^{3}$
常用的是mm制：
- Elastic Modulus/Shear Modulus：(210000)MPa/(83000)MPa
- Poisson Ratio(0.25)：注意设置了E和G后，Poisson Ratio会自动计算(根据钢板算例测试出来的)
- Density：(7.8599998E-09) $Mg/mm^3$
- $Kgf \cdot s^{2} /mm^{4}$ $(Kg/m^{3} =kgf \cdot s^{2}/m^{4})$ $(1Kgf = 9.8N = 9.8 Kg \cdot m/s^{2})$

> [patran中数据的输入输出单位 - 百度文库](https://wenku.baidu.com/view/9f2572f37c1cfad6195fa7d6.html?fr=income1-doc-search&_wkts_=1714641483975&wkQuery=patran%E4%B8%AD%E6%95%B0%E6%8D%AE%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E5%8D%95%E4%BD%8D)

$1Mg = 10^{3}Kg= 10^{6}g=10^{9}mg$ 

![1564932c6e363ac45d0902478c047d4dc859a6a2.png@1192w.avif (1192×457)|666](https://i2.hdslb.com/bfs/article/1564932c6e363ac45d0902478c047d4dc859a6a2.png@1192w.avif)



![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240708153849.png)

### Mode Frequency

-->生成 **bdf**

1. File --> New --> `*.db`
2. Menu --> Preferences --> Geometry --> 1000.0 (Millimeters) --> **Apply**
3. File --> Import --> `*.x_t` --> **Apply** (Display --> Smooth Shade)
4. Meshing -->(RHS) Mesh --> Solid -->(Main window) select solid --> Automatic Calculation  --> **Apply**
5. Properties --> Isotropic -->(RHS) *Material Name* --> Input properties (Elastic Modulus: 210000MPa, Shear Modulus: 83000, Density: 7.86E-09 Kg/m^3) --> **Apply**
6. Properties --> Solid --> Propert Set Name -->(RHS) Input properties --> Mat Prop Name select *gangban(Material Name)* -->  Select Application Region -->(Main window) select solid --> Add --> **Apply**
7. Analysis --> Solution Type --> NORMAL MODES --> Solution Type --> Solution Parameters --> Results Output Format(XDB or OP2) --> Subcase --> Subcase Parameters --> Number of Desired Roots : 20 --> **Apply** Run nastran --> Get .bdf

### Frequency Response Function

在Nastran动力学如频率响应和瞬态响应计算中，有模态法和直接法两种计算方法
- [MSC Patran-Nastran 2021应用实例—直接法频率响应分析（案例八）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1eZ4y1A7F6/?buvid=XXA7157A2595D8DAA7D27D13F5911BB415F26&from_spmid=search.search-result.0.0&is_story_h5=false&mid=5E%2FE0HONObjFbvgpVZnCxw%3D%3D&plat_id=116&share_from=ugc&share_medium=android&share_plat=android&share_session_id=bbb8e522-8d38-49e4-8f32-d742b36779a7&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1735035868&unique_k=AoMBTyz&up_id=624875096&vd_source=1dba7493016a36a32b27a14ed2891088)
- [MSC Patran-Nastran 2021应用实例—模态法频率响应分析（案例九）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ob4y1R7Yk/?share_source=copy_web&vd_source=a372600987f019e355fd7480b9a36a68)

1. File --> New --> `*.db`
2. Menu --> Preferences --> Geometry --> 1000.0 (Millimeters) --> **Apply**
3. File --> Import --> `*.x_t` --> **Apply** (Display --> Smooth Shade)
4. Meshing -->(RHS) Mesh --> Solid -->(Main window) select solid --> Automatic Calculation  --> **Apply**
5. Properties --> Isotropic -->(RHS) *Material Name* --> Input properties (Elastic Modulus: 210000MPa, Shear Modulus: 83000, Density: 7.86E-09 Kg/m^3) --> **Apply**
6. Properties --> Solid --> Propert Set Name -->(RHS) Input properties --> Mat Prop Name select *gangban(Material Name)* -->  Select Application Region -->(Main window) select solid --> Add --> **Apply**

mesh --> Properties create --> set properties for solid 一样的步骤
  
之后：
直接法：约束--> 非空间场(频率范围？) --> 时间依赖的工况 --> 载荷 --> 分析
1. Loads/BCs --> Nodal --> Displacement Constraint --> (RHS)*New Set Name* --> Input Data --> T1 T2 T3 R1 R2 R3 all is 0 (全约束) -->  OK --> Select Application Region --> Select FEM --> 一条边上的点 Node --> Add --> OK --> Apply
2. Loads/BCs --> LBC Fields 选第一排第一个 --> (RHS) 非空间场Non Spatial + Tabular Input --> Field Name --> Table Definition select Frequency --> Input Data --> f-1 2e1 1 | f-2 1e3 1 --> OK--> Apply
3. Loads/BCs --> Load Cases --> Type: Time Dependent --> Load Case Name --> Input Data --> Select Individual Loads/BCs --> 选择之前设置的约束 --> OK --> Apply 
4. Loads/BCs --> Nodal --> Force --> New Set Name --> Input Data -->  F1 F2 F3 < 0 0 1> 要看自己的力方向设置 --> 同行Time/Freq. Dependence 选择非空间场 --> Select Application Region --> FEM select nodes --> Add --> OK --> Apply
5. Analysis --> Solution Type --> FREQUENCY RESPONSE --> **Formulation Direct 直接法** --> Solution Type --> Solution Parameters --> Results Output Format(XDB or OP2) --> Subcase --> 选择自定义时间依赖的case --> Subcase Select --> **Apply** Run nastran --> Get .bdf

nastran算不出来

模态法：
1. Loads/BCs --> Nodal --> Displacement Constraint --> (RHS)*New Set Name* --> Input Data --> T1 T2 T3 R1 R2 R3 all is 0 (全约束) -->  OK --> Select Application Region --> Select FEM --> 一条边上的点 Node --> Add --> OK --> Apply
2. Loads/BCs --> LBC Fields 选第一排第一个 --> (RHS) 非空间场Non Spatial + Tabular Input --> Field Name --> Table Definition select Frequency 非空间场
  1. 与压力有关的数据 p --> Field Type: Real --> Input Data --> f-1 2e1 1 | f-2 1e3 1 --> OK--> 
  2. 与集中力有关的数据 f --> Field Type: Complex --> Input Data --> row1 2e1 1 -4.5e1 | row2 1e3 1 -4.5e1 --> OK--> 
3. Loads/BCs --> Load Cases --> Type: Time Dependent --> Load Case Name --> Input Data --> Select Individual Loads/BCs --> 选择之前设置的约束 --> OK --> Apply 
4. Loads/BCs --> Element Uniform --> Pressure --> New Set Name --> Input Data -->  压力+集中力载荷
  1. Pressure: -0.3 --> 同行Time/Freq. Dependence 选择非空间场 p --> Select Application Region --> 节点全选--> Add --> OK --> Apply
  2. F1 F2 F3 < 0 0 1> 要看自己的力方向设置 --> 同行Time/Freq. Dependence 选择非空间场 --> Select Application Region --> FEM select nodes --> Add --> OK --> Apply
5. Analysis --> Solution Type --> FREQUENCY RESPONSE --> **Formulation Modal 模态法** --> Solution Type --> Solution Parameters (Mass Calculation: Coupled) --> Results Output Format(XDB or OP2) --> Subcase --> 选择自定义时间依赖的case --> Subcase Select  --> **Apply** Run nastran --> Get .bdf

### 共节点


## Nastran

[如何用matlab被nastran给整的明明白白 PART 1 KNOW YOUR ENEMY——.bdf文件 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/33538970)

Nastran的Python库：[Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)

Debug: [Nastran Error List 1. | PDF](https://www.scribd.com/doc/70652924/Nastran-Error-List-1)

`D:\Software\Nastran\Nastran_install\bin\nastranw.exe *.bdf`

### bdf文件

输出文件格式：`PARAM    POST    0`
- POST 0 不输出op2文件
- POST,-1 生成 OP2 文件，但不包含几何数据
- **POST,1 生成 OP2 文件，并包含几何数据**
- POST,2 生成 OP2 文件，并包含几何数据和优化数据. (在卫星算例中需要改成POST 2，但是在折叠翼算例中只需要POST 1 即可，***可能是版本不同的原因***)

输出节点定义：`SET xxx= 节点编号`
- 比如定义11个节点， `SET 1 = 118,237,255,381,416,446,521,556,587,728,827`

**Material ID** 是材料的唯一标识符，用于在 BDF 文件中定义材料的属性。包括弹性模量、泊松比、密度等
`MAT1    1       2.1+11   .3      7800.` 材料ID1，弹性模量E，泊松比nu，密度rho
修改材料属性：
```python
material_id = 1
if material_id in bdf_model.materials:
    material = bdf_model.materials[material_id]
    print(f"材料 ID {material_id} 的属性: {material}")
    material.E = 3.0e11  # 修改弹性模量
```

**Properties** 是单元属性的定义，用于描述单元的几何和材料特性。单元类型、厚度、材料 ID 等
`PSHELL  1       1       .01` 属性ID1，引用材料的ID1，厚度0.01m
修改单元属性：
```python
property_id = 1
if property_id in bdf_model.properties:
    property = bdf_model.properties[property_id]
    property.t = 0.02  # 修改厚度
    print(f"属性 ID {property_id} 的定义: {property}")
    print(f"引用的材料 ID: {property.mid}")  # 获取属性引用的材料 ID
```

阻尼表
`TABDMP1  ID   TYPE   F1   G1   F2   G2   ...   FN   GN`
- ID：阻尼表的唯一标识符（整数）。
- TYPE：阻尼类型，通常为 G（模态阻尼比）。
- F1, F2, ..., FN：频率值（Hz）。
- G1, G2, ..., GN：对应频率的阻尼比。

在模态分析（如 `EIGRL` 或 `EIGRA`）中，通过 `SDAMPING` 参数引用定义的阻尼表。
`EIGRL    SID     V1      V2      ND      MSGLVL  MAXSET  SHFSCL  NORM`

>  [Simcenter Nastran Basic Dynamic Analysis User's Guide](https://iberisa.wordpress.com/wp-content/uploads/2021/01/simcenter-nastran-basic-dynamics-user-guide.pdf)


#### Question

- nastran没有输出指定的op2文件 --> 取消勾选HDF5格式输出
- nastran无法得到运行结果：
  - 内存不足，在log中有`OPEN CORE Allocation Failed` [Nastran求解只有log文件没有结果如何解决-百度经验](https://jingyan.baidu.com/article/6b182309f8c46cba59e15961.html) --> `D:\Software\Nastran\Nastran_install\conf\NAST20200.rcf`修改 memory 为`0.4*physical`


#### 卫星算例.bdf

不同结构参数生成结构特征量FR

```bdf file
$ Elements and Element Properties for region : Shear_Panels
PSHELL   1       1      .003     1               1

- 36  行 .003 Shear_Panels 厚度 theta5 --> P1
- 429 行 .002 Central_Cylinder 厚度 theta3 --> P2
- 666 行 .001 Adapter 厚度 theta2 本来应该是密度2.7 --> P3
- 723 行 .002 Upper_platform 厚度 theta6 --> P4
- 864 行 .001 Lower_platform 厚度 theta4 --> P5
- 还有一个不用修改的P6 是 Navigation_Platform：一个实现导航功能的平台
- 1020行 7.   mat_N 弹性模量  theta1   
- 1023行 7.   mat_CC 弹性模量  theta1  
- 1026行 7.   mat_L 弹性模量  theta1  
- 1029行 7.   mat_SP 弹性模量  theta1   
- 1032行 7.   mat_U 弹性模量  theta1  
- 主弹性模量不包括 mat_A 适配器的材料属性
```

- **主弹性模量**$\theta_1$ 70Gpa，
- **主密度** $\theta_2$  ，密度2.7x $10^{3} kg/m^{3}$ (英文论文) or 适配器厚度 1mm(本 1)
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度 $\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度 $\theta_6$ 2.5mm

### FEA二次开发(Python)

Ansys: 
> [Ansys与Python：r/ANSYS --- Ansys with Python : r/ANSYS](https://www.reddit.com/r/ANSYS/comments/14pak2j/ansys_with_python/)
> [PyAnsys — PyAnsys](https://docs.pyansys.com/version/stable/)

Nastran: 
> [Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)


[PyNastran](../../Project/PyNastran.md) 项目代码

pynastranGUI 支持对有限元模型和仿真结果的可视化，支持的模型文件格式：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241016122523.png)


# Other

## 打开MBT

Model Browser Tree : `toggleModelTree()` in command

## FE & Blender

可否指教一下blender如何渲染abaqus求解文件odb？

odb写个脚本导出obj文件，然后导进blender一张张渲染就好