---
title: Folding Fin
date: 2025-08-13 13:21:27
tags: 
categories: ModelUpdating/Cases
Year: 
Journal:
---
## Folding Fin

非线性的折叠翼（folding fins）：
- 前板
- 后板
- 转轴


>  [飞行器非线性振动试验与模型修正研究进展](https://lxjz.cstam.org.cn/cn/article/doi/10.6052/1000-0992-24-011)

>  [基于热/力试验的折叠舵连接刚度与颤振分析](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2022.27927) 中国运载火箭技术研究院
>  [航天飞行器折叠翼锁紧机构力学模型](http://www.jasp.com.cn/hkdlxb/article/pdf/preview/20220032.pdf)


> [有间隙折叠舵面的振动实验与非线性建模研究](https://lxxb.cstam.org.cn/cn/article/doi/10.6052/0459-1879-19-119) 哈工大
> [Nonlinear system identification framework of folding fins with freeplay using backbone curves - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1000936122001030?via%3Dihub) 
> [Nonlinear aeroelastic analysis of the folding fin with freeplay under thermal environment - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1000936120302089#f0005)



>  [ABAQUS 非线性有限元分析与实例](https://www.ecsponline.com/yz/BEAE93A3A1F674DE1A799288D185CE221000.pdf) 补全——ABAQUS 非线性有限元分析与实例

非线性：接触、材料、几何

非线性接触的应用：
- 各类装配体结构的强度计算;
- 塑性成型计算;
- **配合连接计算**(过盈;间隙)
- 螺栓连接结构计算
- 结构振动与冲击接触等。


待修正——连接刚度——>体现在那里(刚度？摩擦系数？尽量两个)
- 先做线性修正参数
- 再用修正后的参数做非线性(model validation)

修改激振力和阻尼比
```python
model = BDF()
model.read_bdf(copy_path_bdf)

######## force ########
force_card_list  = model.loads[load_id_to_modify]
force_card = force_card_list[0]

print(f"origin force_card:{force_card}")

force_card.mag = force[i]

print(f"modified force_card:{force_card}")

######## damping ########
for m in range(len(model.materials)):
    material_card = model.materials[m+1]
    print(f"origin material_card:{material_card}")
    if material_card.type == 'MAT1':
        # damping ratio
        material_card.ge = damping[i, m]
    print(f"modified material_card:{material_card}")

model.write_bdf(copy_path_bdf, enddata=True)
```


可修正参数：

| 部件       | 前翼板            | 后翼板            | 转轴             |
| -------- | -------------- | -------------- | -------------- |
| 材料参数     | 铝              | 铝              | 钢              |
| 弹性模量$E$  | 68-73 GPa      | 68-73 GPa      | 200 GPa        |
|          |                | 65, 75         | 190, 210       |
| 泊松比$\nu$ | 0.33           | 0.33           | 0.3            |
|          |                | 0.3, 0.36      | 0.27, 0.33     |
| 密度$\rho$ | 2.7 $g/cm^{3}$ | 2.7 $g/cm^{3}$ | 7.8 $g/cm^{3}$ |
|          |                | 2.5, 3.0       | 7.5, 8.0       |
| 几何参数     | $a_{1}$        | $a_{2}$        | 直径$d$          |
|          | 80 mm          | 160 mm         | 6 mm           |
|          | 75, 85         | 150, 170       | 4, 6           |
|          | $b_{1}$        | $b_{2}$        | 位置$p$          |
|          | 425 mm         | 200 mm         | 20 mm 10~26    |
|          | 415, 435       | 190, 210       | None           |


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250918122814.png)

