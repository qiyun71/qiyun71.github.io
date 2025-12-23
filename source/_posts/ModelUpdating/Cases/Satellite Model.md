---
title: Satellite Model
date: 2025-08-13 13:20:50
tags: 
categories: ModelUpdating/Cases
Year: 
Journal:
---
## Satellite Model

>  [qiyun71/SatelliteModelUpdating: Satellite FE Model Updating](https://github.com/qiyun71/SatelliteModelUpdating)
>  “The satellite model and its sampling locations.” ([Zhang 等, 2021, p. 1484](zotero://select/library/items/5Q59AHMI)) ([pdf](zotero://open-pdf/library/items/GKWTRL7Q?page=6&annotation=X9864HQ6))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231225211854.png)

主体结构6个参数为：
- **主弹性模量**$\theta_1$ 70Gpa，
- 主密度 $\theta_2$  ，密度2.7x $10^{3} kg/m^{3}$ (英文论文) or **适配器厚度** 1mm(本 1)
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度 $\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度 $\theta_6$ 2.5mm
  - 其中 $\theta_{4}、\theta_6$ 为常数，其他为待修正参数，$\theta_2$ 训练效果很差，直接去掉了，可能造成影响(本 1)
  - 其他参数：模型高 1250mm，中心筒直径 400mm，顶板边长 600mm，剪切板长 1000mm，宽 400mm

卫星模型选取了 11 个节点，并测量了水平 X 和竖直 Y 两个方向的加速度频响数据，频率范围为 ==0-50Hz==（实际使用30Hz），频率间隔为 0.5。得到最终输入网络的频响图像尺寸为 2×11×61，对应标签形状为 4×1。

FR 数据转三维数组(多通道图片)：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231225212852.png)


### 传统优化方法

- 有限元方法获取数据集耗费时间长，使用一个代理模型来代替有限元计算模型。
- 根据随机样本 $X_{s}$ 通过代理模型得到模拟响应 $Y_{s}$，动力学实验得到实验响应 $Y_{e}(f_{1} ... f_{6})$
- 用子区间相似度计算模拟样本 $Y_{s}$ 和试验样本 $Y_{e}$ 之间的值，并作为目标函数。
- 用麻雀搜索算法将目标函数迭代寻优，得到修正后的均值 $\mu$ 和标准差 $\sigma$

目标函数：子区间相似度、标准差椭圆

### 基于NN方法

#### UCNN

***数据集生成与预处理***

数据集：
- 输入：测量的卫星模型加速度频响数据，将加速度频响数据经处理转化为频响图像
- 标签：选取的不确定性参数($\theta_1,\theta_2,\theta_3,\theta_5$)组成向量

**数据集生成**
- 先在一定范围内生成均匀生成 10000 组待修正参数
  - **弹性模量** 50.0~90.0GPa
  - ~~**适配器厚度** 0.50~1.50mm~~
  - ~~**中心筒厚度** 1.00~3.00mm~~
  - ~~**剪切板厚度** 1.00~3.00mm # 本科生 1 数据集中生成的是 0.50~1.50mm~~
  - 顶板厚度 1.50~3.50mm

```markdown
/data/train 训练数据分为25组，每一组400条，共10000次仿真数据
- fre-400-1
  - sita1.xlsx #400x1
  - sita2.xlsx
  - ...
  - sita6.xlsx
  - xiangying.xlsx #80800x11 频响数据，每101×2行 为一条，对应0-50Hz 101个频率点x、y两个方向的数据. 从第0行 (x方向)开始, 偶数列为x方向,奇数列为y方向
- fre-400-2
- fre-400-3
- fre-400-4
- ...
- fre-400-25

/data/test 400条测试集，用于测试网络的精度
- fre-400
  - sita1.xlsx #400x1
  - sita2.xlsx 
  - ...
  - sita6.xlsx
  - xiangying.xlsx #80800x11
```


**卫星算例.bdf** 不同结构参数生成结构特征量FR

```bdf file
$ Elements and Element Properties for region : Shear_Panels
PSHELL   1       1      .003     1               1

- 36  行 .003 Shear_Panels 厚度 theta5
- 429 行 .002 Central_Cylinder 厚度 theta3
- 666 行 .001 Adapter 厚度 theta2 本来应该是密度2.7
- 723 行 .002 Upper_platform 厚度 theta6
- 864 行 .001 Lower_platform 厚度 theta4
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

***网络结构***

UCNN 单向卷积

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231227133859.png)

***评价指标***

测试集错误率：$错误率 = \frac{|预测值-标签GT|}{标签GT}$

#### Other NN（Improvement）

***数据集生成***


**将excel数据转换成处理更快的numpy数组**
处理后训练集包含两类npy文件:
- `train_FRdata{i}.npy` 存储FRF数据 --> 400条数据,共25组 (其中一组损坏,实际24组) ==> (9600, 2, 61, 11)
- `label_{i}.npy` 存储 参数$\theta$ 数据 --> 400条数据,共25组 ==> (9600, 4)
测试集包含: `test_FRdata.npy` (400, 2, 61, 11) 与 `label.npy` (400, 4)
试验集包含: `test_FRdata.npy` (1000, 2, 61, 11) 与 `label.npy` (1000, 4) 

excel： n为data sets，训练集和测试集都是25个文件(25个sita1.xlsx)，每个文件中n为400. 试验集只有一个文件，n为1000
- **sita1.xlsx** 一列，n行
- **sita2.xlsx**
- **sita3.xlsx**
- sita4.xlsx
- **sita5.xlsx**
- sita6.xlsx
- **xiangying.xlsx**  80800行(400x101x2) 11列

npy: 
- test_FRdata.npy (data sets, 2, 61, 11) xy两方向、前61个频率点、选取的11个测量参考节点 
- label.npy (data sets, 4)

|              | $\theta_{1}$                    | $\theta_{2}$            | $\theta_{3}$            | $\theta_{4}$      | $\theta_{5}$            | $\theta_{6}$      |
| ------------ | ------------------------------- | ----------------------- | ----------------------- | ----------------- | ----------------------- | ----------------- |
| 物理参数         | 主弹性模量 E                         | Adapter 厚度              | Central_Cylinder 厚度     | Lower_platform 厚度 | Shear_Panels 厚度         | Upper_platform 厚度 |
| 名义值          | 70Gpa                           | 1mm                     | 2 mm                    | 1 mm              | **2 mm**                | 2.5 mm            |
| In Paper     | 70Gpa                           | 1mm                     | 2 mm                    | 2 mm 审稿后修改为1      | **1 mm**                | 2.5 mm            |
| 训练集          |                                 |                         |                         |                   |                         |                   |
| Excel        | $7 (\times 10^{10}Pa)$          | 0.001 (m)               | 0.002 (m)               | 0.001 (m)         | **0.001 (m)**           | 0.0025 (m)        |
| Code(python) | $7 (\times 10^{10}Pa)$          | 1 (mm)                  | 2 (mm)                  | No                | 1 (mm)                  | No                |
| 训练集 范围       | $[5.0, 9.0] (\times 10^{10}Pa)$ | $[0.5, 1.5] (mm)$       | $[1.0, 3.0] (mm)$       |                   | $[0.5, 1.5] (mm)$       |                   |
| 试验集          |                                 |                         |                         |                   |                         |                   |
| 试验数据 范围      | $E\sim N(7, 0.3^2)$             | $T_{1}\sim N(1, 0.2^2)$ | $T_{2}\sim N(2, 0.4^2)$ | $T_{4}=0.001$     | $T_{3}\sim N(1, 0.2^2)$ | $T_{5}=0.0025$    |
|              |                                 |                         |                         |                   |                         |                   |
|              |                                 |                         |                         |                   |                         |                   |


数据集详细记录日志，data/
- FE：生成数据集的matlab程序，需要调用nastran
- test：测试网络精度数据集（均匀）
- test_1：Exp实验数据(Target目标数据)，测试网络修正结果的数据集（正态）
- test_1_pred：根据**test_1**预测的参数结果，输入nastran得到FR
- test_1_pred_norm：根据**test_1归一化后**预测的结果，输入nastran得到FR
- train：训练网络数据集（均匀）
- train_npy、test_npy、test_1_npy、test_1_pred_npy：excel转为npy格式
- train_npy copy：train转为npy时选取的频率范围为0~50Hz，其他为30Hz，用于绘制数据处理例子的流程图
- train_npy_24_3theta、test_npy_24_3theta为本科生的修正3个参数的数据

将FRF归一化后，主要数据文件夹： data/
- train_npy_norm 训练集 --> 训练网络
- test_npy_norm 测试集 --> 测试网络的训练效果
- test_1_npy_norm??? 实验集/目标集 --> 验证网络的校准效果(model calibration)
- test_1_npy??? 试验集没有归一化，为了最后的模型验证

添加噪声
- test_1_npy_norm1000_noise？ --> Trained NN --> /pred.npy --> FEA --> test_1_pred_norm_npy1000_noise?
- test_1_npy1000_noise？(Target without normalization)  <-->  test_1_pred_norm_npy1000_noise? (Calibrated)

***网络结构***

```ad-tldr UCNN 
**UCNN**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231227133859.png)

**MLP**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111150150.png) 

**MLP_S**
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111150239.png) 

**VGG16**
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111145858.png) 

**ResNet50**

![resnet.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/picturesresnet.png)
```


***实验记录***

```ad-info
20231227-111217: 10, 0.9 UCNN
20231227-111738: 25, 0.99 UCNN
20231227-160116: 25, 0.99 MLP 4x128
20231227-160651: 25, 0.99 UCNN 3.5min
20231227-162720: 25, 0.99 MLP 8x256
20231227-163116: 25, 0.99 MLP 8x256 **400** epoch 4.0min
# 20240102-164844: 25, 0.99 MLP_S 14个4x256的MLP 200 epoch 9min
# 20240102-170301: 25, 0.99 MLP_S 14个2x256的MLP 200 epoch 5.6min
20240102-203115: 25, 0.99 MLP_S 14个2x256的MLP **400** epoch 15.1min
20240102-205253: 25, 0.99 MLP_S 14个4x256的MLP **400** epoch 23.27min
20240102-211805: 25, 0.99 UCNN **400** epoch 7min
# 20240111-160514: 25, 0.99 VGG11 200 epoch (由于图片太小使用VGG11)
20240111-162059: 25, 0.99 VGG11 400 epoch 26.34 min
20240113-190506: 25, 0.99 ResNet50 400 epoch 36.1 min

# 测试两个是不是过拟合了，error rate随epoch变化趋势
20240115-092119: 25, 0.99 UCNN 1000 epoch
20240115-111929: 25, 0.99 VGG11 1000 epoch

# 24x400 数据 --> theta1,3,5
20240116-142116: 25, 0.99 VGG11 400 epoch
20240116-153711: 25, 0.99 UCNN 400 epoch

# 24x400 数据 --> theta1,2,3,5
## 25, 0.99 | 400 epoch
@20240116-164606_ucnn	
@20240116-170957_vgg	
@20240116-180846_resnet	
@20240116-193319_mlp	
@20240116-194754_mlp_s

## 25, 0.99 | 1000 epoch
@20240116-213751_mlp	
@20240116-222413_ucnn	
@20240116-231550_vgg	
@20240117-014321_mlp_s	
@20240117-065536_resnet

## 25, 0.99 | 500 epoch
@20240118-232440_ucnn
@20240118-234300_vgg
@20240119-005355_mlp
@20240119-010645_mlp_s
@20240119-021612_resnet

## 50, 0.99 | 500 epoch
@20240119-225843_ucnn
@20240119-231920_vgg
@20240120-003157_mlp
@20240120-004544_mlp_s
@20240120-015533_resnet

## 100, 0.99 | 500 epoch
@20240120-171537_ucnn
@20240120-175110_mlp
@20240120-180848_mlp_s
@20240120-195338_vgg
@20240120-220032_resnet

## 200, 0.99 | 500 epoch
@20240120-234329_ucnn
@20240121-000302_mlp
@20240121-001554_mlp_s
@20240121-012709_vgg
@20240121-023818_resnet

## 300, 0.99 | 500 epoch
@20240121-040844_ucnn
@20240121-042747_mlp
@20240121-044111_mlp_s
@20240121-055239_vgg
@20240121-070346_resnet
``` 

不论 MLP 还是 UCNN，中间 sita3 预测的误差都很大

```ad-info
(satellite) PS D:\0Proj\ModelUpdating\satellite_UCNN> python run.py --test --resume outputs\@20231227-163116_mlp\400_mlp.pth --net mlp   
error_rate=:0.05787282592434471=(0.19222232587635518/3.321460855007172)
=====================
error_rate_each0=:0.04022682424495547=(0.28112122416496277/6.9884021282196045)
error_rate_each1=:0.12927590329299216=(0.25523426607251165/1.9743375182151794)
error_rate_each2=:0.04024536597068831=(0.040311464481055735/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python run.py --test --resume  outputs\@20240102-203115_mlp_s\400_mlp_s.pth --net mlp_s
error_rate=:0.03934166444889552=(0.13067179843783377/3.321460855007172)
=====================
error_rate_each0=:0.028448767469817206=(0.1988114271312952/6.9884021282196045)
error_rate_each1=:0.0800526909024898=(0.15805103108286858/1.9743375182151794)
error_rate_each2=:0.03509528007871003=(0.03515292014926672/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python run.py --test --resume  outputs\@20240102-205253_mlp_s\400_mlp_s.pth --net mlp_s # 过拟合了，训练集上的损失比较小，但是测试集上的loss比较大
error_rate=:0.03124857323569491=(0.10379091277718544/3.321460855007172)
=====================
error_rate_each0=:0.01964181444371989=(0.13726489786058665/6.9884021282196045)
error_rate_each1=:0.07125288617244709=(0.14067724645137786/1.9743375182151794)
error_rate_each2=:0.03337576539351412=(0.03343058135360479/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --test --resume outputs\@20240102-211805_ucnn\400_ucnn.pth
error_rate=:0.02647251309147154=(0.08792741596698761/3.321460855007172)
=====================
error_rate_each0=:0.022187629130279763=(0.15505607463419438/6.9884021282196045)
error_rate_each1=:0.03919995023153533=(0.07739393245428801/1.9743375182151794)
error_rate_each2=:0.031280856565034834=(0.03133223187178373/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --net vgg --test --resume outputs\202401\@20240111-162059_vgg\400_vgg.pth
error_rate=:0.02620232199857664=(0.0870299868285656/3.321460855007172)
=====================
error_rate_each0=:0.016490224581949674=(0.11524032056331635/6.9884021282196045)
error_rate_each1=:0.05937809189634055=(0.11723239459097386/1.9743375182151794)
error_rate_each2=:0.02857031368553574=(0.028617237228900194/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --test --resume outputs\202401\@20240113-190506_resnet\400_resnet.pth --net resnet
error_rate=:0.12867085358744207=(0.42737520337104795/3.321460855007172)
=====================
error_rate_each0=:0.1063032449177=(0.7428898230195046/6.9884021282196045)
error_rate_each1=:0.22810707873746822=(0.4503603637218475/1.9743375182151794)
error_rate_each2=:0.08872962392607951=(0.08887535240501165/1.0016423881053924)
```



### 问题讨论

[基于NN的卫星算例问题](基于NN的卫星算例问题.md)

- 根据目标参数expT的均值(7,1,2,1)和标准差(0.3,0.2,0.4,0.2)，生成服从正态分布的m组exp数据
  - 生成的数据根据torch.mean和torch.std再计算会有一定偏差，比如表中正态分布生成m=100组exp数据和m=1000组exp数据，1000组数据经过再次计算得到的均值和标准差，相较于100组偏差更小
- 对m组的实验FR数据进行修正，得到m组修正后的参数，然后根据torch.mean和torch.std再计算得到均值和标准差
  - 论文中是跟原始的（用于生成分布的）均值和标准差进行对比，但是实际应该与再计算的均值和标准差进行比较才更准确


**使用model.eva()的结果**：

|                | expT | 100组exp    | pred1  | pred2  | pred3  | 1000组exp   | pred1  | pred2  | pred3  |
| -------------- | ---- | ---------- | ------ | ------ | ------ | ---------- | ------ | ------ | ------ |
| $\mu_E$        | 7    | **6.9781** | 6.9613 | 6.9696 | 6.9764 | **6.9981** | 6.9762 | 6.9833 | 6.9874 |
| $\mu_{T_1}$    | 1    | **1.0415** | 1.0320 | 1.0368 | 1.0120 | **1.0061** | 0.9951 | 1.0041 | 1.0197 |
| $\mu_{T_2}$    | 2    | **2.0619** | 2.0641 | 2.0557 | 2.0562 | **2.0008** | 2.0005 | 1.9978 | 1.9730 |
| $\mu_{T_3}$    | 1    | **1.0046** | 1.0055 | 1.0031 | 1.0062 | **0.9965** | 0.9968 | 0.9941 | 0.9940 |
| $\sigma_{E}$   | 0.3  | **0.3235** | 0.3385 | 0.3333 | 0.3436 | **0.2940** | 0.3263 | 0.3161 | 0.3219 |
| $\sigma_{T_1}$ | 0.2  | **0.1961** | 0.1935 | 0.1899 | 0.1731 | **0.1905** | 0.1942 | 0.1884 | 0.1861 |
| $\sigma_{T_3}$ | 0.4  | **0.3903** | 0.3965 | 0.3904 | 0.3790 | **0.3828** | 0.3918 | 0.3858 | 0.3789 |
| $\sigma_{T_3}$ | 0.2  | **0.1954** | 0.1959 | 0.1928 | 0.1916 | **0.1932** | 0.1928 | 0.1912 | 0.1899 |
| ER_T(%)        | 0    | 2.82       | 3.31   | 3.67   | 5.30   | 1.93       | 2.31   | 2.56   | 3.58   |
| ER(%)          |      | 0          | 1.14   | 1.07   | 3.26   | 0          | 2.12   | 1.40   | 2.21   |
|                |      |            |        |        |        |            |        |        |        |
|                |      |            |        |        |        |            |        |        |        |

说明
- pred1：归一化处理FR数据、500epoch训练结果
- pred2：归一化处理FR数据、1000epoch训练结果
- pred3：不归一化，500epoch训练结果
- ER\_T(%) = $\frac{1}{8} \sum\limits_{i=1}^8 \frac{|pred_i-expT_i|}{expT_i}$ x100 (%)，**论文中关注的是这个指标**
- ER(%) = $\frac{1}{8} \sum\limits_{i=1}^8 \frac{|pred_i-exp_i|}{exp_i}$ x100 (%)，**实际应该是这个指标更准确一点**

由平均误差，可以看出：
1. **归一化处理**pred1比没有归一化处理pred3，训练的网络精度高~~（主要是方差精度，excel可以看出）~~
2. **epoch** 越多，ER越低(这是正常的)，但是ER_T会更高，这是由于ER_T是(pred)与生成分布时地均值和标准差(expT)对比的，exp与expT的误差与pred与exp的误差累积了(两个随机相互影响)
3. **数据量m**越多，ER_T更低，但是ER反而会更高(未归一化pred3反而更低)，**此外数据量多了还有一个问题**：

这是训练集 label（这副图的exp和updated是同一个变量，两者都是label）
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308143958.png)

使用测试集测试网络训练的精度（均匀分布生成的测试集）
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308144116.png)

实验数据得到修正结果(m=1000)（正态分布）
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308144229.png)

实验数据(m=100)，数据量少的时候，这种特征还不明显
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308144522.png)


更加明显，$T_1$和$T_3$集中在了(等间隔0.1的位置)，在微小变动时，
![1709889649879.jpg|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/1709889649879.jpg)

**问题**：$T_3$预测的分布规律奇怪，虽然也是正态分布，但是大多数值集中在了1.0、1.1等间隔0.1的位置，$T_1$也有类似情况但是没有$T_{3}$这么明显。
**分析**：神经网络对$T_3$不敏感，网络预测将$T_3$小数点后两位的数省略了（只识别出了1.0、1.1mm等规律的值），且这一现象与**是否对FR归一化**无关，没有FR归一化的预测也有这个现象：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308165855.png)

***猜想1（错误）***

与**结构参数归一化**有关，将FR和结构参数，也就是NN的输入和输出都归一化后，结果：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308201527.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308201704.png)


***猜想2（错误）***

训练过程就识别不到其他位置的参数了，训练的1个epoch，发现除了边缘无法很好预测，其他地方也很满（中间有的地方也很空，但不会出现聚集在特定值的现象）

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308203251.png)


***猜想3（错误）***

可能FR数据生成有问题，数据生成根据4个参数的均值和方差，生成1000组正态分布数据，然后根据1000组四参数，使用Nastran生成FR数据
可能当改变剪切板厚度$T_3$时，生成的FR数据过于相似，例如将$T_{3}=1.12$或者$T_{3}=1.13$等生成的FR数据跟$T_{3}=1.1$生成的FR数据近似，NN分辨不出来

400组数据的测试集中也有这个现象：
![image.png|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240309164056.png)

   
***真正原因***

在test和exp的时候，添加了`model.eval()`代码，会将网络中的BN层和Dropout层关闭（训练时是开启的），(test数据batchsize是直接设置成整个test dataset的大小，可以)不使用eval，相当于在训练集上过拟合了以后，可以更好地对先验区间中的实验数据进行更好的预测

**不使用**model.eval()的结果：

|                | expT | 100组exp    | pred1  | pred2  | pred3  | 1000组exp   | pred1  | pred2    | pred3  |
| -------------- | ---- | ---------- | ------ | ------ | ------ | ---------- | ------ | -------- | ------ |
| $\mu_E$        | 7    | **6.9781** | 6.9893 | 6.9781 | 6.9948 | **6.9981** | 6.9983 | 6.9962   | 7.0011 |
| $\mu_{T_1}$    | 1    | **1.0415** | 1.0336 | 1.0435 | 1.0505 | **1.0061** | 0.9921 | 0.9995   | 1.0215 |
| $\mu_{T_2}$    | 2    | **2.0619** | 2.0647 | 2.0583 | 2.042  | **2.0008** | 1.9988 | 1.9951   | 1.9789 |
| $\mu_{T_3}$    | 1    | **1.0046** | 1.0045 | 1.0033 | 1.0036 | **0.9965** | 0.9966 | 0.9953   | 0.9963 |
| $\sigma_{E}$   | 0.3  | **0.3235** | 0.3673 | 0.344  | 0.3365 | **0.2940** | 0.3299 | 0.3232   | 0.323  |
| $\sigma_{T_1}$ | 0.2  | **0.1961** | 0.1956 | 0.1937 | 0.1947 | **0.1905** | 0.1957 | 0.1919   | 0.1925 |
| $\sigma_{T_3}$ | 0.4  | **0.3903** | 0.3977 | 0.3965 | 0.3885 | **0.3828** | 0.3934 | 0.3919   | 0.3852 |
| $\sigma_{T_3}$ | 0.2  | **0.1954** | 0.1988 | 0.1914 | 0.1928 | **0.1932** | 0.1925 | 0.1925   | 0.1926 |
| ER_T(%)        | 0    | 2.82       | 4.13   | 3.86   | 3.61   | 1.93       | 2.34   | **2.30** | 2.80   |
| ER(%)          |      | 0          | 2.31   | 1.46   | 1.09   | 0          | 2.45   | 1.81     | 1.82   |

论文关注的ER_T，其中1000组exp的pred2效果更好(归一化处理FR数据、1000epoch训练结果)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240309201551.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240309201706.png)

#### 程序问题

matlab 生成FRFs的程序问题，并不是11个节点的加速度幅值

```f06
      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
0          118      G     -1.893191E-02   2.744646E-02  -1.148516E-02  -5.853226E-02  -1.082212E-04   1.105988E-01
                           4.502489E-07  -6.520212E-07   2.733072E-07   1.392180E-06   2.492752E-09  -2.660447E-06
0          237      G      1.893236E-02   2.744877E-02   1.148540E-02  -5.853223E-02   1.081836E-04   1.106056E-01
                          -4.502597E-07  -6.520763E-07  -2.733128E-07   1.392179E-06  -2.491843E-09  -2.660618E-06
0          255      G      4.782312E-08   1.635185E-02   2.330007E-08  -5.961377E-02  -2.843844E-07   9.934657E-02
                          -1.020500E-12  -3.889889E-07  -4.840865E-13   1.414843E-06   5.956261E-12  -2.356463E-06
0          381      G      4.910261E-10   5.798578E-02  -1.367990E-09   1.038435E+00   6.459272E-07   9.908993E-02
                          -9.960504E-15  -1.379422E-06   2.324609E-14  -1.986041E-05  -1.446684E-11  -2.354878E-06
0          416      G      3.527358E-09  -6.694251E-01   2.848346E-09  -7.681564E-01  -1.106894E-08  -2.531782E+00
                          -5.386538E-14   1.314468E-05  -6.025993E-14   1.584632E-05   2.442403E-13   5.072122E-05
0          446      G     -6.481036E-09   1.164889E-01   5.534058E-09  -1.548480E-01  -5.967503E-08   1.012482E-01
                           1.533396E-13  -2.771816E-06  -1.133209E-13   3.226627E-06   1.407623E-12  -2.393049E-06
0          521      G      6.857802E-09  -5.797762E-02  -5.915352E-09  -2.194655E+00  -1.031755E-06   9.962211E-02
                          -1.611111E-13   1.379118E-06   1.601390E-13   6.070103E-05   2.778342E-11  -2.378132E-06
0          556      G     -2.086969E-07   1.367283E+00  -1.148249E-09   1.063863E+00   1.623497E-07  -4.669222E+00
                           5.754569E-12  -3.771298E-05   2.932038E-14  -2.973616E-05  -4.530125E-12   1.288144E-04
0          587      G      2.343825E-02  -1.353126E-02   1.413584E-02  -1.120405E-01  -8.874521E-02   1.027448E-01
                          -5.575564E-07   3.218754E-07  -3.363756E-07   2.844023E-06   2.379223E-06  -2.463951E-06
0          728      G      4.197688E-11   6.445141E-03  -5.843937E-09  -5.821381E-02  -9.521039E-08   9.667463E-02
                          -8.090375E-16  -1.533222E-07   1.454717E-13   1.387308E-06   1.444107E-12  -2.299757E-06
0          827      G     -6.594866E-09   5.828727E-02  -9.611829E-09  -5.830408E-02  -9.875006E-09   9.698171E-02
                           1.556223E-13  -1.386983E-06   2.443410E-13   1.387674E-06   3.523319E-13  -2.307504E-06
```

matlab读取.f06文件时，获取T1列，然后存入motai.txt中的一行

```matlab
for i=1:21 
   line=fgetl(fid);
   wlc=[wlc;str2double(line(27:40))];
end
% 然后存入motai.txt
fclose(fid);
fid=fopen('motai.txt','a');
fprintf(fid,'%3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d\r\n',wlc);
fclose(fid);

A =importdata('motai.txt');
% a=A(1:202,[3 19 11 7 10 13 15 1 21 5 17]);
a=A(:,[3 19 11 7 10 13 15 1 21 5 17]);
disp(size(a));
% save('sita1.mat','Z1')
% save('sita6.mat','Z2')
% save('sita2.mat','Z3')
% save('sita3.mat','Z4')
% save('sita4.mat','Z5')
% save('sita5.mat','Z6')
% save('xiangying.mat','a')
xlswrite('xiangying.xlsx',a);
```

这句话`a=A(:,[3 19 11 7 10 13 15 1 21 5 17]);` 相当于读取的11个数是11个点的实部/虚部
顺序为：

| Node number  | 1        | 2   | 3   | 4   | 5    | 6   | 7   | 8   | 9   | 10  | 11  |
| ------------ | -------- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
| Point ID     | 237      | 728 | 446 | 381 | 416  | 521 | 556 | 118 | 827 | 255 | 587 |
| Point number | 2 (默认实部) | 10  | 6   | 4   | 5 虚部 | 7   | 8   | 1   | 11  | 3   | 9   |
| Line order   | 3        | 19  | 11  | 7   | 10   | 13  | 15  | 1   | 21  | 5   | 17  |

**解决**：
- [ ] 要么重新修改训练数据集(之后的论文要保证FRFs的一致性)
- [x] 要么修改代码 --> 这个问题

```python
if need_select_node:
    print("Select node")
    print(test_FRdata.shape)
    new_test_FRdata[:, :, :, 0] = test_FRdata[:, :, :, 2-1].real
    new_test_FRdata[:, :, :, 1] = test_FRdata[:, :, :, 10-1].real
    new_test_FRdata[:, :, :, 2] = test_FRdata[:, :, :, 6-1].real
    new_test_FRdata[:, :, :, 3] = test_FRdata[:, :, :, 4-1].real
    new_test_FRdata[:, :, :, 4] = test_FRdata[:, :, :, 5-1].imag
    new_test_FRdata[:, :, :, 5] = test_FRdata[:, :, :, 7-1].real
    new_test_FRdata[:, :, :, 6] = test_FRdata[:, :, :, 8-1].real
    new_test_FRdata[:, :, :, 7] = test_FRdata[:, :, :, 1-1].real
    new_test_FRdata[:, :, :, 8] = test_FRdata[:, :, :, 11-1].real
    new_test_FRdata[:, :, :, 9] = test_FRdata[:, :, :, 3-1].real
    new_test_FRdata[:, :, :, 10] = test_FRdata[:, :, :, 9-1].real
    test_FRdatas.append(new_test_FRdata)
```
