---
title: Nastran by python
date: 2024-05-10 11:25:26
tags: 
categories: Project
---
Python--> Nastran

使用PyNastran库对 Patran&Nastran有限元分析软件进行二次开发，可以自动对目标模型进行有限元分析

> [Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)


<!-- more -->

# BDF file

BDF文件是使用Patran对模型进行前处理产生的，包括划网格、定义结构参数、添加约束等操作，
Patran在设置好后进行分析时，还会输出一个bdf文件，然后在cmd中运行`nastran xxx.bdf`指令即可调用Nastran进行有限元求解。

- xref：Cross-referencing，可以很方便地追踪对象，如果xref=False，只会返回数据对象的raw data，需要分别对bdf的element和node做索引

```python
# read
model = BDF()
bdf = read_bdf(bdf_filename, xref=False,debug=False)
cquad = bdf.elements[1]
nid1 = cquad.nodes[0]
n1 = bdf.nodes[nid1]
cd4 = n1.cd
c4 = bdf.coords[cd4]

bdf_xref = read_bdf(bdf_filename, xref=True,debug=False)
bdf_xref.elements[1].nodes_ref[0].cd_ref.i
```


```python
bdf.nodes.items()
node.get_position() # get xyz
```

## 修改材料参数

材料参数的修改比较简单：
直接定位到材料参数所在的line，然后直接替换掉原来的字符即可

```python
bdf_copy = open(bdf_copy_path, 'r+')
lines = bdf_copy.readlines()
lines[8682] = lines[8682].replace('210000.', f'{E:.3f}')
lines[8682] = lines[8682].replace('83000.', f'{G:.3f}')
bdf_copy.seek(0)
bdf_copy.writelines(lines)
bdf_copy.close()
```

## 修改几何尺寸

对于简单的结构，例如长方体钢板，通过缩放修改BDF文件中nodes的坐标，就可以实现修改结构的厚度/长度等几何尺寸参数

问题1：只有单个零件的结构比较容易修改，但是如果结构中有许多子结构，在BDF文件中很难区分哪些是子结构1，哪些是子结构2的 
😵尝试1：可以给每个子结构设置不同的坐标系coords，(Patran)

> [nodes Module — pyNastran 1.4 1.4 documentation](https://pynastran-git.readthedocs.io/en/1.4/reference/bdf/cards/pyNastran.bdf.cards.nodes.html#pyNastran.bdf.cards.nodes.GRID)

```python
eid100 = bdf_xref.elements[100]
print(eid100)
print("nodes = %s" % eid100.nodes)
print("--node0--\n%s" % eid100.nodes_ref[0])
print("--cd--\n%s" % eid100.nodes_ref[0].cd)
print("cd.cid = %s" % eid100.nodes_ref[0].cd_ref.cid)
```

| 1    | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9    |
| ---- | --- | --- | --- | --- | --- | --- | --- | ---- |
| GRID | NID | CP  | X1  | X2  | X3  | CD  | PS  | SEID |

😵要不还是学学ABAQUS，看看是否可行。

# Results file

结果文件可以在Patran中设置，可以以多种不同的格式输出：
- OP2
- XDB
- HDF5，如果要用其他两种格式，需要取消勾选输出该文件 (可能是软件bug？😂)
- 此外还固定输出f06文件，方便用户查看结果

## f06

f06文件中存储的是仿真的结果，可以直接用txt打开查看，从f06中提取modal frequency比较方便

直接根据特定的字符串，定位到模态频率的列表上，然后读取特征频率对应的行

### modal frequency

```python
def get_modes(f06_copy_path:str):
    """
    Get the modes from the f06 file
    """
    find_txt = "NO.       ORDER                                                                       MASS              STIFFNESS"
    line_num = 0
    f = open(f06_copy_path, 'r')
    lines_mode = f.readlines()
    while True:
        line = lines_mode[line_num]
        if find_txt in line:
            break
        line_num += 1 # python 从0开始计数
    # print(line_num)
    modes = []
    for i in range(line_num+7, line_num+7+5):
        line_mode = lines_mode[i]
        modes.append(float(line_mode[67:80]))
    return modes
```

## op2

>  [OP2 Introduction — pyNastran 1.5-dev 1.5-dev documentation](https://pynastran-git.readthedocs.io/en/latest/quick_start/op2_demo.html#why-use-the-op2-why-not-use-the-f06-pch-file)

op2与f06的内容相同，只是格式不同，相较于f06解析困难，OP2 非常结构化

op2转f06：`test_op2 -f solid_bending.op2`
- `-f` tells us to print out `solid_bending.test_op2.f06` 不必重新运行得到f06文件?？
- `-c` flag disables double-reading of the OP2

导入需要的包，并读取op2文件

```python
import os
import copy
import numpy as np
np.set_printoptions(precision=2, threshold=20, suppress=True)

import pyNastran
pkg_path = pyNastran.__path__[0]

from pyNastran.utils import print_bad_path
from pyNastran.op2.op2 import read_op2
from pyNastran.utils import object_methods, object_attributes
from pyNastran.utils.nastran_utils import run_nastran

import pandas as pd

pd.set_option('display.precision', 3)
np.set_printoptions(precision=3, threshold=20)

# op2_filename = "./data/SteelPlate/gangban_frf-modal-v1.op2"
op2_filename = "./data/SteelPlate/sxban.op2"

op2 = read_op2(op2_filename, build_dataframe=True, debug=False)
```

查看文件内容

```python
print(op2.get_op2_stats())

op2_results.cstm: 
CSTM: 
headers_str = dict_keys(['cid', 'cid_type', 'unused_int_index', 'unused_double_index', 'ox', 'oy', 'oz', 'T11', 'T12', 'T13', 'T21', 'T22', 'T23', 'T31', 'T32', 'T33']) 
headers_ints = dict_values([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) 
data = None 

displacements[1] 
isubcase = 1 
type=ComplexDisplacementArray ntimes=101 nnodes=3023, table_name=OUG1 
data: [t1, t2, t3, r1, r2, r3] shape=[101, 3023, 6] dtype=complex64 
node_gridtype.shape = (3023, 2) 
sort1 
freqs = [ 0. 2.5 5. ... 245. 247.5 250. ]; dtype=float32 

spc_forces[1] 
isubcase = 1 
type=ComplexSPCForcesArray ntimes=101 nnodes=3023, table_name=OQG1 
data: [t1, t2, t3, r1, r2, r3] shape=[101, 3023, 6] dtype=complex64 
node_gridtype.shape = (3023, 2) 
sort1 
freqs = [ 0. 2.5 5. ... 245. 247.5 250. ]; dtype=float32

print(op2.get_op2_stats(short=True))

displacements[1] spc_forces[1]
```

### FRF

- freqency为频率的采样点值
- disp_data为每个频率下的位移大小

```python
displacements = op2.displacements[1]
freqency = displacements.freqs # freq: n_freq
disp_data = displacements.data # n_freq, n_nodes, 6 (6: tx, ty, tz, rx, ry, rz)
# print(displacements.node_gridtype.shape)
# print(disp_data)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
dircections_label = ["x", "y", "z"]
for i in range(10):
    for j in range(3):
        ax[j%3].plot(freqency, disp_data[:, i, j],label=f"Node{i+1}")
        ax[j%3].set_xlabel("Frequency(Hz)")
        ax[j%3].set_ylabel("Displacement(m)")
        ax[j%3].set_title(f"Displacement in {dircections_label[j]} direction")
        if j == 0:
            ax[j%3].legend(loc='upper right')
    # y ticks label
plt.show()
```

# Example (Steel Plate Structure)

Generate simulation datasets by FE (solidworks & patran & nastran & python)

Model and Simulate to get `.bdf` file
- solidworks project Part.SLDPRT, export `.x_t` file
- patran 材料属性参数设置+网格划分+约束+负载，并定义好输出的类型(mode shape/FRF...) to get `.bdf` file(其中保存着有限元模型前处理的结果，包括网格节点编号、坐标，材料属性)
- run nastran
  - input: `.bdf` file 使用python修改bdf中的参数，可以获得多个bdf，然后输入到nastran中进行仿真计算
  - output: `.f06` file 包含仿真输出的结果

## 修改厚度T

修改钢板厚度 Tickness思路：
- 读取所有节点坐标 nodes
- 计算厚度缩放比例 `ratio = T / T_origin`，T是需要修改的厚度，T是bdf文件中初始的厚度
- 将nodes坐标沿着厚度反向进行缩放，得到nodes_modify
- 将修改后的nodes坐标写入bdf文件中
- 然后运行nastran根据修改后的bdf文件仿真计算得到f06文件
- 读取f06文件中的结果并保存

main code

```python   
def get_output(bdf_path,input_path,nastran_path,save_dir):
  N_samples = 1000
  Tickness = np.random.uniform(2, 4, N_samples) # 2~4 mm
  
  for i in range(0,len(Tickness)):
      T = Tickness[i]
      bdf_copy_path = f'gangban_{i}.bdf'
      status_copy = shutil.copyfile(bdf_path, bdf_copy_path)
      if status_copy == bdf_copy_path:
          nodes = read_bdf_nodes(bdf_copy_path)
          T_origin = nodes[:, 1].max() - nodes[:, 1].min() # 3mm
          ratio_T = T / T_origin
          ratio = [1, ratio_T, 1]
          nodes_modify = nodes * ratio
          write_bdf_nodes(bdf_copy_path, nodes_modify)
  
          p = subprocess.Popen(nastran_path+' '+bdf_copy_path, shell=True)
          return_code = p.wait(timeout=1000)
          # time.sleep(15)
          time.sleep(7)
          print(f'Finished running Nastran for {i+1}th sample, T_origin: {T_origin}mm, T: {T}mm')
          
          modes = get_modes(bdf_copy_path.replace('.bdf', '.f06'))
          # all_modes.append(modes)
          # create a new txt file to store the modes
          save_txt = open(bdf_path.replace('.bdf', '.txt'), 'a')
          save_txt.write(str(modes) + '\n')
          save_txt.close()
      else:
          print(f'Error in copying file {bdf_copy_path}')
  
      bdf_copy_path_prefix = bdf_copy_path.split('.')[0]
      for suffix in ['.bdf', '.f04', '.f06', '.log', '.op2','.h5']:
          os.remove(bdf_copy_path_prefix + suffix)
  
  # all_modes = np.array(all_modes)
  read_save_txt = open(bdf_path.replace('.bdf', '.txt'), 'r')
  all_modes = read_save_txt.readlines()
  read_save_txt.close()
  
  all_modes = np.array([eval(mode) for mode in all_modes])
  if "updated" in input_path:
      np.savez(os.path.join(save_dir, 'modes_updated.npz'), modes = all_modes)
  else:
      np.savez(os.path.join(save_dir, 'modes.npz'), modes = all_modes)
  # np.savez(os.path.join(save_dir, 'modes_updated.npz'), modes = all_modes)
  print("Finished saving modes, shape is ", all_modes.shape)
```

```python
def read_bdf_nodes(bdf_filename:str) -> np.ndarray:
  """
  Read the node information from the bdf file
  """
  model = BDF()
  bdf = read_bdf(bdf_filename, xref=False,debug=False)
  # print(bdf.get_bdf_stats())
  # print('____________________________________________________________________________')
  # print(object_attributes(bdf))
  # print(object_methods(bdf))
  node_pos_all = []
  for nid,node in sorted(bdf.nodes.items()):
      # print(bdf.nodes[nid].xyz)
      # print(node)
      # print(node.get_position())
      # exit()
      node_pos = node.get_position()
      node_pos_all.append(node_pos)
  
  node_pos_all = np.array(node_pos_all)
  # print('Nodes shape:', node_pos_all.shape)
  # exit()
  
  return node_pos_all

def write_bdf_nodes(bdf_filename:str, nodes:np.ndarray):
  """
  Write the node information to the bdf file
  """
  model = BDF()
  bdf_xref = read_bdf(bdf_filename, xref=True,debug=False)
  # wrte the nodes to the bdf file
  for nid,node in sorted(bdf_xref.nodes.items()):
      node_pos = nodes[nid-1]
      # print(node.xyz)
      node.xyz = node_pos
      # print(node.xyz)
  bdf_xref.write_bdf(bdf_filename)

def get_modes(f06_copy_path:str):
  """
  Get the modes from the f06 file
  """
  find_txt = "NO.       ORDER                                                                       MASS              STIFFNESS"
  line_num = 0
  f = open(f06_copy_path, 'r')
  lines_mode = f.readlines()
  while True:
      line = lines_mode[line_num]
      if find_txt in line:
          break
      line_num += 1 # python 从0开始计数
  # print(line_num)
  modes = []
  for i in range(line_num+7, line_num+7+5):
      line_mode = lines_mode[i]
      modes.append(float(line_mode[67:80]))
  return modes
```


## 修改弹性模量E和剪切模量G

```python
def get_output(bdf_path,input_path,nastran_path,save_dir):
    N_samples = 10000
    # Generate random data
    E_modulus = np.random.uniform(190, 220, N_samples) * 1000
    G_modulus = np.random.uniform(77, 89, N_samples) * 1000

    for i in range(0,len(E_modulus)):
        E = E_modulus[i]
        G = G_modulus[i]
        bdf_copy_path = f'gangban_{i}.bdf'
        status_copy = shutil.copyfile(bdf_path, bdf_copy_path)
        if status_copy == bdf_copy_path:
            bdf_copy = open(bdf_copy_path, 'r+')
            lines = bdf_copy.readlines()
            lines[8682] = lines[8682].replace('210000.', f'{E:.3f}')
            lines[8682] = lines[8682].replace('83000.', f'{G:.3f}')
            bdf_copy.seek(0)
            bdf_copy.writelines(lines)
            # os.remove(bdf_copy_path)
            bdf_copy.close()
            # exit()
            p = subprocess.Popen(nastran_path+' '+bdf_copy_path, shell=True)
            return_code = p.wait(timeout=1000)
            # time.sleep(15)
            time.sleep(7)
            print(f'Finished running Nastran for {i+1}th sample')
            
            modes = get_modes(bdf_copy_path.replace('.bdf', '.f06'))
            # all_modes.append(modes)
            # create a new txt file to store the modes
            save_txt = open(bdf_path.replace('.bdf', '.txt'), 'a')
            save_txt.write(str(modes) + '\n')
            save_txt.close()
        else:
            print(f'Error in copying file {bdf_copy_path}')

        bdf_copy_path_prefix = bdf_copy_path.split('.')[0]
        for suffix in ['.bdf', '.f04', '.f06', '.log', '.op2','.h5']:
            os.remove(bdf_copy_path_prefix + suffix)

    read_save_txt = open(bdf_path.replace('.bdf', '.txt'), 'r')
    all_modes = read_save_txt.readlines()
    read_save_txt.close()

    all_modes = np.array([eval(mode) for mode in all_modes])
    if "updated" in input_path:
        np.savez(os.path.join(save_dir, 'modes_updated.npz'), modes = all_modes)
    else:
        np.savez(os.path.join(save_dir, 'modes.npz'), modes = all_modes)
    print("Finished saving modes, shape is ", all_modes.shape)
```