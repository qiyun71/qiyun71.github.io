---
title: Nastran by python
date: 2024-05-10 11:25:26
tags: 
categories: Project
---
 
> [Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)

使用PyNastran库对 Patran&Nastran有限元分析软件进行二次开发，可以自动对目标模型进行有限元分析

<!-- more -->

# BDF file

BDF文件是使用Patran对模型进行前处理产生的，包括划网格、定义结构参数、添加约束等操作

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


## 修改几何尺寸

缩放修改BDF文件中nodes的坐标，就可以实现修改结构的厚度/长度等几何尺寸参数

问题1：只有单个零件的结构比较容易修改，但是如果结构中有许多子结构，在BDF文件中很难区分哪些是子结构1，哪些是子结构2的
尝试1：可以给每个子结构设置不同的坐标系coords，(Patran)

> [nodes Module — pyNastran 1.4 1.4 documentation](https://pynastran-git.readthedocs.io/en/1.4/reference/bdf/cards/pyNastran.bdf.cards.nodes.html#pyNastran.bdf.cards.nodes.GRID)

```python
eid100 = bdf_xref.elements[100]
print(eid100)
print("nodes = %s" % eid100.nodes)
print("--node0--\n%s" % eid100.nodes_ref[0])
print("--cd--\n%s" % eid100.nodes_ref[0].cd)
print("cd.cid = %s" % eid100.nodes_ref[0].cd_ref.cid)
```

|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|
|GRID|NID|CP|X1|X2|X3|CD|PS|SEID|

# Example (Steel Plate Structure)

Generate simulation datasets by FE (solidworks & patran & nastran & python)

Model and Simulate to get `.bdf` file
- solidworks project Part.SLDPRT, export `.x_t` file
- patran 材料属性参数设置+网格划分+约束+负载，并定义好输出的类型(mode shape/FRF...) to get `.bdf` file(其中保存着有限元模型前处理的结果，包括网格节点编号、坐标，材料属性)
- run nastran
  - input: `.bdf` file 使用python修改bdf中的参数，可以获得多个bdf，然后输入到nastran中进行仿真计算
  - output: `.f06` file 包含仿真输出的结果


## python

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