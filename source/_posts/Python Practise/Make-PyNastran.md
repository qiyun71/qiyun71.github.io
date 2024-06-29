---
title: Make-PyNastran
date: 2024-05-10 11:25:26
tags:
  - 
categories: Python Practise
---
 
> [Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)

<!-- more -->

# BDF

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
