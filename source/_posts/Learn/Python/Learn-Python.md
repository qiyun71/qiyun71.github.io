---
title: Python-Note
date: 2020-02-09 15:38:38
toc: true
summary: Python 语言学习笔记本
categories: Learn
tags:
  - Python
---
   
Python查缺补漏

<!-- more -->


# Python (with matlab)

>[安装用于 Python 的 MATLAB Engine API - MATLAB & Simulink - MathWorks 中国](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

- 要从 MATLAB 文件夹安装，请在 Windows® 上键入：
  - `cd "_matlabroot_\extern\engines\python cd D:\Software\Matlab\extern\engines\python"`
  - `python -m pip install .`
- 使用以下命令从 [https://pypi.org/project/matlabengine](https://pypi.org/project/matlabengine) 安装引擎 API：
  - `python -m pip install matlabengine`

### 启动 MATLAB Engine

启动 Python。在 Python 提示符下键入以下命令，以导入 MATLAB 模块并启动引擎：

```python
import matlab.engine
eng = matlab.engine.start_matlab()
```

matlab2023 最低python版本为 3.9
matlab2019 最高python版本为 3.7


# Project

| Project introduction                                                                                                                    | Link                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 许多 **Linux** 发行版都用 **systemd 来管理系统的服务**，比如开机启动、自动重启、守护进程等。该项目讲解了如何入门 systemd，并提供了一个 Python 脚本和 systemd unit 文件，可以在此基础上快速开发出 systemd 服务。 | [torfsen/python-systemd-tutorial: A tutorial for writing a systemd service in Python](https://github.com/torfsen/python-systemd-tutorial)                                      |
| 基于 OpenCV 的拼接模块开发的用于**快速拼接图片的 Python 库**                                                                                                | [OpenStitching/stitching: A Python package for fast and robust Image Stitching](https://github.com/OpenStitching/stitching)                                                    |
| 手写实现李航《统计学习方法》书中全部算法                                                                                                                    | [Dod-o/Statistical-Learning-Method_Code: 手写实现李航《统计学习方法》书中全部算法](https://github.com/Dod-o/Statistical-Learning-Method_Code)                                                      |
| 使用 Python 和 Matplotlib 进行科学可视化的开源书籍                                                                                                     | [rougier/scientific-visualization-book: An open access book on scientific visualization using python and matplotlib](https://github.com/rougier/scientific-visualization-book) |
| Python 写的人脸识别和面部属性分析框架，可根据人脸图像智能识别年龄、性别、情绪等信息                                                                                           | [serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python](https://github.com/serengil/deepface)     |


# 积累编写技巧

## 元组索引列表

```python
def topk_freq(self, x_freq):
  length = x_freq.shape[1]
  top_k = int(self.factor * math.log(length))
  # print(top_k)
  values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True) # (b, top_k, d)
  # print(values.shape, indices.shape); print(values); print(indices)
  mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij') # (b, d)
  index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1)) # (b, 1, d) (b, top_k, d) (b, 1, d)
  x_freq = x_freq[index_tuple] # (b, top_k, d)
return x_freq, index_tuple
```


- 变量的交换`a,b = b,a`
- 字符串格式化

`print("Hi, I'm %s . I'm from %s . And I'm %d" % (name,country,age))`
`print("Hi, I'm {} . I'm from {} . And I'm {}".format(name,country,age))`
`print(f"Hi, I'm {name} . I'm from {country} . And I'm {age+1}"`

[[双字] {Python}中5个好用的<字符串格式化>技巧_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Ux4y1179J/?vd_source=1dba7493016a36a32b27a14ed2891088)

```python
n: int = 1000000000
print(f'{n:_}') # 1_000_000_000
print(f'{n:,}') # 1,000,000,000

var: str = 'var'
print(f'{var:>20}') # 一共20个空格，右对齐
print(f'{var:20}') # 左对齐
print(f'{var:<20}') # 左对齐
print(f'{var:^20}') # 居中

print(f'{var:_>20}') # 空格变成下划线
print(f'{var:#>20}') # 空格变成#

from datetime import datetime
now: datetime = datetime.now()
print(f'{now:%d.%m.%y (%H:%M:%S)}') # 17.02.24 (11:20:56)
print(f'{now:%c}') # Sat Feb 17 11:20:56 2024
print(f'{now:%I%p}') # 11AM

n: float = 1234.5678
print(f'{n:.2f}') # 1234.57
print(f'{n:.0f}') # 1235
print(f'{n:,.3f}') # 123,4.568

a: int = 5
b: int = 10
print(f'{a+b=}') # a+b=15

my_var: str = 'my variable'
print(f'{my_var=}') # my_var='my variable'
print(f'my_var={my_var}') # my_var=my variable
```

## Yield

yield不需要整个列表生成完毕后再输出，可以一个一个输出。每当一个数据生成时，可以直接输出

```python
def fibonacci(n):
    a = 0
    b = 1
    for _ in range(n):
        yield a
        a, b = b, a+b
    return nums
for i in fibonacci(10):
    print(i)
```

## map和zip

`map(function,sequence)`
对序列sequence中每个元素都执行函数function操作，如`map(str,mylist)`：将列表中的每一项转换成字符串。
list()将每一项转换成列表

`zip(*list)`返回的是一个元组，转置
```python
list = [[1,2,3],[4,5,6],[7,8,9]]
t = zip(*list)
print t

# output
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]

x = [1,2,3,4,5]
y = [6,7,8,9,10]
a = zip(x,y)
print a

# output
[(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]
```

eg：将多个列表合并创建json数组

new_list = list(map(list, zip(address, temp)))
jsonify({
    'data': new_list
})

## partial

```python
from functools import partial

# 如果 clip_x_start 则创建一个maybe_clip函数，根据输入的x_ 输出 torch.clamp(x_, min=-1., max=1.)
maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
```

# Python基础

## try...except

```py
#python 异常处理
try:
<语句>        #运行别的代码
except <名字>：
<语句>        #如果在try部份引发了'name'异常
except <名字>，<数据>:
<语句>        #如果引发了'name'异常，获得附加的数据
else:
<语句>        #如果没有异常发生
```

单个 try-except 语句捕获所有发生的异常，不是一个很好的方式。可以使用多个except异常语句

- finally 退出try时总会执行
```py
try:
<语句>
finally:
<语句>    #退出try时总会执行
raise
```

```py
try:
    正常的操作
   ......................
except ExceptionType, Argument:
    你可以在这输出 Argument 的值...
```

### 用户自定义异常

- 通过创建一个新的异常类，程序可以命名它们自己的异常。异常应该是典型的继承自 Exception 类，通过直接或间接的方式
- 以下为与 RuntimeError 相关的实例,实例中创建了一个类，基类为 RuntimeError，用于在异常触发时输出更多的信息
- 在 try 语句块中，用户自定义的异常后执行 except 块语句，变量 e 是用于创建 Networkerror 类的实例

```py
class Networkerror(RuntimeError):
    def __init__(self, arg):
        self.args = arg
```

- **在你定义以上类后，你可以触发该异常，如下所示：**

```py
try:
    raise Networkerror("Bad hostname")
except Networkerror,e:
    print e.args
```

|           异常名称            |                        描述                        |
| :---------------------------: | :------------------------------------------------: |
|         BaseException         |                   所有异常的基类                   |
|          SystemExit           |                   解释器请求退出                   |
|       KeyboardInterrupt       |             用户中断执行(通常是输入^C)             |
|           Exception           |                   常规错误的基类                   |
|         StopIteration         |                 迭代器没有更多的值                 |
|         GeneratorExit         |        生成器(generator)发生异常来通知退出         |
|         StandardError         |              所有的内建标准异常的基类              |
|        ArithmeticError        |               所有数值计算错误的基类               |
|      FloatingPointError       |                    浮点计算错误                    |
|         OverflowError         |                数值运算超出最大限制                |
|       ZeroDivisionError       |            除(或取模)零 (所有数据类型)             |
|        AssertionError         |                    断言语句失败                    |
|        AttributeError         |                  对象没有这个属性                  |
|           EOFError            |             没有内建输入,到达 EOF 标记             |
|       EnvironmentError        |                 操作系统错误的基类                 |
|            IOError            |                 输入/输出操作失败                  |
|            OSError            |                    操作系统错误                    |
|         WindowsError          |                    系统调用失败                    |
|          ImportError          |                 导入模块/对象失败                  |
|          LookupError          |                 无效数据查询的基类                 |
|          IndexError           |              序列中没有此索引(index)               |
|           KeyError            |                  映射中没有这个键                  |
|          MemoryError          |     内存溢出错误(对于 Python 解释器不是致命的)     |
|           NameError           |            未声明/初始化对象 (没有属性)            |
|       UnboundLocalError       |               访问未初始化的本地变量               |
|        ReferenceError         | 弱引用(Weak reference)试图访问已经垃圾回收了的对象 |
|         RuntimeError          |                  一般的运行时错误                  |
|      NotImplementedError      |                   尚未实现的方法                   |
|      SyntaxError Python       |                      语法错误                      |
|       IndentationError        |                      缩进错误                      |
|         TabError Tab          |                     和空格混用                     |
|          SystemError          |                一般的解释器系统错误                |
|           TypeError           |                  对类型无效的操作                  |
|          ValueError           |                   传入无效的参数                   |
|     UnicodeError Unicode      |                     相关的错误                     |
|  UnicodeDecodeError Unicode   |                    解码时的错误                    |
|  UnicodeEncodeError Unicode   |                     编码时错误                     |
| UnicodeTranslateError Unicode |                     转换时错误                     |
|            Warning            |                     警告的基类                     |
|      DeprecationWarning       |               关于被弃用的特征的警告               |
|         FutureWarning         |           关于构造将来语义会有改变的警告           |
|        OverflowWarning        |        旧的关于自动提升为长整型(long)的警告        |
|   PendingDeprecationWarning   |              关于特性将会被废弃的警告              |
|        RuntimeWarning         |      可疑的运行时行为(runtime behavior)的警告      |
|         SyntaxWarning         |                  可疑的语法的警告                  |
|          UserWarning          |                 用户代码生成的警告                 |

> 参考[教程](https://www.runoob.com/python/python-exceptions.html)

## 加减乘除
### python 保留小数位

保留两位小数 `a=2.32424`
`round(a,2)` 
`"%.2f" %a`
`f'{a:.2f}'`

### 除法和取模

Python中
// ：地板除，即向负无穷取整
- 8//3 结果为2
- -8//3 结果为-3

Python中：% 取模（modulus）**与取余数有很大区别**
- -10%3 结果为2
- -90%8结果为6

C++中：% 取余
- -10%3 结果为-1
- -90%8结果为-2

```
# 取模，Python中可直接用%，计算模，r = a % b
def mod(a, b):    
    c = a // b
    r = a - c * b
    return r
 
# 取余 
def rem(a, b):
    c = int(a / b)
    r = a - c * b
    return r
```

## 字符串操作

- `str.split()` 对字符串进行切片--返回一个列表
  - 语法`str.split(str="", num=string.count(str)).`
    * str：分隔符，默认为所有的空字符，包括空格、换行、指标
    * num：分割次数，默认为-1，即分割所有
* `str.find()` **检测字符串中是否包含子字符串 str**
  - _如果包含子字符串返回开始的索引值，否则返回-1_
  * 语法`str.find(str, beg=0, end=len(string))`
    * str -- 指定检索的字符串
    * beg -- 开始索引，默认为 0
    * end -- 结束索引，默认为字符串的长度
- `string.join()`
  - 语法 `'sep'.join(seq)`以 sep 作为分隔符，将 seq 所有的元素合并成一个新的字符串 返回值：返回一个以分隔符 sep 连接各个元素后生成的字符串
    - sep：分隔符。可以为空
    - seq：要连接的元素序列、字符串、元组、字典
- ord() & chr()
  - ord()将字符转化为ascii码
  - chr()将ascii码转化为字母或实际数字

## 文件操作

### txt文件

```py
fd = open('file.txt','w',encoding='utf-8')      #utf-8 or GBK
fd.write(content)
fd.close()
```

其中 content 可以是字符串，变量，\t ......

<hr/>

|  r   |   w    |   a    |
| :--: | :----: | :----: |
| 只读 | 覆盖写 | 添加写 |

#### 读取txt行

```python
f.readlines()# 全部读取最后返回一个列表存所有的类,每行后面都会带有“\n”
f.readline()# 读取一列数据
```


### csv文件

#### 写入列表

- headers :表头
- rows :内容
- `f*csv = csv.writer(f)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\_f 为 open('file.txt','w',encoding='utf-8')*`
- f_csv.writerow(headers)
- f_csv.writerows(rows)

```py
import csv

headers = ['class','name','sex','height','year']

rows = [
        [1,'xiaoming','male',168,23],
        [1,'xiaohong','female',162,22],
        [2,'xiaozhang','female',163,21],
        [2,'xiaoli','male',158,21]
    ]

with open('test.csv','w',newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
```

_注意：如果打开 csv 文件出现空行的情况，那么需要添加一个参数 newline=”_
`with open('test.csv','w',newline='')as f:`

| class |   name    |  sex   | height | year |
| :---: | :-------: | :----: | :----: | :--: |
|   1   | xiaoming  |  male  |  168   |  23  |
|   1   | xiaohong  | female |  162   |  22  |
|   2   | xiaozhang | female |  163   |  21  |
|   2   |  xiaoli   |  male  |  158   |  21  |

#### 写入字典

- headers :表头
- rows :内容
- f_csv = DictWriter(f,headers)
- f_csv.writeheader()
- f_csv.writerows(rows)

```py
import csv

headers = ['class','name','sex','height','year']

rows = [
        {'class':1,'name':'xiaoming','sex':'male','height':168,'year':23},
        {'class':1,'name':'xiaohong','sex':'female','height':162,'year':22},
        {'class':2,'name':'xiaozhang','sex':'female','height':163,'year':21},
        {'class':2,'name':'xiaoli','sex':'male','height':158,'year':21},
    ]

with open('test2.csv','w',newline='')as f:
    f_csv = csv.DictWriter(f,headers)
    f_csv.writeheader()
    f_csv.writerows(rows)
```

| class |   name    |  sex   | height | year |
| :---: | :-------: | :----: | :----: | :--: |
|   1   | xiaoming  |  male  |  168   |  23  |
|   1   | xiaohong  | female |  162   |  22  |
|   2   | xiaozhang | female |  163   |  21  |
|   2   |  xiaoli   |  male  |  158   |  21  |

#### # csv读取

```py
import csv
with open('test.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        print(row)

'''result
['class', 'name', 'sex', 'height', 'year']
['1', 'xiaoming', 'male', '168', '23']
['1', 'xiaohong', 'female', '162', '22']
['2', 'xiaozhang', 'female', '163', '21']
['2', 'xiaoli', 'male', '158', '21']
'''

with open('test.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        print(row[0])

'''result
class
1
1
2
2
'''
```

> 参考[网站](https://blog.csdn.net/katyusha1/article/details/81606175 "CSDN")

- with open () as 读写文件

```py
# 读文件
with open('file.txt','r',) as f:
    print(f.read())
# 不需调用f.close()
# 如果文件过大则用read(size)比较保险
# 如果文件是配置文件readlines()较为方便


# 写文件
with open('file.txt','w',encoding='utf-8') as f:
    f.write('Hello World !')
# 文本文件    encoding 字符编码：gbk，utf-8
# 二进制文件  rb模式读取:图片,视频
```

> 参考[网站](https://blog.csdn.net/xrinosvip/article/details/82019844 "CSDN")

### os文件夹

```py
import os

main_path = 'E:/os/'    #创建一个路径
if not os.path.exists(main_path):   #如果该路径不存在
    os.makedirs(main_path)  #则新建一个路径

# 或者直接
os.makedirs(path, exist_ok=True)
```

- 删除文件: `os.remove(path)`
- 获取绝对路径: `os.path.abspath`
- 路径拼接：`os.path.join()`

```
print(os.path.abspath('.'))

运行PS E:\BaiduSyncdisk\NeRF_Proj\NeRO> & F:/miniconda/envs/nero/python.exe e:/BaiduSyncdisk/NeRF_Proj/NeRO/blender_backend/relight_backend.py

输出：E:\BaiduSyncdisk\NeRF_Proj\NeRO
```

# 概念

> [【Python】详解 collections.defaultdict-CSDN博客](https://blog.csdn.net/qq_39478403/article/details/105746952)

**collections** 作为 Python 的内建集合模块，实现了许多十分高效的特殊容器数据类型，即除了 Python 通用内置容器： dict、list、set 和 tuple 等的替代方案

# 装饰器

>[装饰器 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/1016959663602400/1017451662295584)

eg：

```
import time 
import functools

def metric(fn):
    # print('%s executed in %s ms' % (fn.__name__, 10.24))
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        start = time.time()
        res = fn(*args, **kw)
        end = time.time()
        print('{:s} executed in {:5f} ms, ' 'and the result is {:d}'.format(fn.__name__, (end - start), res))
        return fn(*args, **kw)
    return wrapper

# 测试
@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;

@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

f = fast(11, 22)
s = slow(11, 22, 33)
if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')

"""
print(fast.__name__)
如果有@functools.wraps(fn)，则返回fast
如果没有@functools.wraps(fn)，则返回wrapper
"""
print(fast.__name__)

output: 
fast executed in 0.013472 ms, and the result is 33
slow executed in 0.124733 ms, and the result is 7986
fast
```

```
def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

@log
def now():
    print('2015-3-25')

执行：now()
output:
call now():
2015-3-25
```

把`@log`放到`now()`函数的定义处，相当于执行了语句：

```
now = log(now)
```

由于log()是一个decorator，返回一个函数，所以，原来的now()函数仍然存在，只是现在同名的now变量指向了新的函数，于是调用now()将执行新函数，即在log()函数中返回的wrapper()函数。

## 装饰器需要参数

如果decorator本身需要传入参数，那就需要编写一个返回decorator的高阶函数，写出来会更复杂。比如，要自定义log的文本：

```
def log(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator
```
这个3层嵌套的decorator用法如下：

```
@log('execute')
def now():
    print('2015-3-25')
```

执行结果如下：

```
>>> now()
execute now():
2015-3-25
```

和两层嵌套的decorator相比，3层嵌套的效果是这样的：

```
>>> now = log('execute')(now)
```

## 原始函数属性复制到装饰后的函数中


以上两种decorator的定义都没有问题，但还差最后一步。因为我们讲了函数也是对象，它有`__name__`等属性，但你去看经过decorator装饰之后的函数，它们的`__name__`已经从原来的`'now'`变成了`'wrapper'`：

```
>>> now.__name__
'wrapper'
```

因为返回的那个`wrapper()`函数名字就是`'wrapper'`，所以，需要把原始函数的`__name__`等属性复制到`wrapper()`函数中，否则，有些依赖函数签名的代码执行就会出错。

不需要编写`wrapper.__name__ = func.__name__`这样的代码，Python内置的`functools.wraps`就是干这个事的，所以，一个完整的decorator的写法如下：

```
import functools

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper
```

或者针对带参数的decorator：

```
import functools

def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator
```

`import functools`是导入`functools`模块。模块的概念稍候讲解。现在，只需记住在定义`wrapper()`的前面加上`@functools.wraps(func)`即可。

## 类的定义和实例创建

```python
# 定义类，类名为Cname
class Cname(object):
    pass

# 创建cname1实例
cname1 = Cname()
```

## 类中的实例属性与类属性

* 实例属性：用于区分不同的实例，不同的类有不同的实例属性
* 类属性：是每个实例共有的属性，每个实例共有的属性

实例属性，每个实例有各自的属性

```python
# 实例属性（复杂）
cname1.name = y
cname2.name = q

# 实例属性（统一添加）
class Cname(object):
    def __init__(self,name):  # 初始化一个属性r
        self.name = name

cname1 = Cname(y)
print(cname1.name)
```

类的属性绑定后，所有实例都可以访问，而且**实例访问的类属性都相同**

```python
class Cname(object):
    zhongz = 'people'

    def __init__(self,name):
        self.name = name

cname1 = Cname('y')
cname2 = Cname('q')

print('--------')
print(cname1.zhongz)    # people
print(cname2.zhongz)    # people

# 通过类名修改了类属性后
Cname.zhongz = 'tenshi'
print('--------')
print(cname1.zhongz)    # tenshi
print(cname2.zhongz)    # tenshi

# 通过实例名修改了类属性
cname1.zhongz = 'mea'
print('--------')
print(cname1.zhongz)    # mea
print(cname2.zhongz)    # tenshi

# 删除了cname1的类属性zhongz后
print('--------')
print(cname1.zhongz)    # mea
del cname1.zhongz
print(cname1.zhongz)    # tenshi
```

**要修改类属性，不要再实例上修改，而是在类名上修改**

## 类的实例方法

```python
class Cname(object):
    zhongz = 'people'

    def __init__(self,name):
        self.name = name

    def printname(self):
        print(self.name) # 打印名字

cname1 = Cname('y')
cname2 = Cname('q')
cname1.printname()  # y
cname2.printname()  # q
```

`printname(self)`就是一个最简单的方法

## 类中的访问限制

### 属性的访问限制

python的类中的属性，如果有些属性不希望被外部访问，我们可以属性命名时以双下划线开头 `__`，如 `__age`

>但，如果一个属性以"\_\_xxx\_\_"的形式定义，那么它可以被外部访问。以"\_\_xxx\_\_"定义的属性在Python的类中被称为特殊属性，有很多预定义的特殊属性是以“\_\_xxx\_\_”定义，所以我们不要把普通属性用"\_\_xxx\_\_"定义。

>**加双下划线\_\_xx 的属性，可以通过“ \_类名\_\_xx ”可以访问到属性的值 如`Cname._Cname__age`**

###  方法的访问限制

双下划线，如`def __printage():`
- 此时，该方法只能在类的内部使用，而无法被外部调用

单下划线，不能通过 from module import * 这种方式导入，可以通过其他方式：
- import A
- `b = A._B()`

## 类中的装饰方法

* `@classmethod`    用来修饰类方法。使用在与类进行交互，但不和其实例进行交互的函数方法上
* `@staticmethod`   用来修饰静态方法。使用在有些与类相关函数，但不使用该类或该类的实例。如更改环境变量、修改其他类的属性等

*classmethod必须使用类的对象作为第一个参数，而staticmethod则可以不传递任何参数*

### @classmethod 修饰方法——类方法

类方法，我们不用通过实例化类就能访问的方法。而且@classmethod 装饰的方法不能使用实例属性，只能是类属性。它主要使用在和类进行交互，但不和其实例进行交互的函数方法上。

```python
class Cname(object):
    zhongz = 'people'

    def __init__(self,name):
        self.name = name

    def printname(self):
        print(self.name)
    
    @classmethod
    def printwe(cls):
        print(cls.zhongz)

# Cname.printname()   # 没有实例化 ，会发生错误
Cname.printwe()     # 没有实例化也可以访问
```

>printwe(cls)中cls表示的是类，它和self类实例有一定的差别。类方法中都是使用cls，实例方法中使用self

### @staticmethod 修饰方法——静态方法

`@staticmethod` 不强制要求传递参数（它做的事与类方法或实例方法一样）
`@staticmethod` 使用在有些和类相关函数，但不使用该类或者该类的实例。如更改环境变量、修改其他类的属性等
`@staticmethod` 修饰的方法是放在类外的函数，我们为了方便将他移动到了类里面，它对类的运行无影响

```python
class Date(object):
   day = 0
   month = 0
   year = 0

   def __init__(self, year=0, month=0, day=0):
       self.day = day
       self.month = month
       self.year = year

   @classmethod
   def from_string(cls, date_as_string):
       year, month, day = date_as_string.split('-')
       date = cls(year, month, day)
       return date
    # 返回的是类的实例

   @staticmethod
   def is_date_valid(date_as_string):
       """
      用来校验日期的格式是否正确
       """
       year, month, day = date_as_string.split('-')
       return int(year) <= 3999 and int(month) <= 12 and int(day) <= 31

date1 = Date.from_string('2012-05-10')
print(date1.year, date1.month, date1.day)
is_date = Date.is_date_valid('2012-09-18') # 格式正确 返回True
```

is_date_valid(date_as_string) 只有一个参数，它的运行不会影响类的属性

>@staticmethod修饰方法 is_date_valid(date_as_string)中无实例化参数self或者cls；而@classmethod修饰的方法中有from_string(cls, date_as_string) 类参数cls

##  python中的property的使用

property的作用
* 作为装饰器 @property将类方法转换为类属性（只读）
* property重新实现一个属性的setter和getter方法

### @property将类方法转换为只读属性

经常使用，将类的属性设置为不可修改

将一个类方法转变成一个类属性

```python
class Circle(object):
   __pi = 3.14

   def __init__(self, r):
       self.r = r

   @property
   def pi(self):
       return self.__pi

circle1 = Circle(2)
print(circle1.pi)
circle1.pi=3.14159  # 出现AttributeError异常
```

创建实例后我们可以使用circle1.pi 自己获取方法的返回值，而且他只能读不能修改

### property重新实现setter和getter方法

```python
class Circle(object):
   __pi = 3.14

   def __init__(self, r):
       self.r = r

   def get_pi(self):
       return self.__pi

   def set_pi(self, pi):
       Circle.__pi = pi

   pi = property(get_pi, set_pi)

circle1 = Circle(2)
circle1.pi = 3.14  # 设置 pi的值
print(circle1.pi)  # 访问 pi的值
```

当我们以这种方式使用属性函数时，它允许pi属性设置并获取值本身而不破坏原有代码

```python
class Circle(object):
   __pi = 3.14

   def __init__(self, r):
       self.r = r

   @property
   def pi(self):
       return self.__pi

   @pi.setter
   def pi(self, pi):
       Circle.__pi = pi

circle1 = Circle(2)
circle1.pi = 3.14  # 设置 pi的值
print(circle1.pi)  # 访问 pi的值
```

把一个getter方法变成属性，只需要加上@property就可以了，如上此时pi(self)方法，@property本身又创建了另一个装饰器@pi.setter，负责把一个setter方法变成属性赋值，于是，将@pi.setter加到pi(self, pi)上，我们就拥有一个可控的属性操作


>参考[知乎大佬](https://www.zhihu.com/people/lyzf)的[教程](https://zhuanlan.zhihu.com/p/30223570)
>感谢大佬让我搞懂了python的类，虽然最后的不太懂，但是基础是懂了


## 类的继承 

### 类的继承

```python
class Animal(object):  #  python3中所有类都可以继承于object基类
   def __init__(self, name, age):
       self.name = name
       self.age = age

   def call(self):
       print(self.name, '会叫')
# 现在我们需要定义一个Cat猫类继承于Animal，猫类比动物类多一个sex属性。
class Cat(Animal):
   def __init__(self,name,age,sex):
       super(Cat, self).__init__(name,age)  # 不要忘记从Animal类引入属性
       self.sex=sex

if __name__ == '__main__':  # 单模块被引用时下面代码不会受影响，用于调试
    c = Cat('喵喵', 2, '男')  #  Cat继承了父类Animal的属性
    c.call()  # 输出 喵喵 会叫 ，Cat继承了父类Animal的方法 
```

类的继承一般都是object，然后如果想要继承自己的类，则可以把object继承对象改一下，原来类名后括号里的东西是继承对象

一定要用 `super(Cat, self).__init__(name,age)` 去初始化父类，否则，继承自 Animal的 Cat子类将没有 `name` 和 `age` 两个属性

函数`super(Cat, self)`将返回当前类继承的父类，即 Animal，然后调用`__init__()`方法，注意self参数已在`super()`中传入，在`__init__()`中将隐式传递，不能再写出self

### Python对子类方法的重构

子类中的方法要求跟父类中的方法不同时，可以在子类中重构方法

```python
class Cat(Animal):
   def __init__(self, name, age, sex):
       super(Cat, self).__init__(name,age)
       self.sex = sex

   def call(self):
       print(self.name,'会“喵喵”叫')

if __name__ == '__main__':
   c = Cat('喵喵', 2, '男')
   c.call()  # 输出：喵喵 会“喵喵”叫
```

当我们在子类中重构父类的方法后，Cat子类的实例先会在自己的类Cat中查找该方法，当找不到该方法时才会去父类Animal中查找对应的方法

### Python中子类与父类的关系

```python
class Animal(object):
   pass

class Cat(Animal):
   pass

A= Animal()
C = Cat()
```

* “A”是Animal类的实例，但，“A”不是Cat类的实例。
* “C”是Animal类的实例，“C”也是Cat类的实例。

函数 `isinstance(变量,类型)`
判断变量的类型，判断对象之间的关系

```python
print('"A" IS Animal?', isinstance(A, Animal))
print('"A" IS Cat?', isinstance(A, Cat))
print('"C" IS Animal?', isinstance(C, Animal))
print('"C" IS Cat?', isinstance(C, Cat))

# 输出
"A" IS Animal? True
"A" IS Cat? False
"C" IS Animal? True
"C" IS Cat? True
```

### python中多态

类具有继承关系，并且子类类型可以向上转型看做父类类型，如果我们从 Animal派生出 Cat和Dog，并都写了一个 call() 方法

```python
class Animal(object):  
   def __init__(self, name, age):
       self.name = name
       self.age = age
   def call(self):
       print(self.name, '会叫')

class Cat(Animal):
   def __init__(self, name, age, sex):
       super(Cat, self).__init__(name, age)
       self.sex = sex

   def call(self):
       print(self.name, '会“喵喵”叫')

class Dog(Animal):
   def __init__(self, name, age, sex):
       super(Dog, self).__init__(name, age)
       self.sex = sex
   def call(self):
       print(self.name, '会“汪汪”叫')
```

我们定义一个do函数，接收一个变量 ‘all’,如下：

```python
def do(all):
   all.call()

A = Animal('小黑',4)
C = Cat('喵喵', 2, '男')
D = Dog('旺财', 5, '女')

for x in (A,C,D):
   do(x)

# 输出结果
# 小黑 会叫
# 喵喵 会“喵喵”叫
# 旺财 会“汪汪”叫
```

这种行为称为多态。也就是说，方法调用将作用在 all 的实际类型上。C 是 Cat 类型，它实际上拥有自己的 call() 方法以及从 Animal 继承的 call 方法
**而调用 C .call() 总是先查找它自身的定义，如果没有定义，则顺着继承链向上查找，直到在某个父类中找到为止**

>注意事项
> * 在继承中基类的构造方法（`__init__()方法`）不会被自动调用，它需要在其派生类的构造方法中亲自专门调用。
> * 在调用基类的方法时，需要加上基类的类名前缀，且需要带上self参数变量。而在类中调用普通函数时并不需要带上self参数
> * Python总是首先查找对应类的方法，如果它不能在派生类中找到对应的方法，它才开始到基类中逐个查找。（先在本类中查找调用的方法，找不到才去基类中找）

## 类的实例可调用

[【python】特殊函数 __call__ - 知乎](https://zhuanlan.zhihu.com/p/609832956)

可以模糊对象和函数的关系

```python
    def __init__(self,name):
        self.name=name

    def __call__(self):
        print("hello "+self.name)


def main():
    a = People('abc!')
    a.__call__()
    a()

if __name__ == '__main__':
    main()
```