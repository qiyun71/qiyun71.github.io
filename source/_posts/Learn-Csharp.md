---
title: C#-Note
img:
top: false
toc: true
cover: false
date: 2020-04-08 09:23:49
categories: 学习力
tags:
  - C#
summary: 学习C#的笔记
password:
---

学习 C#的笔记

<!--more-->

> 参考[b 站的搬运视频学习](https://www.bilibili.com/video/BV1wx411K7rb)
> 原视频是 YouTube 上的[刘铁猛老师录制的教程](https://www.youtube.com/watch?v=EgIbwCnQ680&list=PLZX6sKChTg8GQxnABqxYGX2zLs4Hfa4Ca)
> Timothy Liu 老师新一期 C#教程的[C#语言入门详解》第二季](https://www.youtube.com/watch?v=HF3JGXV07Uo&list=PLZX6sKChTg8HP3MF9d8CEc3nrtEpiQ4Jf)

还发现刘老师还做了 Python 的教程，有机会去看看[pandas 玩转 excel](https://www.youtube.com/watch?v=i3TYCCY2WSE&list=PLZX6sKChTg8HyHfrmk97iblmsQLrlRHxn)

![20220609173231.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609173231.png)

# 0. C#学习前的 bb

C# 窗体应用，我的初步感受就是 VB+Java（狗头）
VB 现在正在学，但是 Java 是没学过 hhh

C# 是面向对象的语言，然而 C# 进一步提供了对面向组件 (component-oriented) 编程的支持。现代软件设计日益依赖于自包含和自描述功能包形式的软件组件。这种组件的关键在于，它们通过属性、方法和事件来提供编程模型；它们具有提供了关于组件的声明性信息的特性；同时，它们还编入了自己的文档。C# 提供的语言构造直接支持这些概念，这使得 C# 语言自然而然成为创建和使用软件组件之选。（copy from _csharp language specification 5.0 中文_）

# 1. 语言标准的——hello world!!!

## 1.1 Console

控制台.NET Framework

`Console.WriteLine("Hello World!!!");`

## 1.2 WPF

新的 windows forms（大概）,感觉就是更高级的 VB，更自由更美观的界面开发
跟 windows forms 一样

`textBoxShowHello.Text = "Hello World!";`

### WPF App(.NET)

跟 VB 很像，拖拽控件，编写控件程序，设计界面，所见即所得。

![2022-06-09-14-56-30.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/2022-06-09-14-56-30.png "Visual Studio的WPF App项目界面")

![2022-06-09-15-00-46.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/2022-06-09-15-00-46.png "HelloWorld-program")

## 1.3 Windows Forms App(old)

窗体程序，学过一点 VB，无所畏惧

button
`textBoxShowHello.Text = "Hello World!";`

## 1.4 ASP.NET Web Forms(old)

网络应用程序，网页
Controller 中
`<h1>Hello World!<h1>`

## 1.5 ASP.NET MVC

程序开发架构，可以将不同语言的代码放在不同的目录中
Controller 中
`<h1>Hello World!<h1>`

## 1.6 WCF

纯网络服务，读取数据库、向数据库输入数据

```csharp
public string SayHello()
{
  return "hello world!!!";
}
```

## 1.7 Windows Store Application

平板电脑,也是窗体设计
`textBoxShowHello.Text = "Hello World!!!";`

## 1.8 Windows Phone Application(已经凉透了？？？)

`textBoxShowHello.Text = "Hello World!!!";`

## 1.9 Cloud

云计算 Azure
`<h1>Hello World!<h1>`

## 1.10 WF

窗体设计
直接在 writeline 控件里写
`"hello world!!!"`

---

# 2. 类与名称空间

**class & namespace**

## 2.1 剖析 Hello World 程序

- 类是构成程序的主体
- 名称空间是以树型结构组织类（和其他类型），如 Button 类和 Path 类

Console App : `Console.WriteLine("Hello World~");`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApphello
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World");
        }
    }
}
```

类 class `Program`(自己写的)和`Console`(调用 C#的类)
使用 Console 类中的`WriteLine`方法来进行 print

名称空间 namespace `HelloWorld`，默认跟创建 project 时名称一样

**核心理解：**

**又如`System`名称空间中的`Console`类，类中的`WriteLine`方法**

**`using System`** 跟 python 中的`import`差不多，将类引用到自己的程序中

就是：
有`using System;` 直接用`Console.WriteLine`
没有`using System;`则必须用`System.Console.WriteLine`

## 2.2 类库

类库的引用可以分为两种

- 黑盒 DLL 引用（无源代码）
- 白盒 项目直接引用（有源代码）

### 黑盒：

#### a.本地引用

从其他人手中获得 dll 类库和 doc 文档，来进行引用

![20220609192636.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609192636.png)

浏览打开 dll 文件，即可引用类库

![20220609192848.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609192848.png)

#### b.网络引用

![20220609194354.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609194354.png)

黑盒引用无法修改 dll 中的错误

#### c.NuGet 添加类库

手动添加网络类库时，可能引用的某个类库还依赖着其他的底层类库，需要将该类库的所有依赖类库也一起引用进来，很麻烦。

使用 NuGet 可以将需要引用的类库与其依赖的底层类库一起打包引用进来，_有点像 Pycharm 中的安装轮子，pip？_
![20220609200539.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609200539.png)

### 白盒：

**一个项目可以属于多个 solution，一个 solution 中可以有多个项目**

添加已存在项目到 solution
![20220609201227.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609201227.png)

然后就可以直接引用项目中的类库了
![20220609201316.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609201316.png)

可以通过打断点，然后 start debugging，逐语句执行找错误，**找到真正的错误:Root cause!**
![20220609202044.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609202044.png)

### 建立自己的类库项目

![20220609202321.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609202321.png)

**注意：使用.Net Framework 创建**

![20220609203808.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609203808.png)

## 2.3 依赖关系

在自己写的类中引用了别人的类库，即为我的类依赖于别人的类库，如果该类库出现错误，则我的类也会有问题。

极其重要，尽量选择较弱的依赖关系，**高内聚，低耦合**

### UML 类图

通用建模语言，用图表达程序。

## 2.4 类

类(class)是对现实世界事务进行抽象所得到的的结果

### 2.4.1 类与对象

对象（现实世界）=实例（程序世界），是类经过实例化后得到的内存中的实体

可以将类看做概念，对象则是概念所指的实体

唯物主义

- 类：脑子中的飞机概念
- 对象：现实世界的飞机

### 2.4.2 创建类的实例

使用 new 操作符创建类的实例

![20220610094719](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220610094719.png)

### 2.4.3 引用变量和实例的关系

**就是 python 中的引用类`lol_watcher = LolWatcher('api-key')`，将一个类 new 出实例后复制个一个变量**

`Form myForm;`引用变量定义
`myForm = new Form();` new 出的实例赋值给引用变量

通过引用变量引用实例后，可以多次访问该实例

![20220610095236](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220610095236.png)

同一个实例可以被多个变量引用

```c#
Form myForm1;
Form myForm2;
myForm1 = new Form();
myForm2 = myForm2;
```

### 2.4.4 类的三大成员

- 属性
  - 存储数据，组合起来表示类或对象当前的状态
- 方法
  - C 语言中的函数
- 事件
  - 类或对象通知其它类或对象的机制，C#独有，要善用事件机制

#### 不同的类或对象侧重点不同

- 侧重于属性，Entity Framework
- 侧重于方法，Math，Console
- 侧重于事件，Timer

**Timer 的使用，新建一个 WPF APP 项目**

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace EventSample
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromSeconds(1);
            timer.Tick += Timer_Tick;
            timer.Start();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            this.timeTextBox.Text = DateTime.Now.ToString();
        }
    }
}
```

**效果：**
![20220610105152](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220610105152.png)

### 2.4.5 静态成员与实例成员

- 静态成员（Static）：类的成员，eg：人类的总数，人类的平均身高
  - 使用时，不需要 new 一个实例出来，全部的实例都可以使用类中的成员
- 实例成员（非静态）：对象的成员，eg：人的身高，体重
  - 使用时，需要先 new 一个实例出来，成员只能用于某一特定实例中

#### 绑定 Binding

- 早绑定语言（微软 C#），编译器编译时就将一个成员与类或对象关联起来
- 晚绑定语言（动态语言 javascript），在程序运行时，由程序决定某个成员属于某个类还是某个对象

#### 操作符.

成员访问操作符 **.**

---

# 3. 构成 C#语言的基本元素

## 3.1 标记 Token

**编译器能够识别的信息**

### 3.1.1 关键字 Keyword

[C#关键字](https://docs.microsoft.com/zh-cn/dotnet/csharp/language-reference/keywords/)

### 3.1.2 操作符 Operator

[C#操作符](https://docs.microsoft.com/zh-cn/dotnet/csharp/language-reference/operators/)

### 3.1.3 标识符 Identifier

自己起的名字，必须有一套规范
[C#标识符](https://docs.microsoft.com/zh-cn/dotnet/csharp/fundamentals/coding-style/identifier-names)

**命名规范：**

- 类名：名词
- 类的成员：
  - 属性：名词
  - 方法：动词、动词短语

**大小写规范：**

- 驼峰命名法 myVariable（变量名）
- MyVariable（方法名、类名...）

### 3.1.4 标点符号

不表示运算思想，如：

- 分号;表示语句结束
- 花括号{}

### 3.1.5 文本（字面值）

eg:`int x = 2;`

- 整数，_默认 int_
  - int，32bit `int x = 2;`
  - long 长整型，64bit ` long y = 3l;``long y = 3L; `
- 实数，_默认 double_
  - float，32bit `float x = 3.0F;`
  - double 双精度浮点数，64bit `double y = 4.0D;`
- 字符
  - char `char c = 'a';` _单引号只能引一个字符_
- 字符串
  - string `string str = "abcd";` _双引号可以引字符串_
- 布尔值
  - bool `bool b = true;`
- 空 null
  - `string str = null;`

## 3.2 注释与空白

- 行注释
- 块注释

注释快捷键 ctrl+k , ctrl+c
取消注释快捷键 ctrl+k, ctrl+u

```c#
// 行注释
/*块注释*/
```

- 空白
  - 空格，有的地方必须加空格
  - 回车

_也可以使用快捷键自动填补多余空白_ ctrl+k , ctrl+d
![20220610120551](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220610120551.png)

## 3.3 数据类型

var 声明变量，C#自动推断出变量的类型
`var x = 1;`
查看变量类型
`Console.WriteLine(x.GetType().Name);`
得到：`int32`

## 3.4 变量

变量声明`int x;`
变量赋值`x = 100;`

## 3.5 方法

方法就是函数，加工数据

使用`public 方法`类外部也能访问该方法
默认是`private`

方法的返回值类型写在方法名前

- int 表示方法返回一个 int 型变量
- void 表示方法不返回任何值

eg:

```C#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyExample
{
    class Program
    {
        static void Main(string[] args)
        {
            Calculator c = new Calculator();
            int x = c.Add(2, 3);
            Console.WriteLine(x);
        }
    }
    class Calculator
    {
        public int Add(int a, int b)
        {
            int result = a + b;
            return result;
        }
    }
}

```

## 3.6 程序(汉诺塔递归例子)

程序 = 数据+算法

### 汉诺塔递归法

参考[木子喵 neko 大佬的视频](https://www.bilibili.com/video/BV1SP4y137E9)

汉诺塔递归流程图

![hannuota](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/hannuota.png)

[自己找个在线游戏操作一下较好](https://zhangxiaoleiwk.gitee.io/h.html)
要从 A 将 3 个盘子移到 C：

- 先将 1、2 号盘子从 A 移动到 B
  - 先将 1 号盘子从 A 移动到 C `A-->C`
  - 再将 2 号盘子从 A 移动到 B `A-->B`
  - 最后将 1 号盘子从 C 移动到 B `C-->B`
- 然后将最大的 3 号盘子从 A 移动到 C `A-->C`
- 最后将 1、2 号盘子从 B 移动到 C
  - 先将 1 号盘子从 B 移动到 A `B-->A`
  - 再将 2 号盘子从 B 移动到 C `B-->C`
  - 最后将 1 号盘子从 A 移动到 C `A-->C`

```C#
using System;

namespace Digui
{
    class Program
    {
        static void Main(string[] args)
        {
            char a = 'A';
            char b = 'B';
            char c = 'C';
            int n = 3;
            HanNuoTao hnt = new HanNuoTao();
            hnt.Move(n, a, b, c);
        }
    }
    //汉诺塔
    class HanNuoTao
    {
        public void Move(int n, char a, char b, char c)
        {
            if (n == 0)
            {
                return;
            }
            else
            {
                Move(n - 1, a, c, b);
                Console.WriteLine($"{a}-->{c}");
                Move(n - 1, b, a, c);
            }
        }
    }

//output:
A-->C
A-->B
C-->B
A-->C
B-->A
B-->C
A-->C
```

---

# 4. 类型、变量和对象

## 4.1 数据类型作用

- Data Type：性质相同值的集合，配备一系列针对这种类型值的操作
  - 小内存容纳大尺寸数据会丢失精确度，发生错误
  - 大内存容纳小尺寸数据会导致浪费

{% note primary %}

- 强类型编程语言 C#
  - 数据受类型的约束很强
- 弱类型编程语言 C++/JavaScript
  - 数据受类型的 约束很弱 - c++中可能出现 if(x=200)可以编译成功的错误，一般在 c++写成 if(200==x)的形式，防止错误
    {% endnote %}

数据类型包含的信息

- 存储此类型变量所需的内存空间大小
- 此类型的值可表示的最大、最小值返回
- 此类型所包含的成员（如方法、属性、时间等等）
- 此类型由何种基类派生而来
- 程序运行时，此类型的变量分配在内存什么位置

  - 程序分配到内存后分为堆栈两个区域：
    - 堆：存的东西多，存储对象（实例放在堆里） - Heap
    - 栈：存的东西少，函数调用 - Stack
  - Stack overflow ：栈溢出，函数调用太多/程序有错误，栈上分配太多内存
  - 内存泄漏：分配对象忘了回收，造成内存浪费，_c#垃圾收集器中可以自动释放垃圾内存，防止内存泄漏_，_C++不会自动回收_

- 此类型所允许的运算操作
  - `double result = 3.0/4.0`结果为 0.75 _double 类型做浮点除法_
  - `double result = 3/4`结果为 0 _int 类型做整数除法_

{% note primary %}
静态程序：硬盘
动态程序：内存

编译:

- build：编译自己的代码生成 Assembly，并跟别人的装配件 Assembly 组合在一起
- compile：编译自己的代码，生成 Assembly 装配件
  {% endnote %}

## 4.2 C#五大数据类型

C#类型派生谱系
![20220611111310](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220611111310.png)

- 横线上
  object 和 string 都是类 class 的数据类型
- 横线下
  class、delegate、interface 都是关键字，而不是具体的数据类型，用这三个关键字去定义其他数据的类型

bool 的两个值：true、false
void：函数不返回值时，用 void 定义函数
null：一个变量的值是空的时候，给变量赋值 null
var、dynamic：声明变量，dynamic 仿照弱类型语言

五大数据类型：

- 类 Class
- 结构体 Structures
- 枚举 Enumerations
- 接口 Interfaces
- 委托 Delegates

### 4.2.1 类类型 class

Form 是类类型

- `Type myType = typeof(Form)` `cw(myType.FullName)` `cw(myType.IsClass)`
- 定义的地方有`class Form`

### 4.2.2 结构体类型 struct

int,double 等等
`struct Int32`
将 int32 吸收为关键字 int，int64 为 long

### 4.2.3 枚举 enum

限定用户从一个集合中选取有效值,用户只能在给定的几个值里选择

`enum FormWindowState`

```C#
Form f = new Form();
// 枚举类型窗口大小:只给定了Maximized,Normal,Minimized,选择其中一个
f.WindowState = FormWindowState.Maximized;
f.ShowDialog();
//窗口大小默认normal
```

## 4.3 变量、对象与内存

### 4.3.1 变量

- 表面上：变量是存储数据
- 实际上：变量表示了存储位置，并且每个变量有一个类型，来决定什么样的值能存入变量

**变量就是以变量名所对应的内存地址为起点，以其数据类型所要求的的存储空间为长度的一块内存区域**

- 变量一共有 7 种
  - 静态变量（静态成员变量）
    - `public static int Amount;`
  - 实例变量（非静态成员变量，字段：属性雏形，属性就是让字段只赋规定的值）
    - `public int Age;`
  - 数组元素
    - `int[] array = new int[100];`长度为 100 的整型数组
    - 访问数组第一个元素：`array[0]`‘
  - 值参数 `double c`
  - 引用参数 `ref double a`
  - 输出形参 `out double b`

```c#
class Student
{
  public double Add(ref double a ,out double b, double c)
  {
    return a+b+c;
  }
}
```

- 局部变量（狭义的变量，就是函数里声明的变量）

- 变量的声明
  <有效修饰符组合> 类型 变量名 <初始化器>
  `int a;`
  `public static int Amount;`有效修饰符组合：`public static`
  `int a = 1;`初始化器 `= 1`

### 4.3.2 值类型变量

结构体类型属于值类型
byte/sbyte/short/ushort

- 值类型，按这种类型的实际大小来分配内存
- 值类型的变量没有实例，它的实例与变量是合二为一的，就是在声明变量并赋值时，就分配好了一块内存给他
  变量地址的值就是给变量的赋值

### 4.3.3 引用类型变量

```C#
class Student
{
  uint ID;
  ushort Score;
}
```

- 引用类型
  - `Student stu;`
    - 先给引用类型分 4Byte 地址，里面 32 位全设为 0
  - `stu = new Student();`
    - 先去堆内存里创建 Student 实例，并将堆内存地址保存在 stu 的 4Byte 中
    - 在堆内存里创建实例：
      - `uint ID`给 4Byte
      - `ushort Score`给 2Byte
  - `Student stu2;`
    - 在内存中分配 4Byte 地址给 stu2
  - `stu2 = stu;`
    - 将 stu 中的数据赋值给 stu2

**stu 中装的是 Student 一个实例所在堆内存中的内存地址，也就是 30000001**

![20220611152529](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220611152529.png)

### 4.3.4 其他变量知识

- 局部变量在 stack 栈上分配内存
- 变量的默认值：没有给变量赋值，变量有个默认值：0
- 常量：值不可以改变的变量：`const int changliangpi = 3.14;`,常量的初始化器不能省略，常量不能被再次赋值
- 装箱与拆箱`int x = 100;`
  - 装箱`object obj = x;`obj 所引用的值是栈上值类型的值时，先把值类型的值 copy 到堆上，然后将堆内存地址放到 obj 变量对应的内存空间中
    - 将栈上的值类型值 封装成 obj 的实例，并刻在堆上
  - 拆箱`int y = (int)obj;`，现给 y 变量分配地址，将 obj 引用的在堆内存中的值移动到 y 变量的地址中
    - 将堆上 obj 的实例里的值，存到栈上
  - 装箱拆箱损失程序性能

{% note primary %}

- 内存
  - 栈:小数据 2M，值类型变量，引用类型变量
    - 值类型变量和实例合二为一
    - 引用类型变量中存放的是实例所在堆内存的地址
  - 堆:大数据，引用类型变量的实例 - 引用类型变量的实例（or 对象 or 真正的数据值）存放在堆内存中
    {% endnote %}

![heap_stack](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/heap_stack.png)

---

# 5. 方法

## 5.1 方法的由来

函数(C/C++) --> 方法(C#/Java 等面向对象的语言)

**C 中的函数写法：**

```C
#include <stdio.h>
// 函数：
double Add(double a, double b)
{
	return a + b;
}

int main()
{
	double x = 3.0;
	double y = 5.0;
	double result = Add(x, y);
	printf("%f+%f=%f", x, y, result);
	return 0;
}
```

**C++中的函数写法**

```C++
#include <iostream>

double Add(double a, double b)
{
	return a + b;
}

int main()
{
	double x = 3.0;
	double y = 5.0;
	double result = Add(x, y);
	std::cout << x << "+" << y << "=" << result;
	//std名称空间 ::为.
	//std::cout << "Hello World~";
	return 0;
}
```

当函数以类的成员出现的时候，就变成了方法

**C++中方法的写法：**

C++添加类的方法：
![20220612125102](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220612125102.png)
![20220612125033](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220612125033.png)

生成.h 和.cpp 两个文件

- 在.h 中声明方法：

```C++
#pragma once
class Student
{
public:
	void SayHello();
	double Add(double a, double b);
};
```

- 在.cpp 中对方法定义：

```C++
#include "Student.h"
#include <iostream>

void Student::SayHello()
{
	std::cout << "Hello I'm a student~";
}

double Student::Add(double a, double b)
{
	return a + b;
}
```

- 然后在其他 cpp 文件中调用：

```C++
#include <iostream>
#include "Student.h"
// 标准的类用<>，自己的类用""

int main()
{
	//std名称空间 ::为.
	//std::cout << "Hello World~";
	Student *pStu = new Student();
	pStu->SayHello();
	double x = 3.0;
	double y = 5.0;
	double result = pStu->Add(x, y);
	std::cout << x << "+" << y << "=" << result;
	return 0;
}
```

**C#中的方法(函数)写法**

{% note primary%}

- 函数
- 方法：函数以类的成员出现的时候叫方法
  {% endnote %}

**方法的作用**

- 隐藏复杂的逻辑
- 将大算法分解为小算法
- 复用，示例：

```C#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharpMethodExample
{
    class Program
    {
        static void Main(string[] args)
        {
            Calculator c = new Calculator();
            Console.WriteLine(c.GetCircleArea(10));
        }
    }
    class Calculator
    {
        // 圆面积
        public double GetCircleArea(double r)
        {
            return Math.PI * r * r;
        }
        // 圆柱体积
        public double GetCylinderVolume(double r, double h)
        {
            // 方法的复用
            return GetCircleArea(r) * h;
        }
        // 圆锥体积
        public double GetCooneVolume(double r, double h)
        {
            return GetCylinderVolume(r,h) / 3;
        }
    }
}
```

## 5.2 方法的定义与调用

### 5.2.1 方法的声明定义不分家

- 方法的声明：
  <attributes> <修饰符的有效组合> <partial> 返回类型 方法名(形式参数parameter) 方法体
  - 修饰符：new public private async static 等等
  - 返回类型：type void
  - 方法名：标识符（动词或动词短语）
  - 方法体：语句块：{语句} 或 分号：;

```C#
public double GetCircleArea(double r)
{
  return Mati.PI * r * r;
}
```

- 静态方法 static 与类绑定
- 实例方法 与实例绑定

例如一个类Calculator，new一个实例`Calculator c = new Calculator();`，如果是静态方法，用`Calculator.方法`来调用，如果是实例方法，则需要使用`c.方法`来调用

调用方法需要传入必要的实参：argument
要写成`c.GetCircleArea(x,y)`，而不能写成`c.GetCircleArea(double x,double y)`

{% note primary %}
形参parameter是变量，实参argument是值
{% endnote %}

## 5.3 构造器-特殊的方法

constructor构造器，是类型的成员之一

instance constructor实例构造器，构造实例在内存中的内部结构

**自定义构造器**
`ctor + tab*2`

- 没有参数构造器
![20220629193542](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220629193542.png)

- 有参数构造器
![20220629194137](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220629194137.png)

## 5.4 方法的重载 overload

同一个类下，有两个相同的方法名，但签名不同

声明带有重载的方法：
- 方法签名method signature ，方法签名由方法的名称、类型形参<T>的个数和它的每一个形参的类型和*形参传递模式|种类*（值、引用ref或输出out）组成（按从左到右的顺序）。**不包含返回类型**
- 实例构造函数签名由它的每一个形参的类型和形参传递模式（值、引用或输出）组成（按从左到右的顺序）。
- 重载决策：根据传入的参数类型选择一个最佳的函数成员来实施调用

## 5.5 如何对方法进行debug

- 设置断点
- 观察差方法调用时的call stack调用堆栈：该程序语句的父级（谁调用的当前语句）
- Step-into逐语句，Step-over逐过程（不用进入另一个方法），Step-out跳出（返回上层）
- 观察局部变量的值和变化

## 5.6 方法的调用与栈\*

对satck frame（方法被调用时在内存中的布局）的分析，方法调用如何使用栈内存

main（主调者caller）中调用other（被调者callee）函数，需要传入实参时，实参归main管。

# 6. 操作符

## 6.1 操作符概览

![20220703193132](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220703193132.png)

## 6.1 操作符的本质

## 6.1 操作符的优先级

## 6.1 同级操作符的运算顺序

## 6.1 各类操作符示例


# 7. 表达式、语句