---
title: Learn-Cpp
date: 2024-11-26 15:47:38
tags:
  - 
categories: Learn/Other Interest
---

[C++ for Python Programmers — C++ for Python Programmers](https://runestone.academy/ns/books/published/cpp4python/index.html)

<!-- more -->

Plan:

| Timeline | Chapter               |
| -------- | --------------------- |
| 2024.11  | Data Types            |
|          | Control Structures    |
|          | Functions             |
|          | Collection Data Types |
|          | Input and Output      |
|          | Exception Handling    |
|          | Graphics              |

## Data Types

- Python中每个变量被存储为一个object，需要两块内存空间： reference and value。变量在声明后可以动态修改(动态语言)
- Cpp中每个变量的值被存在内存中，根据地址来进行访问，因此必须提前声明，并且之后变量的数据类型无法更改(静态语言)


指针定义：
- &varN 获取变量的地址
- `*ptrN` 为根据地址获取的值，与 `varN`含义相同

```cpp
int *ptrx;

int varN = 9;
int *ptrN = &varN; // ptrN points to varN address
```

空指针不指向任何内容

`ptrx = nullptr;`

