---
title: Hexo 简单流程
date: 2022-06-08 11:21:35
toc: true
tags: 
    - Hexo
categories: Learn
---

Hexo博客搭建教程，参考[CodeSheep](https://space.bilibili.com/384068749)大佬的[视频教程](https://www.bilibili.com/video/BV1Yb411a7ty)。主要是记录Hexo博客搭建和部署的步骤流程，方便更加快速的配置部署。
<!--more-->

>[手把手教你从0开始搭建自己的个人博客 |无坑版视频教程| hexo_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Yb411a7ty)

# 1.下载安装node.js
>[Node.js 中文网 (nodejs.cn)](http://nodejs.cn/)

*安装完成后，查看版本*
`npm -v`
`node -v`

# 2.换阿里的源
`npm install -g cnpm --registry=https://registry.npm.taobao.org`

*完成cnpm的安装，查看版本*
`cnpm -v`

# 3.安装hexo博客框架
`cnpm install -g hexo-cli`

{% note default %}
-g 全局安装
-v 查看版本
{% endnote %}


# 4.新建一个hexo博客
- 新建文件夹*Blog*(任意一个名字)
- 在该目录下打开终端输入`hexo init`完成博客初始化生成

# 5.在本地启动博客

`hexo server` or `hexo s`

## 问题
`hexo s`后无报错，但是打开本地端口后网页出现错误，显示此站点的连接不安全


![2022-06-09-09-07-42.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/2022-06-09-09-07-42.png "由于http自动变成https")

{% note warning %} 注意：是在http://localhost:4000/ 上进行，而不是https {% endnote %}

我的默认浏览器时Edge，但是Edge会自动将http转换成https，导致无法在本地查看Hexo博客，换成Chrome后可以解决。

# 6.新建博客文章
`hexo n "文章题目"`

写完一篇文章后
- 清理一下`hexo clean`
- 重新生成`hexo g`


# 7.部署到github上
## 7.1 在github上新建一个仓库
仓库名字：`用户名.github.io`

## 7.2 在Blog文件夹下安装deployer
`cnpm install --save hexo-deployer-git`

## 7.3 在Blog文件夹下打开_config.yml文件
```yml
# Deployment

## Docs: https://hexo.io/docs/one-command-deployment

deploy:

  type: 'git'

  repo: 'https://github.com/yq010105/yq010105.github.io.git'

  branch: 'master'
```

## 7.4 然后将Blog部署到github中

`hexo d`

# Last
使用Hexo博客方法
在写完文章后：
- `hexo clean`清除
- `hexo g`生成
- `hexo s`在本地预览，在http://localhost:4000/ 中查看
- `hexo d`推到github上，在[Hexo (yq010105.github.io)](https://yq010105.github.io/)中查看


# Other
## Next主题内置标签 Callout

`{% note class_name %} Content (md partial supported) {% endnote %}`
其中，class_name 可以是以下列表中的一个值：
- default
- primary
- success
- info
- warning
- danger

![20220609172943.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609172943.png "内置标签预览")


{% note default %}
default
{% endnote %}

{% note primary %}
primary
{% endnote %}

{% note success %}
success
{% endnote %}

{% note info %}
info
{% endnote %}

{% note warning %}
warning
{% endnote %}

{% note danger %}
danger
{% endnote %}