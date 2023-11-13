---
title: Git学习
date: 2023-06-26T16:06:54.000Z
tags:
  - Git
categories: Learn
date updated: 2023-11-13T10:38:01.000Z
---

不要乱用 git reset --hard commit_id 回退 git commit 版本

<!-- more -->

# Clone

## 克隆子目录

1. 初始化：

Git init

2. 连接远端库：

Git remote add origin url

3. 启用"Sparse Checkout"功能：

Git config core.Sparsecheckout true

4. 添加想要 clone 的目录：

Echo “子目录路径” >> .git/info/sparse-checkout
注意：子目录路径不包含 clone 的一级文件夹名称：
例如库路径是：
<https://A/B/C/example.git>
我们想 clone example 下的 D/E/F 目录，则：
`echo “D/E/F” >> .git/info/sparse-checkout`

5. Pull 代码：

Git pull origin master
或者不包含历史版本的 clone：
Git pull --depth 1 origin master

> 版权声明：本文为 CSDN 博主「luo870604851」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
> 原文链接： <https://blog.csdn.net/luo870604851/article/details/119748749>

## 克隆私有仓库

> [(21条消息) Git clone 克隆私有项目_git clone 项目_风信子的猫Redamancy的博客-CSDN博客](https://blog.csdn.net/weixin_45508265/article/details/124340158)
> [How to Fix “Support for password authentication was removed” error in GitHub – Collabnix](https://collabnix.com/how-to-fix-support-for-password-authentication-was-removed-error-in-github/)

`git clone http://tokens-name:tokens@github.com/YOUR-USERNAME/YOUR-REPOSITORY`

# Git 教程

平时使用：

```
git add .
git commit -m shuoming
git push
```

> [工作区和暂存区 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/896043488029600/897271968352576)

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230626160832.png)

`git status` 查看暂存区 stage 状态

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230626161153.png)

# Git reset --hard commit_id 导致本地文件丢失

> [(19条消息) 恢复因git reset --hard 但未提交全部文件到仓库导致的文件丢失问题_git reset --hard 把未提交的文件搞丢了_数祖的博客-CSDN博客](https://blog.csdn.net/qq_56098414/article/details/121291539)
> [(19条消息) git add 后git reset --hard xxx的代码丢失，代码如何找回_小小花111111的博客-CSDN博客](https://blog.csdn.net/chailihua0826/article/details/94619904?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-94619904-blog-121291539.235^v38^pc_relevant_anti_vip_base&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

```
git fsck --lost-found

git show hash_id
git ls-tree 文件名

git read-tree --prefix=lib 726d5644919729dd97b7dd23f4a733e2daabab85

git restore lib

```

# 删除远程仓库文件，保留本地文件：

```
git rm --cached 文件 //本地中该文件不会被删除

git rm -r --cached 文件夹 //删除文件夹

git commit -m '删除某个文件'

git push
```

# 新建仓库

使用 token 链接仓库信息
`git remote set-url origin https://<your_token>@github.com/<USERNAME>/<REPO>.git`

# 用 token 克隆私有仓库

Git clone from token:

```
git clone https://NeRF-Mine:xxxtokenxxx@github.com/qiyun71/NeRF-Mine.git
tensorboard --port 6007 --logdir /root/tf-logs
```

Git remote remove origin
Git remote add origin path_repo

配置全局账户
- Git config --global user.Name ...
- Git config --global user.Email ...

Git push 需要配置 token(2023.11.13)
- 生成之后，如果之前执行了下面的命令，设置了存储登录凭据（默认存储在~/.Git-credentials）
  - Git config --global credential.Helper store
  - Git config --global --unset credential.Helper 清理之前保存的账密信息
- Git config --global credential.Helper store
