---
title: Git学习
date: 2023-06-26 16:06:54
tags:
    - Git
categories: Learn
---

不要乱用git reset --hard commit_id回退git commit版本

<!-- more -->


# git教程

>[工作区和暂存区 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/896043488029600/897271968352576)

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230626160832.png)

`git status` 查看暂存区stage状态

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230626161153.png)


# git reset --hard commit_id 导致本地文件丢失

>[(19条消息) 恢复因git reset --hard 但未提交全部文件到仓库导致的文件丢失问题_git reset --hard 把未提交的文件搞丢了_数祖的博客-CSDN博客](https://blog.csdn.net/qq_56098414/article/details/121291539)
>[(19条消息) git add 后git reset --hard xxx的代码丢失，代码如何找回_小小花111111的博客-CSDN博客](https://blog.csdn.net/chailihua0826/article/details/94619904?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-94619904-blog-121291539.235^v38^pc_relevant_anti_vip_base&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

```
git fsck --lost-found

git show hash_id
git ls-tree 文件名

git read-tree --prefix=lib 726d5644919729dd97b7dd23f4a733e2daabab85

git restore lib

```

