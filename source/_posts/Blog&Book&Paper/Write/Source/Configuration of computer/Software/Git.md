# git

github新建仓库，将本地代码上传到github

## upload local dir to remote

本地文件夹下：

git init
git remote add origin https:///OWNER/REPOSITORY.git

如果本地默认是master的话
git branch -m master main

git add .
git commit -m xxx
git push -u origin xxxx -f 

`git push --set-upstream origin main`

新建分支
git checkout -b NEW_BRANCH_NAME

## clone repo with token
git clone https://NeRF-Mine:#####token#####@github.com/qiyun71/NeRF-Mine.git

## 忽略本地修改，强制拉取远程到本地

[git pull时冲突的几种解决方式 - 雪山上的蒲公英 - 博客园](https://www.cnblogs.com/zjfjava/p/10280247.html)

```bash
git fetch --all
git reset --hard origin/dev
git pull
```