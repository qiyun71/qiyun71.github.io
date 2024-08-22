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


新建分支
git checkout -b NEW_BRANCH_NAME

## clone repo with token
git clone https://NeRF-Mine:#####token#####@github.com/qiyun71/NeRF-Mine.git
