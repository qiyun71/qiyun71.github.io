# WSL2

移动到其他盘[WSL2安装Ubuntu20.04 - 王谷雨 - 博客园 (cnblogs.com)](https://www.cnblogs.com/konghuanxi/p/14731846.html)

```powershell
# 查看当前wsl安装的虚拟机
wsl -l -v

# 关闭所有正在运行的虚拟机
wsl --shutdown

# 对需要迁移的分发或虚拟机导出
wsl --export 虚拟机名称Ubuntu 文件导出路径D:\Ubuntu.tar

# 卸载虚拟机（删除C盘的虚拟机数据）
wsl --unregister 虚拟机名称Ubuntu

# 导入新的虚拟机
wsl --import 虚拟机名称Ubuntu 目标路径D:\WSL\Ubuntu 虚拟机文件路径D:\Ubuntu.tar --version 2
# 目标路径就是想要将虚拟机迁移到的具体位置
```

[WSL文件存储位置迁移 - 掘金 (juejin.cn)](https://juejin.cn/post/7284962800668475455)
迁移后还需要切换默认用户`虚拟机名称Ubuntu config --default-user <username>`，Ubuntu为D盘中

## 硬件配置

修改内存和swap大小`C:\Users\Qiyun`下新建`/.wslconfig`文件
[Advanced settings configuration in WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#configuration-setting-for-wslconfig)

```
[wsl2]
memory=16GB
swap=8GB
```

查看linux内存`free -h`

## 系统指令

查看 Ubuntu 版本`lsb_release -a`