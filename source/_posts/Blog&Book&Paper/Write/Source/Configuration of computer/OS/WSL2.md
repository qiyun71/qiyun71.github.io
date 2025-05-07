# WSL2

## 安装

>[windows11 安装WSL2全流程_win11安装wsl2-CSDN博客](https://blog.csdn.net/u011119817/article/details/130745551)

首先打开 windows features 中的 子系统和虚拟化：

```powershell-admin
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
```

之前卸载过一次后安装发现一直报错[Installation Failed with Error 0x8007019e · Issue #2982 · microsoft/WSL](https://github.com/microsoft/WSL/issues/2982)  https://github.com/microsoft/WSL/issues/2982#issuecomment-2218998948 这位大佬解决了我的问题

重新安装 wsl [Releases · microsoft/WSL](https://github.com/microsoft/WSL/releases)

```bash
# 升级到最新版
wsl --update
# 设置wsl2
wsl --set-default-version 2
```

### Ubuntu
方法一：选好版本直接再Microsoft Store中下载，然后打开：

```bash
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username:
```

输入用户名和密码后可以直接进入系统

方法二：

```bash
# 查看分发系统并安装
wsl -l -o  # wsl --list --online

# 安装我们需要的系统：
wsl --install -d Ubuntu-20.04
```


### Archlinux

>  [Install Arch Linux on WSL - ArchWiki](https://wiki.archlinux.org/title/Install_Arch_Linux_on_WSL)

```cmd
wsl --import _Distro_name_ _Install_location_ _WSL_image_

useradd -G wheel -m foobar
passwd foobar

--> /etc/wsl.conf
add:
[user]
default= foobar

root 没有密码：
wsl -d Arch -u root
passwd

```


## 常规操作

移动到其他盘[WSL2安装Ubuntu20.04 - 王谷雨 - 博客园 (cnblogs.com)](https://www.cnblogs.com/konghuanxi/p/14731846.html)

```powershell
# 查看当前wsl安装的虚拟机
wsl -l -v

# 启动
wsl -d name

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

```bash
# 设置默认系统：
wsl --set-default Ubuntu1 # 或wsl -s Ubuntu1
```



## 硬件配置

网络配置

V2rayN中设置局域网代理

windows `ipconfig` 查看ip地址

wsl中配置：
```bash
# ~/.bashrc
# Windows 宿主机 IP
WINDOWS_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
# Windows 宿主机代理端口
WINDOWS_PROXY_PORT=10809

# 更新 Windows 网络信息
function update_windows_net_info() {
    WINDOWS_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
    WINDOWS_PROXY_PORT=10809
}

# 开启代理
function proxy_on() {
    export HTTP_PROXY="http://${WINDOWS_IP}:${WINDOWS_PROXY_PORT}" # http 或 socks5，取决于代理的协议
    export HTTPS_PROXY="http://${WINDOWS_IP}:${WINDOWS_PROXY_PORT}" # http 或 socks5，取决于代理的协议
    export ALL_PROXY="http://${WINDOWS_IP}:${WINDOWS_PROXY_PORT}" # http 或 socks5，取决于代理的协议
    echo -e "Acquire::http::Proxy \"http://${WINDOWS_IP}:${WINDOWS_PROXY_PORT}\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf > /dev/null
    echo -e "Acquire::https::Proxy \"http://${WINDOWS_IP}:${WINDOWS_PROXY_PORT}\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf > /dev/null
    proxy_status
}

# 关闭代理
function proxy_off() {
    unset HTTP_PROXY
    unset HTTPS_PROXY
    unset ALL_PROXY
    sudo sed -i -e '/Acquire::http::Proxy/d' /etc/apt/apt.conf.d/proxy.conf
    sudo sed -i -e '/Acquire::https::Proxy/d' /etc/apt/apt.conf.d/proxy.conf
    proxy_status
}

# 代理状态
function proxy_status() {
    echo "HTTP_PROXY:" "${HTTP_PROXY}"
    echo "HTTPS_PROXY:" "${HTTPS_PROXY}"
    echo "ALL_PROXY:" "${ALL_PROXY}"
}
```

```bash
# ~/.condarc
ssl_verify: false

proxy_servers:
  http: http://ip_address:10809
  https: http://ip_address:10809

# sudo apt

# /etc/wgetrc
https_proxy = http://ip_address:10809
http_proxy = http://ip_address:10809
ftp_proxy = http://ip_address:10809
use_proxy = on
```


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

## GUI

WSL GUI [microsoft/wslg: Enabling the Windows Subsystem for Linux to include support for Wayland and X server related scenarios (github.com)](https://github.com/microsoft/wslg)
