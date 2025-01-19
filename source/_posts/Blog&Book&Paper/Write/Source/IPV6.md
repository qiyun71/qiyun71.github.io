---
title: 校园网IPV4付费，利用IPV6苟活记录
date: 2023-11-16 21:16:09
tags:
  - Tools
categories: Other/Mine
---

校园网 IPV4 流量限量 120G，超额**0.6 元/G**
**开学季|校园网使用指南**URL: https://mp.weixin.qq.com/s/NsojrsEUVeGOfSEEkpY3kw

IPV6 免费使用

<!-- more -->

# PT 站

校园网 IPV6 免费流量

- 北洋园[北洋园PT :: 首页 - Powered by NexusPHP (tjupt.org)](https://tjupt.org/index.php)
- 北邮人 [BYRBT :: 首页 - Powered by NexusPHP](https://byr.pt/index.php)

# 镜像网站

[清华大学开源软件镜像站 | Tsinghua Open Source Mirror(https://mirrors6.tuna.tsinghua.edu.cn/)](https://mirrors6.tuna.tsinghua.edu.cn/) 只解析 IPv6

## Conda

Conda 配置 ipv6 的清华源

> 为 Conda 添加清华软件源 - 知乎 URL: https://zhuanlan.zhihu.com/p/47663391

```
# Anaconda官方库的镜像
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

# Anaconda第三方库 Conda Forge的镜像
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# Pytorch的Anaconda第三方镜像
## win10 win-64
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
## linux
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --set show_channel_urls yes
# 添加完成后可以使用conda info命令查看是否添加成功.
```

- Win10 的 conda 配置文件在 `C:\Users\用户名` 下.Condarc
- Ubuntu 在 `\home\用户名` 下 .condarc

```
channels:
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
```

## Pip

Pip 配置 pip 的包在/pypi/web/simple/目录下：

`pip config set global.index-url https://mirrors6.tuna.tsinghua.edu.cn/pypi/web/simple/`

- win10会在 `C:\Users\用户名\AppData\Roaming\pip` 下生成 pip.Ini 文件
- Ubuntu 在 `/home/用户名/.config/pip/` 下 pip.Conf

```
[global]
index-url = https://mirrors6.tuna.tsinghua.edu.cn/pypi/web/simple/
```


## WSL2配置代理

[WSL2 中访问宿主机 Windows 的代理 - ZingLix Blog](https://zinglix.xyz/2020/04/18/wsl2-proxy/)

V2rayN中需要开启允许局域网的连接

修改`.bashrc`

```bash
# Windows 宿主机 IP
WINDOWS_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
# Windows 宿主机代理端口
WINDOWS_PROXY_PORT=10809

# 更新 Windows 网络信息
function update_windows_net_info() {
    WINDOWS_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
    WINDOWS_PROXY_PORT=10809
}

WINDOWS_IP=xxxx # 可以在win的cmd中使用ipconfig查看IP地址，复制到这里

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

proxy_on
```

`source ~/.bashrc`


# 服务器v2ray

## Vultr自建 💴30/月

> [实现校园网IPv6免流量上网与科学上网 | V2ray教程：X-ui与v2rayN ~ 极星网](https://www.jixing.one/vps/v2ray-xui-v2rayn/)

yum update -y
yum install -y curl
yum install -y socat

### x-ui
bash <(curl -Ls https://raw.githubusercontent.com/vaxilu/x-ui/master/install.sh)
设置面板的账户+密码+端口号54321   

### 开放端口+BBR加速

```
yum install firewalld
sudo systemctl start firewalld

firewall-cmd --permanent --add-port=54321/tcp --add-port=12345/tcp
#开放端口（54321是面板端口，12345是后面节点要用的）

firewall-cmd --permanent --list-ports 
#查看防火墙的开放的端口

firewall-cmd --reload

#重启防火墙(修改配置后要重启防火墙)

# 开启BBR
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p

## 检查是否开启
sysctl -n net.ipv4.tcp_congestion_control
lsmod | grep bbr

reboot
```

浏览器打开
ipv4:port

#### 端口被封了

新开一个
```
firewall-cmd --permanent --add-port=1771/tcp
firewall-cmd --reload
```

### ssh连接

```
ssh -p port root@ip_address
```

### 日志滚动

在x-ui面板设置中添加：
```json
"log": {
      "access": "/var/log/xray/access.log",       
      "error": "/var/log/xray/error.log",       
      "loglevel": "debug"     }
```

然后再bash中输入： `tail -f /var/log/xray/access.log` 

## Error

[Centos8使用yum报错 Couldn‘t resolve host name for http://mirrorlist.centos.org/?releas_couldn't resolve host name for-CSDN博客](https://blog.csdn.net/qq_41688840/article/details/123299876)


# 客户端v2ray

## Windows

> [2dust/v2rayN: A GUI client for Windows, support Xray core and v2fly core and others](https://github.com/2dust/v2rayN)

### 全局模式

一些无法获得服务器的客户端(如二游的启动器下载or更新)，由于v2rayN设置将其从黑名单中移除，而导致其不会通过代理，可以使用TUN模式(虚拟网卡方式，真正的全局代理)

### TUN模式

win10查看路由表：`route print`

修改IP CIDR后需要重启V2rayN

```
tjupt.org;
cc.sjtu.edu.cn
*xgsdk.com;
romkit.ustb.edu.cn;
115.25.60.121;
access.clarivate.com;
byr.pt;
ncsti.gov.cn;
duxiu.com;
pan.baidu.com;
superlib.net;
202.204.48.66;
cipp.ustb.edu.cn;
*autodl.com;
dict.eudic.net;
mirrors6.tuna.tsinghua.edu.cn;
h.ustb.edu.cn;
oakchina.cn;
mems.me;
eefocus.com;
readpaper.com;
cnki.ustb.edu.cn;
discovery.ustb.edu.cn;
elib.ustb.edu.cn;
nlc.cn;
*seetacloud.com;
aipaperpass.com;
simpletex.cn;
bchyai.com;
cubox.pro;
login.weixin.qq.com;
wx2.qq.com;
webpush.wx2.qq.com;
*qq.com;
api.link-ai.chat;
wiki.biligame.com;
www.sciencedirect.com;
pdf.sciencedirectassets.com;
blog.csdn.net;
```


## Ubuntu

> [Linux 后备安装方式 - v2rayA](https://v2raya.org/docs/prologue/installation/linux/#%E6%96%B9%E6%B3%95%E4%B8%89%E6%89%8B%E5%8A%A8%E5%AE%89%E8%A3%85)

下载[v2fly/v2ray-core](https://github.com/v2fly/v2ray-core) 和 [v2rayA/v2rayA](https://github.com/v2rayA/v2rayA/releases/tag/v2.2.5.1) (v2raya_linux_x64_2.2.5.1)


下载的时候需要注意你的 CPU 架构，下载好之后解开压缩包，然后把可执行文件复制到 `/usr/local/bin/` 或 `/usr/bin/`（推荐前者），把几个 dat 格式的文件复制到 `/usr/local/share/v2ray/` 或者 `/usr/share/v2ray/`（推荐前者，xray 用户记得把文件放到 xray 文件夹），最后授予 v2ray/xray 可执行权限

```
unzip v2ray-linux-64.zip -d ./v2ray
mkdir -p /usr/local/share/v2ray && cp ./v2ray/*dat /usr/local/share/v2ray
install -Dm755 ./v2ray/v2ray /usr/local/bin/v2ray
rm -rf ./v2ray v2ray-linux-64.zip
```

v2rayA 只有一个单独的二进制，下载下来放到 `/usr/local/bin/` 或 `/usr/bin/`（推荐前者）即可。和下载 v2ray 一样，下载的时候需要注意你的 CPU 架构。

`install -Dm755 ./v2raya_linux_x64_$version /usr/local/bin/v2raya`

一般情况下，在终端里面直接运行 `v2raya` 命令即可，配置文件夹默认会是 `/etc/v2raya/`。不过，为了方便，在 Linux 系统上一般采用服务的形式运行 v2rayA.



