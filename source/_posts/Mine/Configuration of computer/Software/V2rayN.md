# V2rayN

V2rayN 用户自定义 PAC 设置 Version3.2.9

```v2ray
||github.com,
||google.com,
||chat.openai.com,
||huggingface.co,
||github.io,
||ip138.com,
||paperswithcode.com,
||kaggle.com,
||arxiv.org,
||motorica.ai,
||www.nerfacc.com,
||*.docs.*,
||alicevision.org,
||plotly.com,
||wallhaven.cc,
||ai.meta.com,
||dl.fbaipublicfiles.com,
||awesome-selfhosted.net,
||aka.ms,
||plotly.com,
||trimesh.org.||ten24.info,
||cn.noteai.com,
||ipinfo.io,
||web.stanford.edu,
||awesomeopensource.com,
||*.thecvf.com,
||fnzhan.com,
||skybox.blockadelabs.com,
||www.lumalabs.ai,
||www.ahhhhfs.com,
||www.dropbox.com,
||groups.csail.mit.edu,
||suezjiang.github.io,
||www.arxivdaily.com,
||zero123.cs.columbia.edu,
||happy.5ge.net,
||erikwernquist.com,
||brilliant.org,
||alexyu.net/plenoctrees,
||chenhsuanlin.bitbucket.io,
||discord.com,
||pyflo.net,
||cin.cx
```

V2rayN Version6.29

> [路由规则设定方法 · V2Ray 配置指南|V2Ray 白话文教程 (toutyrater.github.io)](https://toutyrater.github.io/routing/configurate_rules.html)
>软件：[2dust/v2rayN: A GUI client for Windows, support Xray core and v2fly core and others (github.com)](https://github.com/2dust/v2rayN)

路由设置(分流)：
- 全局代理：所有网站都走代理
- GFWList 黑名单模式：黑名单下的国外网站代理，其他直连
  - 坏处：不在黑名单的国外网站也直连，可能无法访问
- ChinaList 白名单模式：白名单下的国内网站直连，其他代理
  - 坏处：不在白名单的国内网站也走代理，可能访问很慢
  - 解决：在系统代理设置中添加国内网站

```domain
tjupt.org;
top25.sciencedirect.com;
mthreads.com;
ieeexplore.ieee.org;
yjszs.nwpu.edu.cn;
*.gov.cn;
kimi.moonshot.cn;
muchong.com/bbs/;
www.letpub.com.cn;
www.letpub.com;
www.letpub.cn;
202.204.52.116;
*.ustb.edu.cn;
nebulapkm.cn;
*.sjtu.edu.cn;
*wanfangdata.com.cn;
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