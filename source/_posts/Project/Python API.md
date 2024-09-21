---
title: Learn API (Python)
date: 2020-02-14 14:31:32
summary: 用bilibili提供的api基于python做出的几个小栗子，获取信息，‘搞’b站，从基础开始
toc: true
categories: Project
tags:
  - Python
  - Practise
  - Bilibili
---


Python 调用API结构

<!--more-->

# Bilibili

**bilibili 提供的 api 接口(一串 json 字符)**
_让基于 bilibili 的开发更简单_
**我基于 python 写的几个使用 api 获取信息的例子**

## 1. bilibili 用户基本信息(name，level，关注，粉丝)获取

`https://api.bilibili.com/x/space/upstat?mid=UUID&jsonp=jsonp`_up 信息，名字，等级，视频总播放量，文章总浏览数_
`https://api.bilibili.com/x/relation/stat?vmid=UUID&jsonp=jsonp`_up 信息，关注数，黑名单，粉丝数_

**简单的代码获取 up 信息**

```py
import json
import requests

mid = input('输入要查询的up的uid：')
url_space = 'https://api.bilibili.com/x/space/acc/info?mid=' + mid
url_relation = 'https://api.bilibili.com/x/relation/stat?vmid='+mid
space = requests.get(url_space).content.decode()
relation =requests.get(url_relation).content.decode()
# print(type(html))
dict_space = json.loads(space)
dict_rela = json.loads(relation)
# print(dict)
up_name = dict_space["data"]["name"]
up_level = dict_space['data']['level']

up_following_num = dict_rela['data']['following']
up_follower_num = dict_rela['data']['follower']

print(f'up名字是:{up_name}')
print(f'up等级达到:{up_level}级')
if int(up_level)>=5:
    print('----哇是个大佬！！！----')
print(f'up关注了{up_following_num}个人')
if int(up_following_num)>=700:
    print('----铁定是个dd！！！----')
print(f'up有{up_follower_num}个粉丝')
```

**示例：**

```py
输入要查询的up的uid：2
up名字是:碧诗
up等级达到:6级
----哇是个大佬！！！----
up关注了191个人
up有804598个粉丝
```

## 2. bilibili 统计某视频评论区，并生成词云

- **获取某视频评论区评论**

```py
import json
import requests
from multiprocessing.dummy import Pool
import re
import os

av = input('请输入视频的av号:')
p_total = input('请输入评论要几页:')

def get_urls():
    urls = []
    p = 1
    while p <= int(p_total):
        url = 'http://api.bilibili.com/x/v2/reply?jsonp=jsonp&;pn=' + str(p) + '&type=1&oid=' + av
        urls.append(url)
        p += 1
    return urls

def get_name_con(url):
    html = requests.get(url).content.decode()
    yh_names = re.findall(r'"uname":"(.*?)","sex":',html,re.S)
    yh_contents = re.findall(r'"message":"(.*?)","plat"',html,re.S)
    del yh_contents[0]
    yh_contents2 = []
    for yh_content in yh_contents:
        yh_contents2.append(yh_content.replace('\\n',' '))
    # print(yh_contents2)
    # exit()
    return yh_names,yh_contents2

def get_names_cons():
    pool = Pool(5)
    urls = get_urls()
    namecons = pool.map(get_name_con,urls)
    names = []
    cons = []
    for namecon in namecons:
        name = namecon[0]
        for n in name :
            names.append(n)
        con = namecon[1]
        for c in con:
            cons.append(c)
    return names,cons

def save():
    tumple = get_names_cons()
    namelst = tumple[0]
    conlst = tumple[1]
    # print(len(conlst))
    # # print(type(namelst))
    # print(len(namelst))
    # exit()
    if len(namelst) != len(conlst):
        tot = len(conlst)
    g = 0
    main_path = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\txt' #修改路径-自定义
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    dir1 = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\txt\\' + 'comment'  + '.txt'  # 自定义文件名
    with open(dir1,'w',encoding='utf-8') as fb:
        for g in range(tot):
            # fb.write(namelst[g])
            # fb.write('\t\t\t')
            fb.write(conlst[g])
            # fb.write('\n')
            g += 1

if __name__ == '__main__':
    save()
    print('----已完成----',end='\t')
    print(f'此视频已获得 {p_total} 页的评论')
```

- **将生成的评论 txt 文件统计为词云**

```py
from wordcloud import WordCloud
import PIL .Image as image
import jieba

def trans_cn(text):
    word_list = jieba.cut(text)
    result = ' '.join(word_list)
    return result

def wc():
    dir1 = './txt/comment.txt'
    with open(dir1,encoding='utf-8') as f:
        text = f.read()
        text = trans_cn(text)
        WordCloud2 = WordCloud(
            font_path = 'C:\\windows\\Fonts\\simfang.ttf'
        ).generate(text)
        image_produce = WordCloud2.to_image()
        image_produce.show()
        WordCloud2.to_file('./txt/comment.png')

wc()
```

## 3. 获取 bilibili 主页各个分区的视频封面和 av 号

`https://www.bilibili.com/index/ding.json`_首页 api，每刷新一次，信息就会改变一次_
_获取的视频信息也就不同，所以可以一直获取信息(理论上来说)_
_缺点是每次只能获取十张图片信息_
_用的是 wb 写入文件，所以即使文件有一样的也会被覆盖..._

```py
import requests
import re
import os
import json

print('-douga-teleplay-kichiku-dance-bangumi-fashion-life-ad-guochuang-movie-music-technology-game-ent--')
fenqu = input('请输入爬取分区:')
if fenqu == '':
    fenqu1 = 'shuma'
else :
    fenqu1 = fenqu

html = requests.get(
    'https://www.bilibili.com/index/ding.json').content.decode()

dict_html = json.loads(html)
i = 0
aids = []
pics = []

for i in range(10):
    aid = dict_html[fenqu][str(i)]['aid']
    pic = dict_html[fenqu][str(i)]['pic']
    aids.append(aid)
    pics.append(pic)

j = 1
h = j-1
for h in range(10):
    main_path = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\pic\\'+fenqu1
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    try:
        piccc = requests.get(pics[h])
    except requests.exceptions.ConnectionError:
        print('图片无法下载')
        continue
    except requests.exceptions.ReadTimeout:
        print('requests.exceptions.ReadTimeout')
        continue
    dir = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\pic\\' + \
         fenqu1 + '\\'  +'av' + str(aids[h]) + '.jpg'
    with open(dir, 'wb') as f:
        print(f'正在爬取第{j}张图')
        f.write(piccc.content)
    j += 1
    h += 1

print('----完成图片爬取----')
```

_略微修改后_
_可能就是因为有重复的，会覆盖前面已下载的_
_爬个 5 次本该有 50 张，但才有 20 几张(dance 区)_
_可能 dance 区首页视频比较少吧，游戏区很多_
**不管了反正这个爬虫也没什么用 hhh**

```py
import requests
import re
import os
import json

def get_pic():
    if fenqu == '':
        fenqu1 = 'shuma'
    else :
        fenqu1 = fenqu

    html = requests.get(
        'https://www.bilibili.com/index/ding.json').content.decode()

    dict_html = json.loads(html)
    i = 0
    aids = []
    pics = []

    for i in range(10):
        aid = dict_html[fenqu][str(i)]['aid']
        pic = dict_html[fenqu][str(i)]['pic']
        aids.append(aid)
        pics.append(pic)

    j = 1
    h = j-1
    for h in range(10):
        main_path = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\pic\\'+fenqu1
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        try:
            piccc = requests.get(pics[h])
        except requests.exceptions.ConnectionError:
            print('图片无法下载')
            continue
        except requests.exceptions.ReadTimeout:
            print('requests.exceptions.ReadTimeout')
            continue
        dir = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\pic\\' + \
            fenqu1 + '\\'  +'av' + str(aids[h]) + '.jpg'
        with open(dir, 'wb') as f:
            print(f'正在爬取第{j}张图')
            f.write(piccc.content)
        j += 1
        h += 1

to = int(input('请输入你要爬多少次---一次最多十张：'))
print('-douga-teleplay-kichiku-dance-bangumi-fashion-life-ad-guochuang-movie-music-technology-game-ent--')
fenqu = input('请输入爬取分区:')
for i in range(to):
    get_pic()
    print(f'----完成第{i}次图片爬取----')
```

> [Github 源码链接](https://github.com/yq010105/spider_learn/tree/master/bilibili/bilibili_api)

## 4. 主站上的实时人数

_所用 api 接口_`https://api.bilibili.com/x/web-interface/online?&;jsonp=jsonp`

```py
import requests
import json
import time

def print_num():
    index = requests.get(
    'https://api.bilibili.com/x/web-interface/online?&;jsonp=jsonp').content.decode()
    dict_index = json.loads(index)
    all_count = dict_index['data']['all_count']
    web_online = dict_index['data']['web_online']
    play_online = dict_index['data']['play_online']
# 应该是人数和实时在线人数
    print(f'all_count:{all_count}')
    print(f'web_online:{web_online}')
    print(f'play_online:{play_online}')


for i in range(100):
    print(f'第{i+1}次计数')
    print_num()
    time.sleep(2)
```

## 5. 用户的粉丝数

_只能获取一页，b 站最多是五页，多了就会有限制_

```py
import requests
import json
import csv
import os
import time

uid = input('请输入查找的up主的uid:')
url = 'https://api.bilibili.com/x/relation/followers?vmid=' + \
    uid + '&ps=0&order=desc&jsonp=jsonp'

html = requests.get(url).content.decode()
dic_html = json.loads(html)

index_order = dic_html['data']['list']
mids, mtimes, unames, signs = [], [], [], []
for i in index_order:
    mid = i['mid']
    mids.append(mid)
    mtime = i['mtime']
    mmtime = time.asctime(time.localtime(mtime))
    mtimes.append(mmtime)
    uname = i['uname']
    unames.append(uname)
    sign = i['sign']
    signs.append(sign)
# print(index_order)
# print(mids)
headers = ['uid', '注册时间', 'up姓名', '个性签名']
rows = []
j = 0
for j in range(len(mids)):
    rows.append([mids[j], mtimes[j], unames[j], signs[j]])

main_path = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\csv'
if not os.path.exists(main_path):
    os.makedirs(main_path)

dir = 'E:\\learn\\py\\git\\spider\\spider_learn\\bilibili\\bilibili_api\\csv\\' + \
    'follers' + '.csv'

with open(dir, 'w', encoding='utf-8') as f:
    fb = csv.writer(f)
    fb.writerow(headers)
    fb.writerows(rows)


print('----最多只显示一页的粉丝数，也就是50个----')
print(f'共有{len(mids)}个粉丝')
```

# LOL

LOL提供了丰富的接口，可以利用这些接口来完成一些操作。使用Python来调用LOL提供的API，可以使用一些大佬造的轮子。

拳头提供的API接口说明可以在[开发者网站](https://developer.riotgames.com/)中查看，可以使用拳头账号获得一个api key来调用api，且key需要每天进行更换。

![riot_developer.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/riot_developer.png)
![Uploading file...i2f0f]()


[网站API](https://developer.riotgames.com/apis)
- [抓包得到的API](http://www.mingweisamuel.com/lcu-schema/tool/#/)

拳头的开发者网站文档：
在[开发者网站doc](https://developer.riotgames.com/docs/lol)可以得到一些json数据，如当前版本全部英雄的信息，全部装备的信息等等

## 1.RiotWatcher

[RiotWatcher](https://riot-watcher.readthedocs.io/en/latest/)参考知乎大佬[京暮研Shinra](https://www.zhihu.com/people/luo-po-gong-zi)的[教程](https://www.zhihu.com/column/c_1376912368698499072)来进行初步学习

### 1.1 使用方式
- 安装`pip install riotwatcher`
- 使用

```python
from riotwatcher import LolWatcher
# api-key需要自己去官网上获得
lol_watcher = LolWatcher('<your-api-key>')
```

定义完成后，就可以使用lol_watcher来调用其他的类库
- [League of Legends Watcher](https://riot-watcher.readthedocs.io/en/latest/riotwatcher/LeagueOfLegends/index.html)
- Legends Of Runeterra Watcher
- Riot Watcher
- Team Fight Tactics Watcher
- Valorant Watcher
- Handlers
- Testing

每个类下拥有很多个函数，调用不同的API

### 1.2 League of Legends Watcher
champion：英雄
summoner：召唤师

#### 获得某个区服的免费英雄

[champion](https://riot-watcher.readthedocs.io/en/latest/riotwatcher/LeagueOfLegends/ChampionApiV3.html#riotwatcher._apis.league_of_legends.ChampionApiV3)


使用`ChampionInfo = lol_watcher.champion.rotations(region: str)`
需要输入一个服务器地区名称region，如日服jp1，并且返回一个ChampionInfo值

```py
region = ['kr','jp1','br1','eun1','euw1','la1','la2','na1','oc1','tr1','ru']
lol_region = region[1]
champion_kr = lol_watcher.champion.rotations(lol_region)
print(champion_kr)

#return:
{'freeChampionIds': [21, 33, 50, 57, 80, 81, 107, 111, 113, 202, 240, 246, 350, 497, 518, 875], 'freeChampionIdsForNewPlayers': [222, 254, 427, 82, 131, 147, 54, 17, 18, 37], 'maxNewPlayerLevel': 10}
```
返回的ChampionInfo中
- freeChampionIds 免费英雄的ID
- freeChampionIdsForNewPlayers 给新手玩家的免费英雄

可以通过ID来查看英雄的名字，使用[拳头官网的json文件](http://ddragon.leagueoflegends.com/cdn/12.9.1/data/zh_CN/champion.json)
![2022-06-09-10-02-03.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/2022-06-09-10-02-03.png)


实现思路：

![20220609214627](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220609214627.png "实现思路")

```py
import requests
import pandas as pd

# 在官网获取英雄的json数据，zh_CN：中文数据
champion_json_url = "http://ddragon.leagueoflegends.com/cdn/12.9.1/data/zh_CN/champion.json"
# 使用request的get获得
r1 = requests.get(champion_json_url)
# 只需要data中的数据
champ_data = r1.json()["data"]
# 处理json数据，使用pandas的json_normalize
champ_df = pd.json_normalize(champ_data.values(),sep='')

# 取出champion返回值中的freeChampionIds 
champions_kr_free = champion_kr['freeChampionIds']
# 将freeChampionIds列表中的int数据转换成字符串并保存在champions_kr_free_str中
champions_kr_free_str = []
for champion_kr_free in champions_kr_free:
    champion_kr_free_str = str(champion_kr_free)
    champions_kr_free_str.append(champion_kr_free_str)
# 在json文件的key值中搜索champions_kr_free_str，搜索到后将其提取出来，并打印name和title两列
print(champ_df[champ_df['key'].isin(champions_kr_free_str)][['name','title']])

# output
        name    title
28      探险家  伊泽瑞尔
48      戏命师     烬
61     暴怒骑士    克烈
73     扭曲树精    茂凯
75     赏金猎人  厄运小姐
81     深海泰坦  诺提勒斯
82     万花通灵    妮蔻
89     不屈之枪    潘森
92     元素女皇   奇亚娜
94       幻翎     洛
95     披甲龙龟   拉莫斯
100   傲之追猎者  雷恩加尔
105    北地之怒   瑟庄妮
108      腕豪    瑟提
118  诺克萨斯统领   斯维因
151    魔法猫咪    悠米
```

**问题**

pandas在pycharm中使用时，在打印列表式，单元格内容会显示不全，采用如下方法来处理
```py
pd.set_option('display.max_columns', None)   # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行
pd.set_option('display.expand_frame_repr', False)  # 设置不折叠数据
pd.set_option('display.max_colwidth', 200)
```

#### 获得某个召唤师的英雄熟练度等信息

[champion_mastery](https://riot-watcher.readthedocs.io/en/latest/riotwatcher/LeagueOfLegends/ChampionMasteryApiV4.html#riotwatcher._apis.league_of_legends.ChampionMasteryApiV4)

**有三种获取方法可以选择**

##### Ⅰ.by_summoner
*获得所有英雄的熟练度信息*

![2022-06-09-11-21-43.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/2022-06-09-11-21-43.png)


使用：`lol_watcher.champion_mastery.by_summoner(region,summoner_id)`
return: List[ChampionMasteryDTO]: This object contains a list of Champion Mastery information for player and champion combination.

```py
faker_champion_master = lol_watcher.champion_mastery.by_summoner(region,faker_id)
print(faker_champion_master)

# output
[
    {
        "championId": 7,
        "championLevel": 7,
        "championPoints": 493741,
        "lastPlayTime": 1653499775000,
        "championPointsSinceLastLevel": 472141,
        "championPointsUntilNextLevel": 0,
        "chestGranted": true,
        "tokensEarned": 0,
        "summonerId": "ht4SB4hyU2CyiERUxtIE6ZD4bQbG4Djo0Hbrl2hU0ilBxg"
    },
    {},...{

    }
]
```
返回数据包括
- 英雄id
- 英雄熟练度等级
- 英雄分数
- 最后使用时间
- 是否有战利品奖励（使用该英雄的对局中有人获得S-以上评分）
等等

##### Ⅱ.by_summoner_by_champion
*获得特定英雄的熟练度信息*

使用：`lol_watcher.champion_mastery.by_summoner_by_champion(region,summoner_id,champion_id)`
return: ChampionMasteryDTO: This object contains single Champion Mastery information for player and champion combination.

```py
faker_champion_master_7 = lol_watcher.champion_mastery.by_summoner_by_champion(region,faker_id,7)
print(faker_champion_master_7)

#output
{'championId': 7, 'championLevel': 7, 'championPoints': 493741, 'lastPlayTime': 1653499775000, 'championPointsSinceLastLevel': 472141, 'championPointsUntilNextLevel': 0, 'chestGranted': True, 'tokensEarned': 0, 'summonerId': 'ht4SB4hyU2CyiERUxtIE6ZD4bQbG4Djo0Hbrl2hU0ilBxg'}
```

##### Ⅲ.scores_by_summoner
*获得玩家的总冠军精通分数，即每个冠军精通等级的总和*

使用：`lol_watcher.champion_mastery.scores_by_summoner(region,summoner_id)`

```py
faker_champion_master_scores = lol_watcher.champion_mastery.scores_by_summoner(region,faker_id)
print(faker_champion_master_scores)

#output
675
```


#### 查询一名召唤师的信息

[summoner](https://riot-watcher.readthedocs.io/en/latest/riotwatcher/LeagueOfLegends/SummonerApiV4.html#riotwatcher._apis.league_of_legends.SummonerApiV4)

**可以通过四种方式查询**

使用：`summonerDTO = lol_watcher.summoner.by_xxxx(region,xxxx)`
return:	SummonerDTO: represents a summoner

##### Ⅰ.by_account

##### Ⅱ.by_id

##### Ⅲ.by_name

![2022-06-09-11-06-41.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/2022-06-09-11-06-41.png)


```py
region = "kr"
summoner_name = 'Hide on bush'

summoner_faker = lol_watcher.summoner.by_name(region, summoner_name)
print(summoner_faker)

# output
{'id': 'ht4SB4hyU2CyiERUxtIE6ZD4bQbG4Djo0Hbrl2hU0ilBxg', 'accountId': 'U9Wyp4etrbQuDFz3dqh8pA9HvqhLm1t7tKcwJbK13q1S', 'puuid': 'f51LYPjnONGFBPsvPdcXZVLr8JvlvVTb4SvGH0xZqGVe3sVsrUKoKZvlKciTh9xkLVl4npQ83zFLWQ', 'name': 'Hide on bush', 'profileIconId': 6, 'revisionDate': 1654345192719, 'summonerLevel': 571}
```
##### Ⅳ.by_puuid



## 2.lcu-driver
[lcu-driver](https://lcu-driver.readthedocs.io/en/latest/index.html)