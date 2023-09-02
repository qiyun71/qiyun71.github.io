---
title: Python调用LOL的API
date: 2022-06-08 21:14:26
tags:
  - LOL
  - Python
categories: Learn
---

LOL提供了丰富的接口，可以利用这些接口来完成一些操作。使用Python来调用LOL提供的API，可以使用一些大佬造的轮子。

<!--more-->

拳头提供的API接口说明可以在[开发者网站](https://developer.riotgames.com/)中查看，可以使用拳头账号获得一个api key来调用api，且key需要每天进行更换。

![riot_developer.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/riot_developer.png)
![Uploading file...i2f0f]()


[网站API](https://developer.riotgames.com/apis)
- [抓包得到的API](http://www.mingweisamuel.com/lcu-schema/tool/#/)

拳头的开发者网站文档：
在[开发者网站doc](https://developer.riotgames.com/docs/lol)可以得到一些json数据，如当前版本全部英雄的信息，全部装备的信息等等

# 1.RiotWatcher

[RiotWatcher](https://riot-watcher.readthedocs.io/en/latest/)参考知乎大佬[京暮研Shinra](https://www.zhihu.com/people/luo-po-gong-zi)的[教程](https://www.zhihu.com/column/c_1376912368698499072)来进行初步学习

## 1.1 使用方式
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

## 1.2 League of Legends Watcher
champion：英雄
summoner：召唤师

### 1.2.1 获得某个区服的免费英雄

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

### 1.2.2 获得某个召唤师的英雄熟练度等信息

[champion_mastery](https://riot-watcher.readthedocs.io/en/latest/riotwatcher/LeagueOfLegends/ChampionMasteryApiV4.html#riotwatcher._apis.league_of_legends.ChampionMasteryApiV4)

**有三种获取方法可以选择**

#### Ⅰ.by_summoner
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

#### Ⅱ.by_summoner_by_champion
*获得特定英雄的熟练度信息*

使用：`lol_watcher.champion_mastery.by_summoner_by_champion(region,summoner_id,champion_id)`
return: ChampionMasteryDTO: This object contains single Champion Mastery information for player and champion combination.

```py
faker_champion_master_7 = lol_watcher.champion_mastery.by_summoner_by_champion(region,faker_id,7)
print(faker_champion_master_7)

#output
{'championId': 7, 'championLevel': 7, 'championPoints': 493741, 'lastPlayTime': 1653499775000, 'championPointsSinceLastLevel': 472141, 'championPointsUntilNextLevel': 0, 'chestGranted': True, 'tokensEarned': 0, 'summonerId': 'ht4SB4hyU2CyiERUxtIE6ZD4bQbG4Djo0Hbrl2hU0ilBxg'}
```

#### Ⅲ.scores_by_summoner
*获得玩家的总冠军精通分数，即每个冠军精通等级的总和*

使用：`lol_watcher.champion_mastery.scores_by_summoner(region,summoner_id)`

```py
faker_champion_master_scores = lol_watcher.champion_mastery.scores_by_summoner(region,faker_id)
print(faker_champion_master_scores)

#output
675
```

### 1.2.4 clash
### 1.2.5 data_dragon
### 1.2.6 league
### 1.2.7 lol_status
### 1.2.8 lol_status_v3
### 1.2.9 lol_status_v4
### 1.2.10 match
### 1.2.11 match_v4
### 1.2.12 match_v5
### 1.2.13 spectator

### 1.2.14 查询一名召唤师的信息

[summoner](https://riot-watcher.readthedocs.io/en/latest/riotwatcher/LeagueOfLegends/SummonerApiV4.html#riotwatcher._apis.league_of_legends.SummonerApiV4)

**可以通过四种方式查询**

使用：`summonerDTO = lol_watcher.summoner.by_xxxx(region,xxxx)`
return:	SummonerDTO: represents a summoner

#### Ⅰ.by_account

#### Ⅱ.by_id

#### Ⅲ.by_name

![2022-06-09-11-06-41.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/2022-06-09-11-06-41.png)


```py
region = "kr"
summoner_name = 'Hide on bush'

summoner_faker = lol_watcher.summoner.by_name(region, summoner_name)
print(summoner_faker)

# output
{'id': 'ht4SB4hyU2CyiERUxtIE6ZD4bQbG4Djo0Hbrl2hU0ilBxg', 'accountId': 'U9Wyp4etrbQuDFz3dqh8pA9HvqhLm1t7tKcwJbK13q1S', 'puuid': 'f51LYPjnONGFBPsvPdcXZVLr8JvlvVTb4SvGH0xZqGVe3sVsrUKoKZvlKciTh9xkLVl4npQ83zFLWQ', 'name': 'Hide on bush', 'profileIconId': 6, 'revisionDate': 1654345192719, 'summonerLevel': 571}
```
#### Ⅳ.by_puuid



### 1.2.15 third_party_code

# 2.lcu-driver
[lcu-driver](https://lcu-driver.readthedocs.io/en/latest/index.html)