---
title: (WW) Isogeometric analysis
date: 2024-12-03 22:14:03
tags:
  - 
categories: Blog&Book&Paper/Read/Papers
---

Wei Wang 王伟 [北京航空航天大学主页平台系统 王伟--LOGOS--研究领域](https://shi.buaa.edu.cn/publius/zh_CN/index.htm)  jrrt@buaa.edu.cn
Xiaoxiao Du 杜孝孝 [北京航空航天大学主页平台系统 杜孝孝--中文主页--首页](https://shi.buaa.edu.cn/duxiaoxiao1/zh_CN/index.htm) 机械工程及自动化 duxiaoxiao@buaa.edu.cn
Jiaming Yang 杨佳明 [北京应用物理与计算数学研究所人力资源](http://www.iapcm.ac.cn/Human_con_335.html) williamyjm@163.com

Pengfei Zhang Research Institute of Aero-Engine, Beihang University, ftd423@buaa.edu.cn
Pengfei Han Research Institute of Aero-Engine, Beihang University, hanpf@buaa.edu.cn
Yazui Liu Research Institute of Aero-Engine, Beihang University, liuyazui@buaa.edu.cn

Gang Zhao School of Mechanical Engineering & Automation, Beihang University, zhaog@buaa.edu.cn CNC通讯
[北航党委副书记赵罡履新清华党委副书记、纪委书记_人事风向_澎湃新闻-The Paper](https://m.thepaper.cn/wifiKey_detail.jsp?contid=19749649&from=wifiKey#) 几何造型、数字化设计与制造技术、飞机数字化装配技术及装备

<!-- more -->

# Wei Wang

[Isogeometric Shape and Topology Optimization of Kirchho-Love Shell Structures Using Analysis-Suitable Unstructured T-splines](https://cad-journal.net/files/vol_22/CAD_22\(2\)_2025_245-260.pdf) 2025 等几何分析+结构拓扑优化 --> 壳模型

壳模型，好的设计只需耗费少量材料即可达到很高的刚度和强度。过去十年，通过数值工具进行优化设计。
FEMs将壳模型划分为面单元(线性单元)，导致几何不准确和continuity reduction，进一步影响结构优化的有效性。并且CAD设计的模型与CAE分析的模型之间的几何数据频繁交换是非常耗时的，IGA等几何分析统一几何表示解决这一问题。**样条函数**在CAD中表示几何，在CAE中作为*形状函数*用于仿真。

IGA被用于结构形状/拓扑优化设计中
- http://doi.org/10.1016/j.ijsolstr.2010.03.004. NURBS-based isogeometric shape optimization of shell structures (based on Reissner-Mindlin (RM) shell theory)
- http://doi.org/10.1016/j.cma.2014.02.001. semi-analytical sensitivity analysis and sensitivity weighting method for NURBS-based isogeometric shape optimization (Kirchho-Love (KL) shell formulation.)
- To handle the complex design domain problems,
- http://doi.org/10.17863/CAM.22608.  used **subdivision surfaces** for isogeometric structural shape optimization
- http://doi.org/10.1016/j.finel.2016.06.003. considered the topologically complex geometries built with trimmed patches in the shape optimizatio
- http://doi.org/10.1016/j.cma.2016.11.012. 太原理工采矿工程 combined T-splines and isogeometric boundary element method for shape sensitivity analysis
-  http://doi.org/10.1016/j.cma.2019.02.042. investigated the shape optimization of non-conforming stiened multi-patch structure
- the optimal design of composite shell：
  - http://doi.org/10.1016/j.cma.2013.05.019.
  - http://doi.org/10.1016/j.cma.2019.05.044. 大连理工大学计算力学国际研究中心 haopeng@dlut.edu.cn
  - http://doi.org/10.1016/j.tws.2023.110593. 哈尔滨工程大学 电力与能源工程学院 guoyongjin@hrbeu.edu.cn
  - http://doi.org/10.1016/j.euromechsol.2023.105142. 湘潭大学机械工程与力学学院 yinsh@xtu.edu.cn
  - http://doi.org/10.1016/j.ijsolstr.2020.11.003.
- calculation of sensitivity （等几何结构优化中）  http://doi.org/10.1016/j.apm.2020.07.027.
- optimization algorithms  http://doi.org/10.1016/j.finel.2023.103989.  河海大学(南京) 力学与材料学院 tiantangyu@hhu.edu.cn
- adaptive refinement http://doi.org/10.1007/s00158-020-02645-w. 西北工业大学机械工程学院 zhangwh@nwpu.edu.cn  liang.meng@nwpu.edu.cn
- thickness distribution http://doi.org/10.1108/EC-08-2017-0292.  湖南大学 车身先进设计与制造国家重点实验室


# Gang Zhao

[Two-dimensional frictionless large deformation contact problems using isogeometric analysis and Nitsche’s method](https://watermark.silverchair.com/qwab070.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA00wggNJBgkqhkiG9w0BBwagggM6MIIDNgIBADCCAy8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM0Gx3nnkwsVxWiCmvAgEQgIIDAGzZZrbDVhlzSY10JALAlDm5MvXCgthcBQV_wiRFeg1fm5dcEYXyVCH9_KRtBDfvd_SVRN-BwtjQaGVwOQjYZEbKYeggrxcOcjM99ftpurQpgSEskSQp2lo-y1oGpkYy6Db0Ydyoi1Lg6K5DUWoD4uLv7ChmETpQtPgdC9vLsMmokVbjLM5k4kCwye3Bv_b29YZwVwQBDHrUyAfBPPMaAMoZXzGQi3bQCnLbCSgi8V16bQx9nXnq2j2jR0zJGw4UqRIGviDNv-lFdeG7uDGUrccT8P_b0hzaQGOv6yDhalhiDPCEVyaWivzEYRWZ07e3IIfPeSO6J6hlFm1j3NldSybYjIcSHk93YpF-AvDOmm18QXJ3jA7ijEm72_udPLCTfR_raUKHXA3ICoPIn57CohunCSyA5v-OVbYINIh68EMaiLGfJBTdgYzCRAU-sZj4plQWzSd_2qHSdXuF25Wm_I6kcB7S5LDTFqiziW6yQwG5abmqofxt3H71_mnzkI0si-fxQ4WRCgkNzWPnrYREC0kyjNpj44J2bUaJrDcpcZZU5OZs3pkHYXm_qeXHjP3FaTsfIRGvzyHKREeE0dMzdRxqBDluYGRRfUMFzRLASXialHMrcDW0jfPNRi6Oh6WnVV2jHse5DxnYW45OeBUv7978Wyshk8q_TyQWJD1FdWSc-SusDHYe9--vw-ijnDhUzN5lAsXPE4DJcOX-okAxIn8B6RIm4vKhDgd6G8x7VbmNDZgLjnRbTv2_o2eRa5eltptXoHof28QLwisV9-G1lXobyCX8sufVd1okftFvpZWqSYZCKJr6Vlm2-Qs_eV9M6sFNPuoI1w3se4psQi7KFtAPbYSBPsGNxzPwFg7ldZ1ze9eKj9Ve4ylp-3zbekXhpOKWmaRGY2UAqaOZx4et2OjjvyThwDkWFLoS7wkGGMngTQQusp-chPBHkvOEoNWJNVbYE54jddd1rbFpskFMV1Sckl3HXhqenWBpG0zDBtO-69xASIv2vLR_kKo4c2lLzA) 2021

[A milling cutting tool selection method for machining features considering energy consumption in the STEP‑NC framework](file:///F:/Download/s00170-022-08964-0.pdf) 2022 铣刀




