---
title: Note!!!
date: 2023-06-27T16:09:09.000Z
top: true
tags:
  - NeRF
categories: NeRF
date updated: 2023-08-09T22:24:16.000Z
---

| My post                                                                                                                                                        | Brief description     | status          |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --------------- |
| [NeRF](/NeRF/NeRF-Principle) + [Code](/NeRF/NeRF-code)                                                                                  | NeRF原理 + 代码理解         | Completed       |
| [Neus](/NeRF/SurfaceReconstruction/Neus) + [Code](/NeRF/SurfaceReconstruction/Neus-code)                                         | 表面重建方法 SDFNetwork     | Completed       |
| [InstantNGP](/NeRF/Efficiency/NeRF-InstantNGP) + [Tiny-cuda-nn](/NeRF/Efficiency/NeRF-InstantNGP-code)                                 | 加速NeRF的训练和推理          | Completed（Tcnn） |
| [Instant-nsr-pl](/NeRF/SurfaceReconstruction/Neus-Instant-nsr-pl) + [Code](/NeRF/SurfaceReconstruction/Neus-Instant-nsr-pl-code) | Neus+Tcnn+NSR+pl      | Completed       |
| [Instant-NSR](/NeRF/SurfaceReconstruction/Instant-NSR) + [Code](/NeRF/SurfaceReconstruction/Instant-NSR-code)                    | 快速表面重建                | Completed       |
| [NeRO](/NeRF/SurfaceReconstruction/Shadow&Highlight/NeRO) + [Code](/NeRF/SurfaceReconstruction/Shadow&Highlight/NeRO-code)       | 考虑镜面和漫反射的体渲染函数        | In Processing   |
| [NeRF-Mine](/NeRF/NeRF-Mine)                                                                                                                       | 基于Instant-nsr-pl创建的项目 | Completed       |

| My post                                      | Brief description  | status                                                                                                       |
| -------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------ |
| [Nerfstudio](/NeRF/NeRF-Studio)  | 一个更加方便训练和查看训练过程的框架 | Completed in AutoDL                                                                                          |
| [NeRF Review](/NeRF/NeRF-review) | 追踪一些NeRF的进展和笔记link | Following [Jason陪你练绝技 in Bilibili](https://space.bilibili.com/455056488/channel/collectiondetail?sid=910368) |

Related link :  [3D Reconstruction](https://paperswithcode.com/task/3d-reconstruction) | [awesome-NeRF-papers](https://github.com/lif314/awesome-NeRF-papers)

<!-- more -->

<p style= "text-align: center;">流程图点击编辑，更方便观看</p>
<div style="text-align:center">
    <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230627160834.png" alt="Image" />
</div>

基于Drawio绘制的流程图，导出为嵌入的Iframe

<iframe frameborder="0" style="width:100%;height:243px;" src="https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1#RnZTBbqMwEEC%2FZo6VwFCwj5hAq1WTQ3Poqjc2uIAEmLhOSfr1tY1JQBBpd6VIsd%2BMxzBvBHhxc34SWVduec5qQE5%2BBm8DCLme76s%2FTS4DCd1gAIWocpt0A%2Fvqm1noWHqqcvY5S5Sc17Lq5vDA25Yd5IxlQvB%2BnvbB6%2FmtXVawBdgfsnpJ36pclgPFKLzxZ1YV5XizG5Ah0mRjsn2TzzLLeT9BXgJeLDiXw6o5x6zWzRv7MpxL70SvDyZYK%2F%2FmwPvuuP%2Bmv5vt8aW7vL573OXPD7bKV1af7AtDQgA7QCJIMNAUcAhJCBgBJZA86i2lmtAYSGByEiAuJKkmODY5KUQpJL7OjB41UTkYQxLoI7qOr4kqrkIkBuqsFVQhZAqqU6mpE%2BpkTDWh6hZXkwgbopKpPjh0Wl5GfYKf2pzpDrjg0b6sJNt32UFHezWwipWyqW34g7fSTqCaUo8uOzy2iwnJzhNkO%2F7EeMOkuKgUG0WBtW%2FHf9z2t1m6snIyR9iyzI5vca18M6wWVvK68C%2F0uj2%2BBWLX5H9SVOyY%2F4s8oBXhvnZC0jV1ajE0WGnZ6J%2FOwcZYoG1E8cShmRdikiPXqPMNMaeiDRA8sboQfl%2Bd87%2FqFp5WbN5X58zVXfcTdx5ecec6%2Fy5PbW9fAhObfE%2B95Ac%3D"></iframe>

