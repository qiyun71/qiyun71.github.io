---
title: Neus代码理解
date: 2023-06-30 13:45:48
tags:
    - Code
    - Python
    - Neus
categories: NeRF/Surface Reconstruction
---

[Neus代码](https://github.com/Totoro97/NeuS)的理解

NeRF与Neus相机坐标系的对比：

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230703144039.png)

| Method | Pixel to Camera coordinate                                                                                                                                                                                                                                                                                         | 
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NeRF   | $\vec d = \begin{pmatrix} \frac{i-\frac{W}{2}}{f} \\ -\frac{j-\frac{H}{2}}{f} \\ -1 \\ \end{pmatrix}$ , $intrinsics = K = \begin{bmatrix} f & 0 & \frac{W}{2}  \\ 0 & f & \frac{H}{2}  \\ 0 & 0 & 1 \\ \end{bmatrix}$                                                                                              | 
| Neus   | $\vec d = intrinsics^{-1} \times  pixel = \begin{bmatrix} \frac{1}{f} & 0 & -\frac{W}{2 \cdot f}  \\ 0 & \frac{1}{f} & -\frac{H}{2 \cdot f} \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{pmatrix} i \\ j \\ 1 \\ \end{pmatrix} = \begin{pmatrix} \frac{i-\frac{W}{2}}{f} \\ \frac{j-\frac{H}{2}}{f} \\ 1 \\ \end{pmatrix}$ |     


<!-- more -->

# Code

## Runner().train流程图
<iframe frameborder="0" style="width:100%;height:1153px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=Runner.drawio#R7V1Zk6M4Ev41RFd1RBPcx6Ndx8zG7vTudE3s7DwRlC3bTBlwA66jf%2F1KQtiA0gbbXG57ImraiEtk6stM5SEJ6p3%2F%2Fkvkrha%2FhVO0FBRp%2Bi6o94KiaJZq4H9Iy0faohiWlrbMI2%2BatsnbhifvB2KNEmtde1MUFy5MwnCZeKti4yQMAjRJCm1uFIVvxctm4bL41pU7R1zD08Rd8q1%2FetNkkbZairlt%2FxV580X2Ztmw0zO%2Bm13MviReuNPwLdekPgjqXRSGSfrLf79DS0K9jC7pfY87zm46FqEgqXPDaKV9cxdf3e9%2F%2BN%2FXc%2FlpFKhfvrCnvLrLNftgQTGW%2BHnjWYgfi3udfDBSGN%2FXYXbiS0wZNcIXyMbqfXsS%2F5qTf7%2BtMTui7Fm4U%2Bnj0pOMHpsnK1G4DqaI9FPGp98WXoKeVu6EnH3D4wq3LRJ%2FyU6TJ7FxIhubp%2BXpkH0UihL0nmtidPkFhT5Kog98SXY2YxobpXLGs7cty2WJtS1y7NZYm8tG2Xzz6C0j8A%2FGiwP4YnA0QlM8LtlhGCWLcB4G7vJh2zreUlHCR9tr%2FhWGK0a7v1GSfDDiueskLFIWvXvJ%2F8jtos6O%2FsqduX9nT6YHH9lBgD83dxM5%2FCt%2FbnsbPcruq83FOFxHE7SHVAqTCm40R8me65jcIXTcOyYitHQT77WI%2F8YZrHDAW0XhBMVxKstm3pxIKw8zdjdYpH7AosuY1Xm4KDoAF1Xn4aK3BRfzCpe6cNFqwkUfFFw0Di73buLGKLm5PQOAqJA%2BgQDSmj6RrStC6iJEr4kQ2RwURHQOIl9R8hZGL%2FE5IMTqW4VYAPm%2BPd58%2Fhyj5UwkKlnQx598MsMRsWk7%2ByTo98OTPapctGRVmaerqnZpyWYdyBH26f6RDU2YvPF05gTpBWdCZaN3KvMTuScvmC%2FRf93Ic4MJ2kvwV3bReVFdA8a2ZnRKdd6K%2F4YwjSJM%2Br0Ej7KrzoziwDjXuh3nKkdxjmrxwl2Rn5N1tPwYR%2B7khajxKvIVfRAtEFOxlbLSk3WOnApATdlqi5wb39TVLKw0CzNzr9ouZKAZiF2Y9TsHmSSMJgsxXCWeL46mrj84wcMbiJrOY6VbAxGQNFesVECgEivZE4eCFZvDyle0fkqVOorOwtmgq31PpZSrO66%2B9zoLalUiZWD%2Ba37CsQzdKW6ZLNDkZRV6NN5Dwz%2FPURb5IQ5t5xlbZOvVGSDJkHpHEm%2FueoGXcLRLyPxtTnHUPeG0UvDMNEVAV0sb67dAOb0lyqlXy7a%2BDKobE1CGZdkqfFQgonFnMYlcLzgLbW2YvcuYa6y5PlLqxgaUYQGFjw1ELEFDoodSmq9xFoiRew8VKHysgAqcQatlwzD7V8rXKXR9UVN3Cq0Oawqt8FPop7Xvu9HHnxGGwHnIGFPrW8ZkL7tCpRoqat05tKoOCioqP4der6ZugpwlcqOAhIcifDRExKiKmIViGGQsKMfFVEWzU9TYV9TURo1aFzXWsFDD%2B0Rw%2Fx3Pd%2BfIWaHIHyJeiIaRC3ixe3cuadecsPpgqesi0YaVFKbyLpI5CrBaCaahj%2F%2F5iB13kDmUmmaIuilt%2FlNKygZyMVqGaJldaht%2BHthM9YRPJ5MSm1RKS4Kt8ymlEKWSpIMm7J1WU2QPvjIqjzBbKrAJyKbqmEsKzyV9%2FOwmk4VDGaHcEUZIgn5P%2FVaELfgCKsRCcbJaEzlGLqEtr%2FmWSbgMo%2FSn78Yv%2BC56%2B136HJk8Ub%2FnWIQpmRT5ECdR%2BILu6NPU%2ByAMiEqcectlqcldevMAH04wfxBuHxO%2BeBN3OWInfG863ekXKgrfVkCqlXgPGO9gAqPSGvOBSJc%2Fd7zpO%2BER5VR%2BJFwMp3S9yCkD1HxQ1mN7rDJ4nD7Ygj0SrEfhwRDsO2E8Ii1Y7lu28KCTdtsQHjRhbJFG3DK%2BE0b0hz0Wxg%2FCg0muxMr74ZGcsu7oNbYwNvY%2B2RCssTCy6cBgjx%2BN6AtVeocpjDX6HlOwNcGmz7AswaZvtnT6DJM8cvx4QUNKMYtDyqw9pMy2hpR%2BdQzXL%2BCqm4eYTTgHMhXR%2BDzEALmRMyN%2FEZ6NxKsFigbp7aKTEaOoMWHUdD0F0Xhve2oP5Syh6QXJtrK6lHWQTXKX%2BlLnZx9k4KccmpEfF8MeQ1VFreg3ViSIQ9YWcJ3wyJA5LlwVUIVeqVRAmQnRvQKyvv%2FD%2BLf3I%2FDff3%2F%2BcL%2BPvX8%2BhH2npeY4vOV3szxulqcwEY1BGRVZv%2FP6jyZrD9KMMFUxyx3ZrIcD6iciJrUOzQidTxDK%2FFhT7zXzYW0si8ysuGN6bMa0WXpHvHKD7BaS6zunhHei%2BXPON5a%2FKNecexvQAegNkzB23AB3ZEmCpV544DsuRvXqWbR5s2gHNPK6dfoY1ySH%2BhLaqKt1h1UooPPeolx9N6n1vlkl8RCltaGLVtFW1TQwtV2zxE4DtzoQdyo6yqXP%2BC9wYtdfLVG88aBSR7di4FZMo%2BA5Xl2WCDRkTZSL%2BlcDixWwAhbVLqcfOj%2BNr%2BAoYad6YUELgH0qyD5pa2d1o8eu7svaeiybFFa7L4c10zB4Lw5d2CFwPH%2BFuUfWz6CClolXZl9mEdS8ZluE%2FvM67kmr6bqoFJNdVRBEJJCfza7yIGptLQKDjwB%2BIH7top9XvulS0XepyRBftE4N9Gs%2BZX3BVjdFzBhWPqXBp4itV8zQGKQXhdjlSgEpuqodYJdrbZW3GEBgPO%2FEALKMiDT7wgQTyWdZolnCpxnlQjq8A6Rs0G%2BcMxWekHzbDwf3OT7snvJ78Uyu%2BgFHuXhgPftIXkHObEarEydodeBXkNFPbWvl82fv6jvaYXlLhmiWIKdAykm1YMdlewqKDyrPSKaX5JE%2FkpMXucEc3UAD5ZZC7mKYqBiaqBR9gLoETp8MzG27HSaCMYUzjL0VTIytxdF4XCazFQbs9YNZeoY87SrWBvEUJOLA7ETe6zhxEyc1HQZrKBYduAaYbNC5oWjyvoRymnPR3CG2DmfmXIzmAhx%2FugGaH9p2atCJ5lJaMvj3BV8hK7Yrmz9Ab84x97U3N1i6MWHyjUfnBMThJgvpohaE6hJs9x3Tj8s1%2FhVNNIvJj7AYxc2i3pLbHUQftBZsI%2Bjjq06KsbQxJ6DZfBT%2FKItpEqKhdSZNP7S8ttp1oAIDFQzYEsNAaWmWCg5UYJ0qmqTkTMIIOeE6ib3pIH1thlF0SZsqFCjYauemrSdY6fYabct%2BH5bH18J8EaQNUBsA07C32oB93W7LcUqvSKuEyNgmkfO9dhR0%2FbGmy9F9T%2B0sZ4YQ0KGc6kj1BhMibXaIaaCphy0umHypoxZFs6sjFZanllqSp2CIT1UUEYi8Gm2pp16DfMOWp3VdNQPzvml9cnQw3reTOCr3tlbUvm7z2ScbwX8eqSdaDfnXZuoJzGveH3dZqSeAh80EA3xtZZ%2FARupPv0DR8WJMBlbv6qhgh946irCtnLuALvge5578n3QF%2BG12U2nJCXsTe3zcdctmIdJdt%2BAfaT%2B2Y2zzQSdIA96vlJuuD3GaDqTEWAelqrc2Vwc2nwnCCxKs5VFvdZrTB7MEqB540IXRozC2M1fU8%2FyWrqdxT1bGSFfPGFlkP0S6Xsa9YMv0x5isyPFgkHU3xgpdoONRsNW3kK6xczlMNnRee4KZFbuWB2hPg%2FY6ERj21C4jerUGHdZMQOGN1X6Cjk0kHvLVSN3mLIIureb7fXzaY65WrjPK0hdP0atH6mmDTl5%2FPIHo0m6dk6hU16zelwudh0arXP%2Fc5WrhFrucNg250ylSp86EmQY7T52eZsBVlGPLY3eR%2BTVVIG8IlaxdOElYB2Ygcns2EL9ERGv6EhrEn%2Bi4%2FJQ%2BqeYYLbapQnHxw7vTIlKsWwXYsO4BUNoF8io0Q52GEh42%2FsvyV1V0n4os1u2c%2BCp19%2BBe1H39Gx27MesAO6LPb%2FSVlypKtrv87PUUa9BCXq3FyYDd%2BFqTJDukhlQtDapgP51lYGdJecAzoWpzuf47iD2dQYP%2B3tf33GvqPn8euVMPD67sHZtjMUJ09%2B%2BbPS%2Bh9LotyuAqmpGpBnuXLEoCy8ySvODV2fltJ9IQ44%2BlWrIXbxsapuZuUVb9DokW27QiYSebgTqpZuttSyPMQVG0gWGxMf%2FGA%2FjqBYQMbLFL9uBC26mfcbEqQyqpDAusa9fgFY5ay1HtdR%2FSgXvggCWN4Asb33f0yBhWKR1SlljDzhAWd4csdxDBygi7fzm31MRwZl6A8nYGbWjK2MgpTvp733MPEKJYN%2BT7HaEk7bXO1MYm6bpZHenEa7%2BoJklLQ9%2FEXuG77%2BwVSRhNFiI%2Bvino5KnnY5aSokly9ILQKm35I1qjW0wBqeLrD%2BjSTjurJnUPHC0DNURAWyA%2F5MoXFOl%2Fom2Qf1HpfLoPxnFDfGcI%2B4jdSDLDgE5uGtHldklsbo5zily321Lk79LMeP8hza3o7e7Xl%2FHfz787Tr9Fyt3p8ZKKPVKxA3uXwtZRv5E0ftU10l2yj0%2FMZ3%2F1nuWhW2XrwwaXzpZEG0ieayLFAwSGegVGbWBAO5WCRFX6xAWwLymgrWj0LVwlno%2BZFIn4L3SIJry5ra3wCM5EEsl5c6ND7iu9mtTE7b57aDA2uH23cruk5pNFbAmsUtvufdc4lHtdjne4UN6N0BpI1jpC8r5O7gcyRkBE60qZkUyu2221Ng8nDjsAQ3bDyZbLcLIsEE6Q30dvD0u9bjYwLCxVYsQ4ESOw80WRTc75Im8Wv8yekwKd3Vri%2BWGumN3wr0JggoI4jJ5DrAXFeOIuyXL6Z4E%2Bzue6E31QimuL6LuQkrZGNBnghQWvs%2FrUZLU8mrH7ipzJAk1eqCu3S5vwNBgp5f04BwKjfpfxOjMcAZWFMFGlPoG0e8%2FqHELwWW%2FqJsjxfHeOzgdH2kCNQblXJ0mxkk2qiaNiJZvcHY4yHlYDqSsvyd5u1kSSj%2BLF%2BQBJN4uld4MB0oB2DBo%2BkOr6KORenRRZN%2FcDab2iMFoiNwq8YE7ysM9IMXH7kkoSH9Ha5Dx2BaZ%2BN0Q8Bkyy0Jd1l62pUbkSr9ovmGo5HPAXpKads0KRf0YwMkoV4QqAIr1bFPVbY9ndKgXNoKjuJCnNte4NRfwsaY4Ch1Y4ukk7SwCc6IgzVFEx7c1%2FZgEn%2BL%2BOtxmFmXohq4I3E2CqPQ3q1TG3pxw5rUsCqiIIAL7QZKU0pd5YvfNFEXt8EMWCp51QrJHK1Aj0ymEGDQSbAqilJkpVYK5U2NRNbEUjZQXf6XN%2FqpVoTxsR5eIlxaw9Ipoog4RHxICM%2FeHLXmB1JfjCU0OXp%2FGUL0gjO3I7M%2FIXhT7LFB2kvaLJoqFu7RW7ABgV3qfB3G5D1I29UrEt%2BtFSNN2jiMrPbFVXulnRD291Q1pvr0KVT%2FQorZ26WQ6wsPNrpyL1p98dsUmRWjfMrPRqzvZbr9XdZlQd8%2FTU2Ty99dB6rW10gomMrPZvV7mWImV287E3qMzMaqq8ay%2FZuQUKB6nordJuw%2BDOPVAtR2tqXVXOD%2BP9ye26HrvOajn2djOHiXCdkLWYWD1j2Z4iZ4Mw8t0lu2BwwNE3QMmkkQZlBAGL6bSHnLOZRDaIgNpFG2ojWg5QY6XsMEMu8TftGZfDerK%2BzCqOdqo%2FWd97QzvqDyhP8fz5DpgziOMLBojwXOFHRkAZnPhuW7uBedsL3eQduJ7%2FFmHS09uOXo%2BPOYu9MIjZQJDcYIr%2Fn%2FI%2FrnjUnoqjRr%2F1i0CK7IOJm6CA5C9sxy3xp85JUHZ%2B3PZ07fQ1j56DOnUOUMtiZHn%2FgQnYoWZbMNOu4bADtHBtO9Tu0w5Vd%2BfXNhQO4xIJhxUNU7loGJCi0WksTBtQ8u3wYQaU7sNU7TWLPetmDmZY0UXuJHGydwzRGaLqumjJB2ZpkD0bjA6tP21AObaDx0sm3arx0mtaoN6vDMxxtH4hz%2FB5qveyDJmiZuIgc6mydLWd0%2FTyDXqVW1srOwLUDub1Gdl%2FLpkOrtHSvUy%2FhioPwD%2BQ6g1f2GuoUuNTvTO8zDy0nMbnhxY4ONQ9WiqqjY%2F2szzTDQt8kutBHD7s0H1n%2BXQoDpdr4sRKj7%2BvUfThzNbB5JoOwpcClEo9VRUUtFBCiNVWKYCuXKVsfSlb16FzspV1mizgHTprfJiinG2SZqtkO7SRSTZRIy2WMB7lwKw%2F0PP2SLA0sksavsrKLhyNyClLEcb25hqyMPuDLliSML4bqARXCtjTwJWWd4ltozX4XSeu9eGXbbBbDb9%2BJ658sC9VRCgWHh4JQOzRV%2BeVjs%2FcYrNbRZlEnhvMl4XLk9zlpYjTGaBNx7g6wEiy2uLMdYWrA9BWd0qh95pQrvNTCowfEnEQ%2F0j%2FHSJC8MDfOaMwVBleSEeF0SK3t7giP6mghEXvK4yD4dFVV6RalIS24DmKjPgwCsl0aetUwh%2B6%2BC2cInLF%2FwE%3D"></iframe>

[数据集自定义](#数据集自定义)：根据imgs2poses.py生成sparse_points.ply和poses.npy文件，若先前没有经过colmap，则会生成`sparse\0\*.bin`[文件](#cameras文件),(cameras.bin, images.bin , points3D.bin)。然后根据gen_cameras.py文件，通过pose.npy读取第一个相机的c2w矩阵将第一个相机的单位坐标系保存为pose.ply文件，通过pose.npy和sparse_points_interest.ply文件生成preprocessed文件夹下的cameras_sphere.npz，并复制images生成image和mask文件夹下图片。
- imgs2poses.py
    - sparse_points.ply：读取points3D文件中的所有点，生成的稀疏点云文件
    - poses.npy：通过cameras.bin和images.bin文件计算出的[pose数据](#images文件)：大小num_images x 3 x 5，包括num_images x 3 x 4的c2w矩阵和num_images x 3的hwf数据
- gen_cameras.py
    - [pose.ply](#pose文件)：读取第一个相机的pose，将该相机坐标系下的原点、xyz轴单位坐标转换到世界坐标系下，然后生成点云保存为pose.ply文件
    - [cameras_sphere.npz](#两个矩阵)
        - world_mat：通过pose.npz读取pose矩阵，分解为c2w和hwf，并将c2w求逆得到w2c，将hwf转化为intrinsic相机内参矩阵，最后得到`world_mat=intrinsic @ w2c`
        - scale_mat：通过sparse_points_interest.ply文件，将其中的感兴趣区域，在世界坐标系下计算出scale_mat，**该矩阵用于将世界坐标系原点缩放并平移到感兴趣区域的中心出，使得世界坐标系下的单位圆即为感兴趣的区域**，这也是不需要mask的原因
        - image和mask：将images数据集文件夹下图片复制到preprocessed文件夹下的image下和并根据数据集图片生成同样大小的白色图片，放入mask文件夹

[数据处理](#dataset)：
- 读取cameras_sphere.npz文件、image和mask文件，获得相机的内外参矩阵intrinsics, pose，并对intrinsics求逆得到intrinsics_inv，在[生成光线](#光线生成)时用于将图片像素的坐标转换为光线在世界坐标系下的原点o和方向向量d。
- 通过o和d，生成场景中的near和far，即在每条光线上采样时，采样点的最近坐标和最远坐标

[渲染](#render)：
- 根据o、d、near和far，以及其他参数，经过MLP网络，得到颜色值、sdf对输入pts_xyz的梯度等信息，然后计算loss，最后通过反向传播不断更新网络的参数，训练出最终的4个MLP网络
- 根据训练好的MLP网络，通过一个新相机点的位置，生成一系列光线，在光线上进行采样获得点云的坐标，然后将坐标输入MLP网络，获得三维空间中每个点云的颜色、SDF和梯度等信息。
    - 颜色跟观察方向有关、SDF与方向无关、梯度与方向无关，颜色可以用来生成新视点的图片、视频，SDF可以用来根据threshold选取零水平集来生成mesh模型表面，梯度可以做法向量图。




## dataset

`self.dataset = Dataset(self.conf['dataset'])`

- 相机内外参数矩阵
- 光线的生成以及坐标变换

BlendedMVS/bmvs_bear/cameras_sphere

```
"""
in gen_cameras : 
w2c = np.linalg.inv(pose)

世界坐标系到像素坐标系转换矩阵
(4, 4) world_mats_np0 = intrinsic @ w2c =w2pixel
[[-1.0889766e+02  3.2340955e+02  6.2724188e+02 -1.6156446e+04] 
[-4.8021997e+02 -3.6971255e+02  2.8318774e+02 -8.9503633e+03]
[ 2.4123600e-01 -4.2752099e-01  8.7122399e-01 -2.1731400e+01]
[ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

将世界坐标系平移缩放到感兴趣物体的中心
(4, 4) scale_mats_np0 : sparse_points_interest中，以中心点为圆心，最远距离为半径的一个区域
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center
[[ 1.6737139  0.         0.        -2.702419 ]
[ 0.         1.6737139  0.        -1.3968586]
[ 0.         0.         1.6737139 27.347609 ]
[ 0.         0.         0.         1.       ]]
"""

P = world_mat @ scale_mat
"""
[[-1.8226353e+02  5.4129504e+02  1.0498235e+03  8.3964941e+02]
 [-8.0375085e+02 -6.1879303e+02  4.7397528e+02  6.0833594e+02]
 [ 4.0376005e-01 -7.1554786e-01  1.4581797e+00  2.0397587e+00]
 [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

[[-1.8226353e+02  5.4129504e+02  1.0498235e+03  8.3964941e+02]
 [-8.0375085e+02 -6.1879303e+02  4.7397528e+02  6.0833594e+02]
 [ 4.0376005e-01 -7.1554786e-01  1.4581797e+00  2.0397587e+00]]
 """
P = P[:3, :4]
```

将P分解为相机内参和外参矩阵，in dataset.py

```
out = cv.decomposeProjectionMatrix(P)
K = out[0] # 3x3
[[1.00980786e+03 1.61999036e-04 6.39247803e+02]
 [0.00000000e+00 1.00980774e+03 4.83591949e+02]
 [0.00000000e+00 0.00000000e+00 1.67371416e+00]]
 
R = out[1] # 3x3
[[-0.33320493  0.8066752   0.48810825]
 [-0.9114712  -0.40804535  0.05214698]
 [ 0.24123597 -0.42752096  0.87122387]]

t = out[2] # 4x1
[[-0.16280915]
 [ 0.30441687]
 [-0.69216055]
 [ 0.6338275 ]]
 
K = K / K[2, 2]
[[6.0333350e+02 9.6790143e-05 3.8193369e+02]
 [0.0000000e+00 6.0333344e+02 2.8893341e+02]
 [0.0000000e+00 0.0000000e+00 1.0000000e+00]]

intrinsics = np.eye(4)
intrinsics[:3, :3] = K # intrinsics: 4x4 为相机内参矩阵
[[6.03333496e+02 9.67901433e-05 3.81933685e+02 0.00000000e+00]
 [0.00000000e+00 6.03333435e+02 2.88933411e+02 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00]]

pose = np.eye(4, dtype=np.float32)
pose[:3, :3] = R.transpose() # 正交矩阵 其转置等于逆 w2c --> c2w
pose[:3, 3] = (t[:3] / t[3])[:, 0] # pose: 4x4 为相机外参矩阵的逆
[[-0.33320493 -0.9114712   0.24123597 -0.25686666]
 [ 0.8066752  -0.40804535 -0.42752096  0.48028347]
 [ 0.48810825  0.05214698  0.87122387 -1.092033  ]
 [ 0.          0.          0.          1.        ]]
 单位向量经过pose变换到世界坐标系后仍然为单位向量

世界坐标系下，光线的原点：
[[-0.25686666]
 [ 0.48028347]
 [-1.092033  ]
 [ 1.        ]]
```

### 光线生成

gen_random_rays_at()随机生成光线
然后生成光线，in `dataset.py/gen_random_rays_at()` by img_idx ，batch_size, 并将rays的像素坐标转换到世界坐标系下

p_pixel --> p_camera --> p_world (or rays_d)

`p_camera = intrinsics_inv @ p_pixel`:  `3x3 @ 3x1`

$\begin{bmatrix} \frac{1}{f} & 0 & -\frac{W}{2 \cdot f}  \\ 0 & \frac{1}{f} & -\frac{H}{2 \cdot f} \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{pmatrix} i \\ j \\ 1 \\ \end{pmatrix} = \begin{pmatrix} \frac{i-\frac{W}{2}}{f} \\ \frac{j-\frac{H}{2}}{f} \\ 1 \\ \end{pmatrix}$

`p_world = pose @ p_camera`:  `3x3 @ 3x1`

$\begin{bmatrix} r_{11}&r_{12}&r_{13}\\ r_{21}&r_{22}&r_{23}\\ r_{31}&r_{32}&r_{33} \end{bmatrix} \begin{pmatrix} x_{c} \\ y_{c} \\ z_{c} \\ \end{pmatrix} = \begin{pmatrix} x_{w} \\ y_{w} \\ z_{w} \\ \end{pmatrix} = rays_d$

`rays_o = pose[:3, 3]` $= \begin{bmatrix} t_{x} \\ t_{y} \\ t_{z} \end{bmatrix}$，为相机坐标系原点在世界坐标系下位置

$pose = \begin{bmatrix}r_{11}&r_{12}&r_{13}&t_x\\ r_{21}&r_{22}&r_{23}&t_y\\ r_{31}&r_{32}&r_{33}&t_z\\ 0&0&0&1\end{bmatrix}$

```
def gen_random_rays_at(self, img_idx, batch_size):
    """
    Generate random rays at world space from one camera.
    """
    pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]) 
    pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
    color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
    mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
    # p : 像素坐标系下的坐标
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
    # 将p转换到相机坐标系下
    # matmul : [1, 3, 3] x [batch_size, 3, 1] -> [batch_size, 3, 1] -> [batch_size, 3]
    p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
    # rays_v ：将p归一化
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
    # rays_v ：将p转换到世界坐标系下
    # matmul : [1, 3, 3] x [batch_size, 3, 1] -> [batch_size, 3, 1] -> [batch_size, 3]
    rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
    # [1,3].expand([batch_size, 3])
    rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
    return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10
```

### 计算near和far(from o,d)
根据rays_o 和rays_d 计算出near和far两个平面

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230803193755.png)


```
def near_far_from_sphere(self, rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    # rays_o 在 rays_d 方向上的投影 / rays_d 在 rays_d 方向上的投影
    near = mid - 1.0
    far = mid + 1.0
    return near, far
```

### box的min和max(to生成mesh模型)

```
'''
(4, 4) scale_mats_np0
[[ 1.6737139  0.         0.        -2.702419 ]
[ 0.         1.6737139  0.        -1.3968586]
[ 0.         0.         1.6737139 27.347609 ]
[ 0.         0.         0.         1.       ]]
'''
object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
# Object scale mat: region of interest to **extract mesh**
object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0'] # 4x4

# object_bbox_? > object_scale_mat缩放+平移 > scale_mat缩放+平移
object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None] # 4x1
object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None] # 4x1
self.object_bbox_min = object_bbox_min[:3, 0] # 3
self.object_bbox_max = object_bbox_max[:3, 0] # 3
如果
render_cameras_name = cameras_sphere.npz
object_cameras_name = cameras_sphere.npz
两文件相同，则 
np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat = 
[[ 1.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00]
 [ 0.0000000e+00  1.0000000e+00  0.0000000e+00 -5.9604645e-08]
 [ 0.0000000e+00  0.0000000e+00  1.0000000e+00  0.0000000e+00]
 [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
object_bbox_min , object_bbox_max 只平移，不缩放
```



## 神经网络结构Network

```
# Networks
params_to_train = []
self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device) # 创建一个NeRF网络
self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device) # 创建一个SDF网络
self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
params_to_train += list(self.nerf_outside.parameters())
params_to_train += list(self.sdf_network.parameters())
params_to_train += list(self.deviation_network.parameters())
params_to_train += list(self.color_network.parameters())

self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

self.renderer = NeuSRenderer(self.nerf_outside,
                             self.sdf_network,
                             self.deviation_network,
                             self.color_network,
                             **self.conf['model.neus_renderer'])
```

Neus中共构建了4个network：
- NeRF：训练物体outside即背景的颜色
- SDFNetwork：训练点云中的sdf值
- RenderingNetwork：训练点云的RGB
- SingleVarianceNetwork：训练一个单变量invs，用于计算$cdf = sigmoid(estimated.sdf  \cdot inv.s)$

### NeRF

同NeRF网络
![Pasted image 20221206180113.png|600](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020221206180113.png)
- 84-->256-->256-->256-->256-->256+84-->256-->256-->256+27-->128-->3
- 84-->256-->256-->256-->256-->256+84-->256-->256-->256-->1

### SDFNetwork

激活函数 $\text{Softplus}(x) = \frac{\log(1 + e^{\beta x})}{\beta}$

网络结构：
![SDFNetwork](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/SDFNetwork_modify.png)
- 39-->256-->256-->256-->217-->256-->256-->256-->256-->257
input: pts, 采样点的三维坐标 batch_size * n_samples x 3
output: 257个数 batch_size * n_samples x 257

`sdf(pts) = output[:, :1]`:  batch_size * n_samples x 1，采样点的sdf值

### RenderingNetwork

![RenderingNetwork.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/RenderingNetwork.png)

input: rendering_input :`[batch_size * n_samples ,  3 + 27 + 3+ 256 = 289]`
`rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)`
- pts: batch_size * n_samples, 3
- gradients: batch_size * n_samples, 3
- dirs: batch_size * n_samples, 3
    - 位置编码 to view_dirs: batch_size * n_samples , 27
- feature_vector: batch_size * n_samples, 256

output: sampled_color采样点的RGB颜色 batch_size * n_samples , 3

### SingleVarianceNetwork

```
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        # variance 模型可以跟踪和优化这个参数，使其在训练过程中进行更新
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        # torch.zeros([1, 3])
        # 大小为 [len(x), 1] 的张量，每个元素都是 exp(variance * 10.0)
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

in Runner:
self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
```

render中
`inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) `

## render

input: 
- rays_o, 
- rays_d, 单位向量
- near, far : batch_sizex1,batch_sizex1
- background_rgb=background_rgb,
- cos_anneal_ratio=self.get_cos_anneal_ratio()

```
image_perm = self.get_image_perm()
res_step = self.end_iter - self.iter_step

for iter_i in tqdm(range(res_step)):
    data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
    # data : [batch_size, 10] : [rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]]
    rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
    
    near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
    
    background_rgb = None
    if self.use_white_bkgd:
        background_rgb = torch.ones([1, 3])

    render_out = self.renderer.render(rays_o, rays_d, near, far,
                                      background_rgb=background_rgb,
                                      cos_anneal_ratio=self.get_cos_anneal_ratio())
```

output: render_out字典
- color_fine: render出来图片的RGB颜色值
- s_val: $= \sum_{i}^{n.samples}(\frac{1.0}{invs_{i}})$
    - inv_s: 一个可以更新的变量 $1 \times e^{10.0 \cdot var}$ ，并将其限制在$1 \times 10^{-6}$ ~ $1 \times 10^{6}$之间
    - `ret_fine['s_val'] = 1.0 / inv_s` # batch_size * n_samples, 1
    - `s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)` # batch_size, 1
- cdf_fine: $pre.cdf = {\Phi_s(f(\mathbf{p}(t_i)))}$
    - batch_size, n_samples
- weight_sum: 一条光线上的权重之和(包括背景outside)
    - batch_size, 1
    - `weights_sum = weights.sum(dim=-1, keepdim=True)`
- weight_max: 一条光线上权重的最大值
    - batch_size, 1
    - `torch.max(weights, dim=-1, keepdim=True)[0]`
- gradients: 梯度,sdf对输入pts_xyz的梯度，与法向量的计算有关
    - batch_size, n_samples, 3
- weights: 权重，每个采样点
    - batch_size, n_samples or batch_size, n_samples + n_outside
- gradient_error: Eikonal损失值$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$ 只计算在relax半径为1.2的圆内的采样点sdf的梯度
    - $\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2} = \sqrt{gx^{2}+gy^{2}+gz^{2}}$
- inside_sphere: 采样点是否在单位圆空间内
    - batch_size, n_samples

```
{
    'color_fine': color_fine, # batch_size, 3
    's_val': s_val, # batch_size, 1
    'cdf_fine': ret_fine['cdf'], # batch_size, n_samples
    'weight_sum': weights_sum, # batch_size, 1
    'weight_max': torch.max(weights, dim=-1, keepdim=True)[0], # batch_size, 1
    'gradients': gradients, # batch_size, n_samples, 3
    'weights': weights, # batch_size, n_samples or batch_size, n_samples + n_outside
    'gradient_error': ret_fine['gradient_error'], # 1
    'inside_sphere': ret_fine['inside_sphere'] # batch_size, n_samples
}
```

```
ret_fine = self.render_core(rays_o,
                            rays_d,
                            z_vals,
                            sample_dist,
                            self.sdf_network,
                            self.deviation_network,
                            self.color_network,
                            background_rgb=background_rgb,
                            background_alpha=background_alpha,
                            background_sampled_color=background_sampled_color,
                            cos_anneal_ratio=cos_anneal_ratio)
                            
# ret_fine:
    # 'color': color, # batch_size, 3
    # 'sdf': sdf, # batch_size * n_samples, 1
    # 'dists': dists, # batch_size, n_samples
    # 'gradients': gradients.reshape(batch_size, n_samples, 3),
    # 's_val': 1.0 / inv_s, # batch_size * n_samples, 1
    # 'mid_z_vals': mid_z_vals, # batch_size, n_samples
    # 'weights': weights, # batch_size, n_samples or batch_size, n_samples + n_outside
    # 'cdf': c.reshape(batch_size, n_samples), # batch_size, n_samples
    # 'gradient_error': gradient_error, # 1
    # 'inside_sphere': inside_sphere # batch_size, n_samples
color_fine = ret_fine['color']
weights = ret_fine['weights']
weights_sum = weights.sum(dim=-1, keepdim=True)
gradients = ret_fine['gradients']
s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True) # [batch_size, 1]
```

function:

```
render:

batch_size = len(rays_o)
sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere 
z_vals = torch.linspace(0.0, 1.0, self.n_samples) # [n_samples]
z_vals = near + (far - near) * z_vals[None, :]  # [batch_size, n_samples]
拍照物体的采样点z方向坐标
```

```
物体外的z坐标(背景)
z_vals_outside = None
if self.n_outside > 0:
    z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside) # [n_outside]

n_samples = self.n_samples
perturb = self.perturb
```

```
添加扰动：
if perturb_overwrite >= 0:
    perturb = perturb_overwrite
if perturb > 0:
    t_rand = (torch.rand([batch_size, 1]) - 0.5) # [batch_size, 1]
    z_vals = z_vals + t_rand * 2.0 / self.n_samples # [batch_size, n_samples]

    if self.n_outside > 0:
        mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1]) # [n_outside - 1]
        upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)     # [n_outside]
        lower = torch.cat([z_vals_outside[..., :1], mids], -1)      # [n_outside]
        t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]]) # [batch_size, n_outside]
        z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
        # Z_vals_outside:  1Xn_outside + 1Xn_outside * batch_sizeXn_outside = batch_sizeXn_outside

if self.n_outside > 0:
    z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples # [batch_size, n_outside]
    # filp: 将tensor的维度进行翻转，如[1,2,3] -> [3,2,1] ，倒序排列

背景outside:
background_alpha = None
background_sampled_color = None
```



### get_cos_anneal_ratio

output: 
- 数1或者比一小的数$\frac{iterstep}{anneal}, anneal=50000$
- or 1 when anneal_end = 0

### 精采样n_importance

if self.n_importance > 0: 精采样

```
with torch.no_grad(): # 不需要计算梯度
    # pts : [batch_size, 1, 3] + [batch_size, 1, 3] * [batch_size, n_samples, 1] = [batch_size, n_samples, 3]
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # [batch_size, n_samples, 3]
    sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
    # pts.reshape(-1, 3) : [batch_size * n_samples, 3]
    # sdf : [batch_size * n_samples , 1] -> [batch_size, n_samples]

    for i in range(self.up_sample_steps):
        # [batch_size, n_importance // up_sample_steps] per step
        new_z_vals = self.up_sample(rays_o,
                                    rays_d,
                                    z_vals,
                                    sdf,
                                    self.n_importance // self.up_sample_steps,
                                    64 * 2**i)
        # # [batch_size, n_samples + n_importance // up_sample_steps], [batch_size, n_samples + n_importance // up_sample_steps]
        z_vals, sdf = self.cat_z_vals(rays_o,
                                    rays_d,
                                    z_vals,
                                    new_z_vals,
                                    sdf,
                                    last=(i + 1 == self.up_sample_steps))
    # new_z_vals : [batch_size, n_importance]
    # z_vals : [batch_size, n_samples + n_importance]

n_samples = self.n_samples + self.n_importance
```

#### up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):

input:
- rays_o,
- rays_d,
- z_vals, batch_size X n_samples
- sdf, batch_size X n_samples
- self.n_importance // self.up_sample_steps, 每步处理$\frac{importance}{sampls.steps}$
- `64 * 2**i` , $64  \cdot  2^{i}$

output:
- new_z_vals: batch_size X n_importance // up_sample_steps * steps_i

function:
- pts: batch_size,n_samples,3
- radius: pts的2-范数norm(ord=2)
    - batch_size, n_samples
- inside_sphere: `inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)`
    - point是否在单位圆的空间内
    - batch_size, n_samples - 1
- prev_sdf, next_sdf: 光线上sdf的前后 `prev_sdf[1] = next_sdf[0] = sdf[1]` 
    -  batch_size, n_samples - 1
- prev_z_vals, next_z_vals:  光线上z坐标的前后 `prev_z_vals[1] = next_z_vals[0] = z_vals[1]` 
    - batch_size, n_samples - 1
- mid.sdf:  $mid.sdf = \frac{prev.sdf + next.sdf}{2} = \frac{f(p_{i})+f(p_{i+1})}{2}$
    - batch_size, n_samples - 1
- cos_val: $cos.val = \frac{next.sdf - prev.sdf}{next.z.vals - prev.z.vals + 1e-5} = \frac{f(p_{i})-f(p_{i+1})}{z_{i}-z_{i+1}}$
    - batch_size, n_samples - 1 

- prev_cos_val： 将cos_val堆叠，且最后一个删除，第一个插入0 `prev_cos_val[0] = 0, prev_cos_val[1] = cos_val[0]`
    - batch_size, n_samples - 1
- cos_val: stack prev_cos_val and cos_val
    - batch_size, n_samples - 1, 2 
- cos_val: 在prev_cos_val和cos_val之间选择最小值，这一步的目的是当发生一条光线穿过物体两次时，具有更好的鲁棒性
    - batch_size, n_samples - 1
- cos_val: 将cos_val限制在$-1 \times 10^{3}$和0之间，并将在单位圆空间外的值置False `cos_val.clip(-1e3, 0.0) * inside_sphere`
    - batch_size, n_samples - 1
- dist:  两点之间的距离 $dist = next.z.vals- prev.z.vals= z_{i+1}-z_{i}$
    - batch_size, n_samples - 1 

batch_size, n_samples - 1: 
- prev_esti_sdf: $\frac{mid.sdf - cos.val * dist}{2} \approx f(p_{i})$
- next_esti_sdf: $\frac{mid.sdf + cos.val * dist}{2} \approx f(p_{i+1})$
- prev_cdf: $prev.cdf = sigmoid(prev.esti.sdf \times inv.s) = sigmoid(\approx f(p_{i})\times 64  \cdot  2^{i})$
- next_cdf: $next.cdf = sigmoid(next.esti.sdf \times inv.s) = sigmoid(\approx f(p_{i+1})\times 64  \cdot  2^{i})$
- alpha: $\alpha = \frac{prev.cdf - next.cdf + 1 \times 10^{-5}}{prev.cdf + 1 \times 10^{-5}}$ is  $\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i))))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right).$
- weights: $w_{i} = \alpha_{i} \cdot T_{i} =\alpha_{i} \cdot \prod_{j=1}^{i-1}(1-\alpha_j)$
    - in code : `weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]`

`z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()`

##### sample_pdf(z_vals, weights, n_importance, det=True)

like NeRF

input:
- z_vals, batch_size X n_samples
- weights, batch_size, n_samples - 1
- n_importance, 
- det=True

output:
- z_samples, batch_size X n_importance 经过逆变换采样得到的采样点的z坐标值

#### cat_z_vals(rays_o,rays_d,z_vals,new_z_vals,sdf,last=(i + 1 == self.up_sample_steps))

将原来的z_vals和经过逆变换采样得到的new_z_vals一起cat起来

input:
- rays_o,
- rays_d,
- z_vals, batch_size X n_samples
- new_z_vals, `batch_size X n_importance // up_sample_steps * steps_i`
- sdf, batch_size X n_samples
- last=(i + 1 == self.up_sample_steps): true(last step) or false

output:
- z_vals, `batch_size X n_samples + n_importance // up_sample_steps * steps_i`
- sdf,  `batch_size X n_samples + n_importance // up_sample_steps * steps_i` when not last

**last:** 
```
z_vals : batch_size X n_samples + n_importance 
n_samples = self.n_samples + self.n_important
```

**then :**
- z_vals : batch_size X n_samples

### render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

```
in render()
# Background model
if self.n_outside > 0:
    z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1) # [batch_size, n_samples + n_outside]
    z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
    ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

    background_sampled_color = ret_outside['sampled_color']
    background_alpha = ret_outside['alpha']
```

input: 
- rays_o, `[batch_size,  3]`
- rays_d, `[batch_size,  3]`
- z_vals_feed, `batch_size, n_samples + n_outside` ,实际上此处为`[batch_size, n_samples + n_outside +n_importance]`
- sample_dist, $sample.dist = \frac{2.0}{n.samples}$
- self.nerf, NeRF神经网络，使用nerf渲染函数进行color的计算
    - 如果使用了白色背景，color还需累加白背景
        - `background_rgb = torch.ones([1, 3])`
        - `color = color + background_rgb * (1.0 - weights_sum)`

output: ret_outside字典
```
{
    'color': color, # batch_size, 3
    'sampled_color': sampled_color, # batch_size, n_samples + n_outside, 3
    'alpha': alpha, # batch_size, n_samples + n_outside
    'weights': weights, # batch_size, n_samples + n_outside
}
```

function: like NeRF
- dis_to_center: 坐标的2范数，并限制在$1$ ~ $1 \times 10^{10}$
    - batch_size, n_samples, 1 
- pts: `torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)`
    - batch_size, n_samples, 4
    - 归一化pts, $\frac{x}{\sqrt{x^{2}+y^{2}+z^{2}}},\frac{y}{\sqrt{x^{2}+y^{2}+z^{2}}},\frac{z}{\sqrt{x^{2}+y^{2}+z^{2}}},\frac{1}{\sqrt{x^{2}+y^{2}+z^{2}}}$

### render_core()

```
render continue
    background_sampled_color = ret_outside['sampled_color']
    background_alpha = ret_outside['alpha']

# Render core
ret_fine = self.render_core(rays_o,
                            rays_d,
                            z_vals,
                            sample_dist,
                            self.sdf_network,
                            self.deviation_network,
                            self.color_network,
                            background_rgb=background_rgb,
                            background_alpha=background_alpha,
                            background_sampled_color=background_sampled_color,
                            cos_anneal_ratio=cos_anneal_ratio)
```

input:
- rays_o, `[batch_size,  3]`
- rays_d, `[batch_size,  3]`
- z_vals, `batch_size, n_samples` ,实际上为`batch_size, n_samples + n_importance` 
- sample_dist, $sample.dist = \frac{2.0}{n.samples}$
- self.sdf_network, sdf神经网络
- self.deviation_network, inv_s参数神经网络
- self.color_network, 采样点color神经网络
- background_rgb=background_rgb, `batch_size, 3`
- background_alpha=background_alpha, `batch_size, n_samples + n_outside`
- background_sampled_color=background_sampled_color, `batch_size, n_samples + n_outside, 3`
- cos_anneal_ratio=cos_anneal_ratio ,数1或者比一小的数$\frac{iterstep}{anneal}, anneal=50000$

output: ret_fine字典
```
{
    'color': color, # batch_size, 3
    'sdf': sdf, # batch_size * n_samples, 1
    'dists': dists, # batch_size, n_samples
    'gradients': gradients.reshape(batch_size, n_samples, 3),
    's_val': 1.0 / inv_s, # batch_size * n_samples, 1
    'mid_z_vals': mid_z_vals, # batch_size, n_samples
    'weights': weights, # batch_size, n_samples or batch_size, n_samples + n_outside
    'cdf': c.reshape(batch_size, n_samples), # batch_size, n_samples
    'gradient_error': gradient_error, # 1
    'inside_sphere': inside_sphere # batch_size, n_samples
}
```

function:
- dists: 采样点间距离,$dists = z_{i+1} - z_{i}$
    - batch_size, n_samples - 1 
- dists: 最后一行添加固定的粗采样点间距: $sample.dist = \frac{2.0}{n.samples}$
    - batch_size, n_samples
- mid_z_vals:  $mid = z_{i} + \frac{dist_{i}}{2}$
    - batch_size, n_samples
- pts:  $pts = \vec o + \vec d \cdot mid$
    - batch_size, n_samples, 3 
- dirs: 方向向量扩展得到 `rays_d[:, None, :].expand(batch_size, n_samples, 3)`
    - batch_size, n_samples, 3 
- pts: reshape to batch_size * n_samples, 3 
- dirs: reshape to batch_size * n_samples, 3 
- sdf_nn_output:  =  sdf_network(pts)
    - batch_size * n_samples, 257
- sdf: `sdf = sdf_nn_output[:, :1]`
    - batch_size * n_samples, 1
- feature_vector:  `feature_vector = sdf_nn_output[:, 1:]`
    - batch_size * n_samples, 256
- gradients:  梯度,sdf对输入pts_xyz的梯度，与法向量有关
    - batch_size * n_samples, 3

```
def gradient(self, x):
    # x : [batch_size * n_samples , 3]
    x.requires_grad_(True) 
    y = self.sdf(x) # y : [batch_size * n_samples , 1]
    d_output = torch.ones_like(y, requires_grad=False, device=y.device) # d_output : [batch_size * n_samples , 1]
    # torch.autograd.grad : 计算梯度,返回一个元组，元组中的每个元素都是输入的梯度
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients.unsqueeze(1) # unsqueeze(1) : 在第1维增加一个维度
    # return : [batch_size * n_samples , 1 , 3]
```

- sampled_color: batch_size, n_samples, 3
    - `color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)`
- inv_s: `deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) `
    - 一个可以更新的变量 $1 \times e^{10.0 \cdot var}$ ，并将其限制在$1 \times 10^{-6}$ ~ $1 \times 10^{6}$之间
    - 这个变量是用于sigmoid函数的输入，使其乘以s

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630181909.png)


```
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        # variance 模型可以跟踪和优化这个参数，使其在训练过程中进行更新
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        # torch.zeros([1, 3])
        # 大小为 [len(x), 1] 的张量，每个元素都是 exp(variance * 10.0)
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
```


>[4.1. 多层感知机 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/mlp.html)


![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630132744.png)


<div style="display:flex; justify-content:space-between;"> <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630174602.png" alt="Image 1" style="width:50%;"><div style="width:10px;"></div> <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630174609.png" alt="Image 2" style="width:50%;"> </div>

可以看出sigmoid函数的导数是一个偶函数，即$\phi(-x) = \phi(x)$

- inv_s: expand a num to `batch_size * n_samples, 1 `
- true_cos: $true.cos = \frac{dx \cdot gx + dy \cdot gy + dz \cdot gz}{\sqrt{dx^{2}+dy^{2}+dz^{2}} \cdot \sqrt{gx^{2}+gy^{2}+gz^{2}}}$ 为sdf梯度方向，即物体表面的法线方向向量$\vec g$与光线方向向量$\vec d$的夹角
    - batch_size * n_samples, 1 
    - `true_cos = (dirs * gradients).sum(-1, keepdim=True)`

{% note info %}
why `true_cos = (dirs * gradients).sum(-1, keepdim=True)`
- cdf对t的导数：$\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}(f(\mathbf{p}(t)))= \nabla f(\mathbf{p}(t))\cdot\mathbf{v} \cdot \phi_s(f(\mathbf{p}(t)))$
- sdf对t的导数：$\frac{\mathrm{d}f(\mathbf{p}(t))}{\mathrm{d}t}= \nabla f(\mathbf{p}(t))\cdot\mathbf{v}$，即为true_cos
{% endnote %}

- iter_cos: $= -[relu(\frac{-true.cos+1}{2}) \cdot (1.0 - cos.anneal.ratio)+  relu(-true.cos) \cdot cos.anneal.ratio]$
    - batch_size * n_samples, 1 
    - iter_cos 总是非正数
    - cos_anneal_ratio 数1或者比一小的数$\frac{iterstep}{anneal}, anneal=50000$ in womask cos_anneal_ratio is from 0 to 1, and always 1 after anneal steps
        - anneal = 0 in wmask, then cos_anneal_ratio is always 1

batch_size * n_samples, 1: 
- estimated_next_sdf: $est.next.sdf = sdf + iter.cos \times dist \times 0.5$
    - `estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5`
- estimated_prev_sdf: $est.prev.sdf = sdf - iter.cos \times dist \times 0.5$
    - `estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5`
- prev_cdf: $prev.cdf = sigmoid(est.prev.sdf \cdot inv.s)$
    - `prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)`
- next_cdf: $next.cdf = sigmoid(est.next.sdf \cdot inv.s)$
    - `next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)`
- `p = prev_cdf - next_cdf ,  c = prev_cdf`

- alpha: $\alpha = \frac{p + 10^{-5}}{c + 10^{-5}} = \frac{prev.cdf - next.cdf}{prev.cdf}$ and in (0.0,1.0)
    -  $$\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i))))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right).$$
    - batch_size, n_samples
    - `alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)`
- pts_norm: $\sqrt{x^{2}+y^{2}+z^{2}}$
    - batch_size, n_samples
    - `pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)`
- inside_sphere, 在单位圆内的点置位 True，在外的为False
    - batch_size, n_samples
    - `inside_sphere = (pts_norm < 1.0).float().detach()`
- relax_inside_sphere，更放松一点的限制：在半径1.2的圆内的点
    - batch_size, n_samples
    - `relax_inside_sphere = (pts_norm < 1.2).float().detach()`

if background_alpha 不是 None，计算过背景的alpha值，将背景与物体前景的alpha和采样点颜色值cat起来

```
if background_alpha is not None:
    alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere) # batch_size, n_samples
    alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1) # batch_size, n_samples + n_outside
    sampled_color = sampled_color * inside_sphere[:, :, None] +\
                    background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None] # batch_size, n_samples, 3
    sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1) # batch_size, n_samples + n_outside, 3
```

- weights，计算每个采样点的权重 $w_{i} = \alpha_{i} \cdot T_{i} =\alpha_{i} \cdot \prod_{j=1}^{i-1}(1-\alpha_j)$
    - batch_size, n_samples **or** batch_size, n_samples + n_outside
    - `weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]`
- weights_sum：权重的和，方便前景颜色与背景的颜色进行累加
    - batch_size, 1
    - `weights_sum = weights.sum(dim=-1, keepdim=True)`
- color：$\hat{C}=\sum_{i=1}^n T_i\alpha_i c_i,$
    - batch_size, 3
    - `color = (sampled_color * weights[:, :, None]).sum(dim=1)

累加背景的颜色值

```
if background_rgb is not None:    # Fixed background, usually black
    color = color + background_rgb * (1.0 - weights_sum) # batch_size, 3
```

计算loss
$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$ 只计算在relax半径为1.2的圆内的采样点sdf的梯度

$\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2} = \sqrt{gx^{2}+gy^{2}+gz^{2}}$

```
# Eikonal loss
gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                    dim=-1) - 1.0) ** 2
# gradient_error : batch_size, n_samples

gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
# gradient_error : 1
```


## render后

### get loss

- color_fine_loss: $\mathcal{L}_{color}=\frac{1}{m}\sum_k\mathcal{R}(\hat{C}_k,C_k).$
- eikonal_loss: $\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$
- mask_loss: $\mathcal{L}_{mask}=\mathrm{BCE}(M_k,\hat{O}_k)$

total loss: $\mathcal L=\mathcal L_{color}+\lambda\mathcal L_{reg}+\beta\mathcal L_{mask}.$
- igr_weight = 0.1
- mask_weight = 0.1 or 0.0 if womask

```
color_error = (color_fine - true_rgb) * mask
color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

eikonal_loss = gradient_error

mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

loss = color_fine_loss +\
       eikonal_loss * self.igr_weight +\
       mask_loss * self.mask_weight
```

### backward

```
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.iter_step += 1
```

### log(tensorboard.scalar)


```
from torch.utils.tensorboard import SummaryWriter
self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
# if in autodl server , use: 
# self.writer = SummaryWriter(log_dir=os.path.join('/root/tf-logs'))

self.writer.add_scalar('Loss/loss', loss, self.iter_step)
self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
```

### other per step

```
if self.iter_step % self.report_freq == 0:
    print(self.base_exp_dir)
    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

if self.iter_step % self.save_freq == 0:
    self.save_checkpoint()

if self.iter_step % self.val_freq == 0:
    self.validate_image()

# 每经过一定的迭代次数，就验证一次mesh， 5000步val一次，默认mesh的resolution=64
if self.iter_step % self.val_mesh_freq == 0:
    self.validate_mesh()

self.update_learning_rate()

if self.iter_step % len(image_perm) == 0:
    image_perm = self.get_image_perm() # 重新随机一下image
```

#### validate_image

将图片缩小resolution_level倍进行光线生成，然后分批次进行渲染，每批大小为batch_size

```
def validate_image(self, idx=-1, resolution_level=-1):
    if idx < 0:
        idx = np.random.randint(self.dataset.n_images)

    print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

    if resolution_level < 0:
        resolution_level = self.validate_resolution_level
    rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
    H, W, _ = rays_o.shape
    rays_o = rays_o.reshape(-1, 3).split(self.batch_size) # H*W / batch_size 个元组，每个元组中有batch_size个ray: (batch_size, 3)
    rays_d = rays_d.reshape(-1, 3).split(self.batch_size) 
```

最终得到该图片每个像素的颜色值out_rgb_fine，以及inside_sphere内的法向量值out_normal_fine

```
for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
    # rays_o_batch: (batch_size, 3) rays_d_batch: (batch_size, 3)
    near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
    background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

    render_out = self.renderer.render(rays_o_batch,
                                      rays_d_batch,
                                      near,
                                      far,
                                      cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                      background_rgb=background_rgb)

    def feasible(key): return (key in render_out) and (render_out[key] is not None)
    
    if feasible('color_fine'):
        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
    if feasible('gradients') and feasible('weights'):
        n_samples = self.renderer.n_samples + self.renderer.n_importance
        # (batch_size, n_samples, 3) * (batch_size, n_samples, 1) -> (batch_size, n_samples, 3)
        normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None] 
        if feasible('inside_sphere'):
            normals = normals * render_out['inside_sphere'][..., None]
        normals = normals.sum(dim=1).detach().cpu().numpy()
        out_normal_fine.append(normals)
    del render_out
```

然后进行图片的拼接和保存

<div style="display:flex; justify-content:space-between;"> <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/00282500_0_22.png" alt="Image 1" style="width:10%;"><div style="width:10px;"></div> <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/00282500_0_22%20(1).png" alt="Image 2" style="width:20%;"> </div>

```
img_fine = None
if len(out_rgb_fine) > 0:
    img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

normal_img = None
if len(out_normal_fine) > 0:
    normal_img = np.concatenate(out_normal_fine, axis=0)
    # pose: c2w 
    # rot: w2c
    rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
    normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                  .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

for i in range(img_fine.shape[-1]):  # img_fine.shape[-1] = 1
    if len(out_rgb_fine) > 0:
        cv.imwrite(os.path.join(self.base_exp_dir,
                                'validations_fine',
                                '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                   np.concatenate([img_fine[..., i],
                                   self.dataset.image_at(idx, resolution_level=resolution_level)]))
    if len(out_normal_fine) > 0:
        cv.imwrite(os.path.join(self.base_exp_dir,
                                'normals',
                                '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                   normal_img[..., i])
```

#### validate_mesh生成mesh模型

根据一个$resolution^3$ 的sdf场，将阈值为0的点使用marching_cubes方法生成vertices和triangles，然后生成mesh的ply文件

##### extract_geometry

**extract_fields**
input:
- bound_min : 3 ; bound_max : 3 ; resolution : 64 
- query_func : pts -> sdf

output: u  
u : resolution x resolution x resolution, 为box 中每个点的sdf值

**extract_geometry**

根据体积数据和阈值重建出表面

>[pmneila/PyMCubes: Marching cubes (and related tools) for Python (github.com)](https://github.com/pmneila/PyMCubes)

input:
- bound_min, bound_max, resolution, 
- threshold, 用于`vertices, triangles = mcubes.marching_cubes(u, threshold)`，在等threshold面上，生成mesh的v和t
- query_func，根据位置pts利用network计算出sdf
    - query_func=lambda pts: -self.sdf_network.sdf(pts)
output:
- vertices：三角形网格点
    - N_v , 3: 3为点的三维坐标
- triangles：三角形网格
    -  N_t , 3: 3为三角形网格顶点的索引index

![images.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630210635.png)

根据v和t，`mesh = trimesh.Trimesh(vertices, triangles)`生成mesh，并导出ply：
```
mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
```


## 数据集自定义

### custom_data流程图
<iframe frameborder="0" style="width:100%;height:833px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=custom_data.drawio#R7Vxdc5s4FP01nmw7Ew%2BID5vHOIm7D7udzmRn2j55FCPbpICokB3TX78SiE8pMc0aQ9bOQ4IuEhb36BxdXSkeGbfB%2FhOB0eZv7CJ%2FBDR3PzLuRgDopq2zP9ySZBZ7AjLDmniuqFQaHrxfSBg1Yd16LoprFSnGPvWiunGJwxAtac0GCcHP9Wor7Nc%2FNYJrJBkeltCXrV89l27EWzhaaf8TeeuN%2BGRnKm4EMK8rDPEGuvi5YjLuR8YtwZhmV8H%2BFvncd7lbsnbzF%2B4W%2FSIopG0aOFMUff30z%2FWX%2B8l8PwnJLJnvrgUWMU3y90Uue31RxIRu8BqH0L8vrTOCt6GL%2BFM1Virr%2FIVxxIw6Mz4hShOBJdxSzEwbGvjiLtp79BtvPrZE6Xvlzt1ePDktJHkhpCSpNOLF79V7ZbO0lLfL3o%2B%2F1ItuE6YYb8lS1Hp6Mr651F2Es59m8FlPrPvJ7joffZCsEX2lHijAZaRAOECsP6wdQT6k3q7eDyhG57qoVyLILgSIakBf6%2BQO%2BlvxSV6wjkGEYxSPo0SCuwST%2B%2F9541H0EMHUD8%2BM0HXgVjikAlXdLpyr8OUOEYr2FZPsDnHXsrIWQh10W7DlueSanjNoU%2BGZpXXkQOv%2FzojWIB4kBGhJCFNJwr4YAiSGrFG4SAnyx4dX%2BKENgR%2FA7Jsf%2BbzcD0H0Cj1KshwiSI0eJVs6J4jZkiDOsAjSK8LvSwLbIjwZFMKmJIErz2dBNmvGfzEfxzQthdwHzC8xYp8jDwuywcHjNu5FGHMdEsJoAFkYizpVYZx2JYy25NSEu7ThMvZ2tO6bmBL8A91iHxNmCXHI6cTw8Bsm6HvrkBWXzGeI2WfcVx5bqNyIG4HnuikXVWDU%2BdkBHmZ9ojIVeJgKOEBXcIBLINdaxSYtVQwMK5KbSIzzMXQXS%2BwHMFq4kMIBBnTGZNxgylQR0uVh30lCuqnkxxCfj3AVEpTPJIbVr3I5EhxkG4pRPcABDSbWQf%2BdeIkiL%2FEeYYxcj4zAbQDpcrOgSYTOaIhPGxDpCoiMU45xfXqZnVvnHY2W07NuD2p6zvtdoWEmYsy2QpBuCVowxhG4pIxLwPY5%2BR751ZpfFXWrjH2lUhQx5h1VHHNW%2B2hFjxQkGw0igiIaqK5bNAUVbasrmCwJJv47uzgPfTScBixW34sXXXL%2BKeXxTUm23uTRbimPR8%2FBpE1vCIFJpUKEvZDGlSd%2F4YZK3kJvhP9i4VwOl%2ByJ5eApuvYfOC4nJ7iv6smd%2BcfxI8%2F42DDgrAsf4ygFpqm5I2t2tYQBIjC%2BYgEVe8qVF8A1KkqZC4y7q5Eli8jQFFmX1mOqTJJxyvUYeCV%2BPRdJ1p3Dq2RbAUp3kixHM%2BmeUTbmIzbeXXGJSHBOONVXz5aCPcpsRmdAGXKe%2BrK0OJDQO5z5G9gWlbyDEcMdGu4urj0GdT2zHAVPyrnoNPOMHBWUeRL%2Byhd9A3oju2Up5iHlPlN3E5Gc9s5jNNfb1WCxf2756bIZR%2BdaOPqG1UjDp%2BJuEdeJp7CYMMxtaB8xMSwCxUUW2o0jP8lrs3eoNqiY097UrcfvIKd9M2TNDjuF%2FLSTNrqfj2a3I%2BdmIP3NOMUbqcPsF7pXtYXbYJEF3OkwBZoh%2Flq%2F%2BY7nw2Fj3KCxbahorBcqfRL9NcAlUmkdqTgtIxVDH1akIu%2FmeCtWXqVZz%2BLIRYi5yH5O6VRIVbnkdtEKbv2iihTeDOcchq2Ia057DiMfABWPB17orZIBxoW2NraBU%2Fmpq9RUUwSJ2rQQs6o%2FzSP4U3lEfNKnSJ3ivJ9SL96mUoZiPaV0aq%2BaZMirJ34EVuT0hnhKXHdAI0unWj6pdjCPMXkrERzQ8SKtJS3qc7feCS1eHu0tWGGeiBXKD3cuOnd0QKd9AirHAQRBV%2BQ1suXh8cMBSdMUDn55H9JqI3NdnaVRO9HsVefeQgu9N1ooDmqonWr0yQt5I0PwopbjWXh8LY9imiV7BsYUEzSZovrPMeVBis6oIkdVRfYsVZwh%2BlFWHL13xek1K%2FKmibg%2FxZm2VJxeBUc%2BX%2FyMie8uAkjlVAcDBmhsKbJwvSUdHF8svc0Mrcp3dMcX%2B93N0D0Grq2naKvX0FWeo2P%2BlQTvkTKgcUJA%2BS%2Fep2WMfOIxIigieInimHkKzNPtFe5UnG4i5bst84%2By8xstAxj%2FOFhJZFoWcbRhIRZbivySt4OOiaLy2NRRQQWq%2FRRbAar9%2B6CyYvktGdmJuPKrRoz7fwE%3D"></iframe>
### imgs2poses.py

是否使用过colmap：
- 如果已经使用colmap生成了`sparse/0/`下的` ['cameras', 'images', 'points3D']`文件，将获得sparse_points.ply
- 若没有，则使用`run_colmap()`，即可生成sparse/0/下文件

#### run_colmap()

```
def run_colmap(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.single_camera', '1',
            # '--SiftExtraction.use_gpu', '0',
    ]
    # subprocess.check_output: 运行命令行程序，等待程序运行完成，然后返回输出结果
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        'colmap', match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    # mapper_args = [
    #     'colmap', 'mapper', 
    #         '--database_path', os.path.join(basedir, 'database.db'), 
    #         '--image_path', os.path.join(basedir, 'images'),
    #         '--output_path', os.path.join(basedir, 'sparse'),
    #         '--Mapper.num_threads', '16',
    #         '--Mapper.init_min_tri_angle', '4',
    # ]
    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )
```

上述代码相当于分别运行:
```
colmap feature_extractor --database_path os.path.join(basedir, 'database.db') --image_path os.path.join(basedir, 'images') --ImageReader.single_camera 1
colmap match_type --database_path os.path.join(basedir, 'database.db')
match_type : exhaustive_matcher Or sequential_matcher
colmap mapper --database_path os.path.join(basedir, 'database.db') --image_path os.path.join(basedir, 'images') --output_path os.path.join(basedir, 'sparse') --Mapper.num_threads 16 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0
```

- feature_extractor: Perform **feature extraction or import features** for a set of images.
- exhaustive_matcher: Perform **feature matching** after performing feature extraction.
- mapper: **Sparse 3D reconstruction / mapping of the dataset** using SfM after performing feature extraction and matching.

然后将命令行的输出结果保存到logfile即`basedir/colmap_output.txt`中

> colmap命令行：[Command-line Interface — COLMAP 3.8-dev documentation](https://colmap.github.io/cli.html)
> dense中深度图转换：[COLMAP简明教程 导入指定参数 命令行 导出深度图 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/584386128)

#### load_colmap_data() to colmap_read_model.py

`python .\colmap_read_model.py E:\BaiduSyncdisk\NeRF_Proj\NeuS\video2bmvs\M590\sparse\0 .bin`

读取`['cameras', 'images', 'points3D']`文件的数据

input:
- basedir

output: 
- poses, shape: 3 x 5 x num_images
    - c2w: 3x4xn 
    - hwf: 3x1xn
- pts3d, 一个长度为num_points字典，key为point3D_id，value为Point3D对象
- perm, # 按照name排序，返回排序后的索引的列表：`[from 0 to num_images-1]`

##### cameras images and pts3d be like: 

>[Output Format — COLMAP 3.8-dev documentation](https://colmap.github.io/format.html#cameras-txt)

| var     | example                                                                                                                                                                                                                                    | info                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| cameras | `{1: Camera(id=1, model='SIMPLE_RADIAL', width=960, height=544, params=array([ 5.07683492e+02,  4.80000000e+02,  2.72000000e+02, -5.37403479e-03])), ...}`                                                                                 | f, cx, cy, k=params                              |
| images  | `{1: Image(id=1, qvec=array([ 0.8999159 , -0.29030237,  0.07162026,  0.31740581]), tvec=array([ 0.29762954, -2.81576928,  1.41888716]), camera_id=1, name='000.png', xys=xys, point3D_ids=point3D_ids, ...}`                               | perm = np.argsort(names),qvec,tvec to m=w2c_mats:4x4, |
| pts3D   | `{1054: Point3D(id=1054, xyz=array([1.03491375, 1.65809594, 3.83718124]), rgb=array([147, 146, 137]), error=array(0.57352093), image_ids=array([115, 116, 117, 114, 113, 112]), point2D_idxs=array([998, 822, 912, 977, 889, 817])), ...}` |                                              |

xys and point3D_ids in images be like:

```
xys=array([[ 83.70032501,   2.57579875],
       [ 83.70032501,   2.57579875],
       [469.29092407,   2.57086968],
       ...,
       [759.08764648, 164.65560913],
       [533.28503418, 297.13980103],
       [837.11437988, 342.07727051]]), 
point3D_ids=array([  -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,       
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 9109,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1, 8781,   -1,   -1, 8628,   -1,   -1,
 -1, 2059,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 8791,   -1,   -1, 8683,   -1, 8387,   -1,
 -1,   -1,   -1,   -1,   -1, 9008, 9007,   -1, 9161, 8786,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1, 9175,   -1,   -1,   -1,
9053,   -1,   -1,   -1,   -1, 8756,   -1,   -1,   -1,   -1,   -1,
 -1, 9024,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, 9111,
 -1,   -1, 9018,   -1, 9004,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1, 8992,   -1,   -1,   -1,   -1,   -1,
4701,   -1, 9067,   -1, 9166, 3880,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1, 8725,   -1, 9112,   -1,
 -1,   -1,   -1, 8990,   -1, 8793, 9118, 8847, 9009, 9140, 9012,
 -1,   -1,   -1, 7743, 9065, 8604, 3935,   -1,   -1,   -1,   -1,
9075,   -1,   -1, 8966,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   19,   -1,   -1,   -1,   -1,   -1,
9017,   -1,   -1,   -1, 9020,   -1, 9005,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, 8696,
 -1,   -1, 8930,   -1,   -1, 8970,   -1,   -1,   -1,   -1, 9076,
 -1, 9114, 8925,   -1, 8915,   -1, 9077, 8851, 8655, 5885, 4073,
 -1, 3839,   -1,   -1,   -1,   -1, 9165, 9078,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 9055,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 9017,   -1,   -1,   -1,   -1,   -1,   -1,
 -1, 8682,   -1,   -1, 9170,   -1, 7562, 7556,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1, 8962, 9079,   -1,   -1,   -1, 8586,
8224,   -1,   -1,   -1,   -1, 1399, 9168, 6439, 9121, 8255, 9169,
 -1, 9151, 8971, 4698, 9171, 9172,   -1,   -1, 8898, 3916,   -1,
 -1,   -1, 1788,   -1,   -1,   -1, 9080,   -1,   -1,   -1,   -1,
 -1,   -1, 2097,   -1, 4103,   -1,   -1,   -1,   -1, 2073,   -1,
 -1, 1771,   -1,   -1,   -1,   -1,   -1,   -1, 8813,   -1, 9030,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, 8841, 9081,
 -1,   -1,   -1, 8977,   -1, 8372, 9057, 6807, 9082, 5941, 4181,
1675,   -1, 1683,   -1,   -1, 1503, 9083, 1973, 9071, 2679, 2412,
3238,   -1, 9164, 1796, 9174,   -1,   -1,   -1,   -1,   -1,   -1,
9042, 9084,   -1,   -1,   -1,   -1,   -1, 9051, 9050,   -1, 9085,
 -1, 9158, 9086,  853, 7671, 9128,   -1,   -1, 9058,   -1, 9087,
 -1, 8502, 9102,   -1, 9106,   -1, 9039,   -1,   -1,   -1, 9069,
 -1, 2261,   -1, 1793, 2643,   -1,   -1, 8810, 8945,   -1,   -1,
 -1,   -1,   -1, 9043,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
9142,   -1,   -1, 9122, 9089, 9090, 8863, 9103, 2161, 2446,   -1,
 -1,   -1,   -1,   -1, 9104,   -1, 9060, 9131,   -1,   -1,   -1,
 -1, 8980, 8706,   -1, 9105, 9091, 9173,   -1,   -1, 2996,   -1,
 -1, 9092,   -1,   -1,   -1,   -1, 9094, 9095, 9096, 9097, 9156,
 -1,   -1,   -1,   -1, 8772, 8818,   -1,   -1, 9162, 9062, 9098,
 -1,   -1, 8907, 9099, 8985, 4624,   -1, 3746, 8951,   -1,   -1,
8908,   -1, 9135, 8986, 9101,   -1,   -1,   -1, 9137,   -1]))}
```

##### cameras文件

input:
- path_to_model_file, `camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')`
output:
- cameras，一个长度为num_cameras字典，key为camera_id，value为Camera对象

>[colmap 相机模型及参数 - 小小灰迪 - 博客园 (cnblogs.com)](https://www.cnblogs.com/xiaohuidi/p/15767477.html)

使用:
```
camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
camdata = read_model.read_cameras_binary(camerasfile)

list_of_keys = list(camdata.keys()) # list from 1 to num_cameras
cam = camdata[list_of_keys[0]] # Camera(id=1, model='SIMPLE_RADIAL', width=960, height=544, params=array([ 5.07683492e+02,  4.80000000e+02,  2.72000000e+02, -5.37403479e-03]))
print( 'Cameras', len(cam)) # Cameras 5

h, w, f = cam.height, cam.width, cam.params[0]
hwf = np.array([h,w,f]).reshape([3,1])
```

##### images文件

input:
- path_to_model_file,`imagesfile = os.path.join(realdir, 'sparse/0/images.bin')`
output:
- images，一个长度为num_reg_images字典，key为image_id，value为Image对象

使用:
```
imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
imdata = read_model.read_images_binary(imagesfile)

w2c_mats = []
bottom = np.array([0,0,0,1.]).reshape([1,4])

names = [imdata[k].name for k in imdata] # 一个长度为num_images的list，每个元素为图片的名字
print( 'Images #', len(names)) 
perm = np.argsort(names) # 按照name排序，返回排序后的索引的列表：[from 0 to num_images-1]
for k in imdata:
    im = imdata[k]
    R = im.qvec2rotmat() # 将旋转向量转换成旋转矩阵 3x3
    t = im.tvec.reshape([3,1]) # 平移向量 3x1
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0) # 4x4
    w2c_mats.append(m) # 一个长度为num_images的list，每个元素为4x4的矩阵

w2c_mats = np.stack(w2c_mats, 0) # num_images x 4 x 4
c2w_mats = np.linalg.inv(w2c_mats) # num_images x 4 x 4

poses = c2w_mats[:, :3, :4].transpose([1,2,0]) # 3 x 4 x num_images
poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
# tile : 将hwf扩展成3 x 1 x 1 ，然后tile成3 x 1 x num_images，tile表示在某个维度上重复多少次
# poses : 3 x 5 x num_images ，c2w：3 x 4 x num_images and hwf: 3 x 1 x num_images

# must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
```

其中`R = im.qvec2rotmat()`将旋转向量转换成旋转矩阵:

如果给定旋转向量为 [qw, qx, qy, qz]，其中 qw 是标量部分，qx, qy, qz 是向量部分，可以通过以下步骤将旋转向量转换为旋转矩阵：

构造单位四元数 q：
```css
q = qw + qx * i + qy * j + qz * k  其中 i, j, k 是虚部的基本单位向量。
```

计算旋转矩阵 R(w2c)：
``` perl
R = | 1 - 2*(qy^2 + qz^2)   2*(qx*qy - qw*qz)   2*(qx*qz + qw*qy) |
    | 2*(qx*qy + qw*qz)     1 - 2*(qx^2 + qz^2) 2*(qy*qz - qw*qx) |
    | 2*(qx*qz - qw*qy)     2*(qy*qz + qw*qx)   1 - 2*(qx^2 + qy^2) |
```

```python
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
```

##### points3D文件

input:
- path_to_model_file: `points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')`
output:
- pts3D, 一个长度为num_points字典，key为point3D_id，value为Point3D对象

```
points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
pts3d = read_model.read_points3d_binary(points3dfile)
```

#### save_poses.py

input:
- basedir, 
- poses, shape: 3 x 5 x num_images
    - c2w: 3x4xn 
    - hwf: 3x1xn
- pts3d, 一个长度为num_points字典，key为point3D_id，value为Point3D对象
    - `{1054: Point3D(id=1054, xyz=array([1.03491375, 1.65809594, 3.83718124]), rgb=array([147, 146, 137]), error=array(0.57352093), image_ids=array([115, 116, 117, 114, 113, 112]), point2D_idxs=array([998, 822, 912, 977, 889, 817])), ...}`
- perm, # 按照name排序，返回排序后的索引的列表：`[from 0 to num_images-1]`

save:
- sparse_points.ply : 
    - pcd = trimesh.PointCloud(pts) , pts: num_points x 3
- poses.npy : num_images x 3 x 5

```python
def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d: # k为空间点的id
        pts_arr.append(pts3d[k].xyz) # 每个空间点的三维坐标
        cams = [0] * poses.shape[-1] # 一个长度为num_images的list，每个元素为0
        for ind in pts3d[k].image_ids: # 每个空间点对应的图片index
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1 # 将第k个空间点对应图片的index 在cams列表中置为1
        vis_arr.append(cams) 
    # pts_arr shape： num_points x 3
    #vis_arr shape： num_points x num_images

    pts = np.stack(pts_arr, axis=0) # num_points x 3
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, 'sparse_points.ply'))

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    # pose: 3 x 5 x num_images
    poses = np.moveaxis(poses, -1, 0) # 将最后一个维度移动到第一个维度 num_images x 3 x 5
    poses = poses[perm] # 按照perm排序 num_images x 3 x 5
    np.save(os.path.join(basedir, 'poses.npy'), poses) # num_images x 3 x 5
```

### gen_cameras.py

根据pose.npy文件和sparse_points_interest.ply文件来生成cameras_sphere.npz
- pose.npy主要保存每张图片的c2w矩阵和hwf
- sparse_points_interest.ply用来生成相机缩放矩阵，将感兴趣的部位保存下来

#### pose文件
pose.ply in Miku
![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702145000.png)


```python
work_dir = sys.argv[1]
poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
poses_raw = poses_hwf[:, :, :4] # n_images, 3, 4
hwf = poses_hwf[:, :, 4] # n_images, 3
pose = np.diag([1.0, 1.0, 1.0, 1.0]) # 4, 4 对角线为1，其余为0
pose[:3, :4] = poses_raw[0] # 3, 4 ， pose: 4, 4
pts = []
# 下面四句是将相机坐标系的四个点转换到世界坐标系
# 世界坐标系下 原点，x轴，y轴，z轴
pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3]) # 4, 1 -> 4 -> 3
pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
pts = np.stack(pts, axis=0)
pcd = trimesh.PointCloud(pts)
pcd.export(os.path.join(work_dir, 'pose.ply'))
```

#### 两个矩阵



**world_mat_{i}:**

```python
h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
intrinsic[0, 2] = (w - 1) * 0.5
intrinsic[1, 2] = (h - 1) * 0.5

intrinsic = 
[[ focal,  0.       ,   (w-1)/2  , 0 ]
[ 0.  ,       focal ,   (h-1)/2   , 0]
[ 0.  ,       0.      ,   1.            , 0]
[ 0.  ,       0.      ,   0.            , 1. ]]
np.float32

convert_mat = np.zeros([4, 4], dtype=np.float32)
convert_mat[0, 1] = 1.0
convert_mat[1, 0] = 1.0
convert_mat[2, 2] =-1.0
convert_mat[3, 3] = 1.0
pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
pose[:3, :4] = poses_raw[i]

pose = 
[[r1,        r2       ,  r3            ,  tx]
[ r1 ,       r2      ,   r3             , ty]
[ r1 ,       r2      ,   r3             , tz]
[ 0.  ,       0.      ,   0.            , 1. ]]
convert_mat =
[[0.,      1.      ,  0           , 0]
[1.,       0.       ,  0            , 0]
[0.,       0.       ,  -1.          , 0]
[0.,       0.       ,  0            , 1.]]
np.float32

pose = pose @ convert_mat

pose = 
[[r2,        r1       ,  -r3            ,  tx]
[ r2 ,       r1      ,   -r3             , ty]
[ r2 ,       r1      ,   -r3             , tz]
[ 0.  ,       0.      ,   0.            , 1. ]]

w2c = np.linalg.inv(pose)

# world_mat is w2pixel
world_mat = intrinsic @ w2c
world_mat = world_mat.astype(np.float32)
```

pose要乘以covert_mat是因为在load_colmap_data时对pose进行了翻转

```python
# must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
```

**scale_mat_{i}:**

```python
pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
vertices = pcd.vertices
bbox_max = np.max(vertices, axis=0) 
bbox_min = np.min(vertices, axis=0)
center = (bbox_max + bbox_min) * 0.5
radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
scale_mat[:3, 3] = center
```


## interpolate_view

生成一个视频，从img_idx_0中间插值生成新视图的图片，过渡到img_idx_1，然后再回到img_idx_0，共2s，60frames

eg: 0 to 38 render video

![00300000_0_38.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/00300000_0_38.gif)

插值：$ratio = \frac{\sin{\left(\frac{i}{frames}-0.5 \right)\cdot \pi}}{2}+\frac{1}{2} = 0.5 \rightarrow 1 \rightarrow 0.5$ 

```python
def interpolate_view(self, img_idx_0, img_idx_1):
    images = []
    n_frames = 60
    for i in range(n_frames):
        print(i)
        images.append(self.render_novel_image(img_idx_0,
                                              img_idx_1,
                                              np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                      resolution_level=4))
    # 做出了视频像是循环的效果
    for i in range(n_frames):
        images.append(images[n_frames - i - 1])

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_dir = os.path.join(self.base_exp_dir, 'render')
    os.makedirs(video_dir, exist_ok=True)
    h, w, _ = images[0].shape
    writer = cv.VideoWriter(os.path.join(video_dir,
                                         '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                            fourcc, 30, (w, h))

    for image in images:
        writer.write(image)

    writer.release()
```


```python
def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
    """
    Interpolate view between two cameras.
    """
    rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
    H, W, _ = rays_o.shape
    rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
    rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

    out_rgb_fine = []
    for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
        near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
        background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

        render_out = self.renderer.render(rays_o_batch,
                                          rays_d_batch,
                                          near,
                                          far,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                          background_rgb=background_rgb)

        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

        del render_out
```

```python
def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
    """
    Interpolate pose between two cameras.
    """
    l = resolution_level # 4
    tx = torch.linspace(0, self.W - 1, self.W // l)
    ty = torch.linspace(0, self.H - 1, self.H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
    pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
    pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    rot = torch.from_numpy(pose[:3, :3]).cuda()
    trans = torch.from_numpy(pose[:3, 3]).cuda()
    rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
```