---
title: Instant-NSR代码理解
date: 2023-07-09 12:17:53
tags:
    - Code
    - Python
    - Instant-NSR
categories: NeRF/Surface Reconstruction
---

对[Instant-NSR](https://github.com/zhaofuq/Instant-NSR)代码的理解

<!-- more -->

# 训练流程图

<iframe frameborder="0" style="width:100%;height:1158px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=train.drawio#R7V1bc6M4Fv41lNNb1S5AIOAxjpPp3eqdnZme2tndFxc2ss02Bi%2FgJJ5fvxI3A1In6gSQaNMPHVvmIs7hO3cdKeDu8PxT7B73f488FCi66j0rYKnoumEAA%2F8hI%2Bd8RNctLR%2FZxb6Xj9UGvvh%2FomJQLUZPvoeSxoFpFAWpf2wObqIwRJu0MebGcfTUPGwbBc27Ht0doga%2BbNyAHv3D99J9Pmrr1mX8E%2FJ3%2B%2FLOGnTyXw5ueXDxJMne9aKn2hC4V8BdHEVp%2FunwfIcCQr2SLvl5D9%2F4tZpYjMKU54T%2Fxr%2BEUfpg7f72z5W2%2FOvu51%2B9Pz%2Faen6ZRzc4FU9czDY9lySIo1PoIXIVVQGLp72foi9Hd0N%2BfcJcx2P79BDgbxr%2BuI3CtOCiBvF3N%2FB3If6ywbNEMR6gp108ySOKU%2FRcGyoe4ycUHVAan%2FEh5a8qNPNzyreqIPHThUPAKA7Z17hTvVVu8VbsqmtfCIc%2FFLT7DjpqFBnT2PXDVYji7QsE1b6foF3QD6gN8pk0%2BTRVpcln9EU9gyIR8jAKi69RnO6jXRS6wf1ldNF8Ky%2FHfI6iY0G6%2F6I0PRe0c09p1CQsevbTf5HT52bx7d%2B1X5bPxZWzL%2BfyS4gft3YS%2Bfrv%2Bm%2BX07Jv5XncTEyiU7xBL5CqgGfqxjuUvnAcyI8jdHzxlYhR4Kb%2BY1Padc5gWspExxQPZJ8JHRIUz7M%2FK%2Fxgyc2HboVQB5ixmxJH0xiYcYbEDJwww4sZwIkZUyrMaEAkh7Uafy%2Fcfo3DDf5e2M3JYc9N9pVa7J%2Fdmi4VvwFLRs5DlD5F8VdaHu6jw%2FqUCJGFVtN80GxaFlYmWV0Wmr1ZX%2BYkDHnRYfKiA3aNjuLUXyI%2FTGvGqN18mwBQ52rtX%2BuC%2BbyLa7Rem2pSb3%2BTTAqECULeCmFwnNO9H%2B6kM0w0vUk%2FHTIsExYae7NMSrlaI%2BJmH0UJwmOVOIMBnsZiHTeICf93It5wRqaPSUanW3yABo%2FPGbHK3%2FGnHfn7M%2Frt4efyirkx2cl1P%2BIP7oHwLlwn2R%2FsxpWieLXddncbtXHlxNuu0k0Ytm8%2FyGORm6NwM9S9hD3jce8mUTzE7ehH7COyEqBt2o0oAa24gKabfE6OBnuTJfa1GcFv1%2Bylxn5dtVtSGb7VA07WW3c8LjW%2BJDwu5%2F0t76aMBBWyWSJnB1rzZugHMKLNus6INtu9CUWLIuahyLzkZKzZRTdYpUceMVzr6q3SgcV3%2BeJtQKfcAprqkKGL%2BlNFDkX1M0oowuHnS5vUSdI4%2BoruogBTHCzDKCTya%2BsHQWuISpUQavkbN7gtfjj4npcJPxY7mgzrgSM67ahRHDEYDNH7Yog%2Bef38aQOVV28YUumNct5cok4%2BKeY0IWMwIFPFPwaJlOlCg8pNzKiyY0YXZmtlp97GsXuuHXAkQa%2BkduVXImmGprbel%2FyKnQbMSiI1Cghy19rzH9%2FlW7fdZ6Ylgw2ZfVZKUjv9rpwBfqJsEtX1ep3XyvPjxgWT4x7FaLV340MU%2BpukPUdm%2FKO3OQf5lNyw04cPT4dV4J5RTF5SPXsudY%2FtFBRichzwGDSaDMln0JzVyJ4ZI3W1RW5aPCHJPrzEzh%2FgiS9cXm0KoxW8ztZ3xtZGSqvL21%2FR6gfEwCtP8%2B77bU6eu8q01ZIECqqvPxodN6f40U1PWE0EUZJUZm3%2BzM2fvuuxO%2FdKizCzZD5p28BmlOVU6epGxLo3CxsIjWaOK2JdOpuvV7PJFc0EEnlR0kceuHmsS8Xjct7fjFj7faVpFXMxI1ngWW5Nz7B%2FUX5MvG35MQ%2FhzhRzSUc9ZAqXmw5nuNzqLeZB5x7CqAcNKWncVjNaOlIVHbel0xdXFUivaocLhkBG%2Ff2wDLGvGiDtxAYUDRBAB7278mOwCvMP%2BLC48jQC97D2XPwhj6tnp6dRvNnPs2Pnt557uMH0dfGzkSiPfhfgk%2B%2FWKHXJF3RMOvONPrzPxxEd7G%2BXz5gOXT4DWJV4vaUsgTPZqdxV47zRfmBLZaeW856ExaiERau8ATKs5IFlhdhs%2BlviFpqouAUYqU8LaJ%2B2swCp2WGM1FrMQixCZvlxszLBVrm%2BRMAkxa%2BBn6Q3mTiaZ8flCwkLAXTz4YNiLYUFjVvPEaL0tUcgBcr4sNYjEEGwaB6YpRZYh7aO89Cjj9%2BsKGQdm03lKZMuKw9t3HMxIQ19hE26vZ%2BnJE4xRbBpo7HtfLAWUzFrrvuLYFuTJuDWBLz1uLJFsOmY2KQJ5NIE%2Fu7gvlMXTPJduHyHrQyladFBAXaGUu1LvhvTegt%2B%2BW5zyndDMvlOh3STzR55p4DlzV8c%2FQzkuS8fxKvqlPnn7NDPv93UYgJ39Uv4aXm2OscPpf7lLyRK4Ic32Q9EaD3g%2F3Wyojc%2FU%2FsgX7luyym3hDvlxpRo5oZqicDXoSqXU27Q0f5NTFCDnbYKqnmQLQznn05rFH%2BOEhl75MAmfGyWK8NKpIDe4DOtEOGHD2%2F825BrhUg573brtZqe%2Bz3%2FLiFiKiOvgIxj0JDRWKvc%2BoPMtOCaHzK8YWBDrgXXpj7xuHseO3Lx%2BOq6hw3AY1MuJ88U2jfzR%2BWxJhWPDXYZKgn80daMRD3hVEYR1bBN4UyhHTJHhg6LFx1y9cg06FLSzPjv1tDvo4c1KPuEl5Y%2F0Od0ZJjZUdbsi5h0g4sDSvYjpCUjbjcoJU06pIMfzyMu6dhoqamM6uRhiUn3Vt8%2BPo6Qkqye0cNSUmg56Lg0osnbJ9WUzLenO5jmOxF4buom6NJ0nTROWZZjnaxbwrf0vZfvIztmjVY1jgPUOQQUaqvwWwO29tzqC7lQopStyoncZiBbGw65vLYslMyb%2F4Ytuwoi16tFs%2FNU0Cn1g2ROwDYn6PpcHNMhkL%2FntrLj2jK4cA0GxzVtKBbJi3n296b5Btzlh9aYQ0Z0Ve04s9EPC5rrQm2ozRn9ZZn59Y5Y8HA2wcb5%2Feff73%2F9z%2FHTs%2F7rR%2F8%2F5dYk12YTtYTe20Qt1GlRyySyUMFaznKUGULbbEquKtAmLEMI6Y0qxtSSz2btdjVoDz5rSs3wSxjesD2UKzAJ6bA9pLM10ilpTS%2F326iaHTOikiwrSbP7spHsqaEOP1x4oxaWXNlqSEctxgGXlnaWAC7WtDEcN1wsXu1iyaVdLKErtMYlErl5DOWqa7RoC%2BL5%2FKcCbmnrUbRJramtRS6Mwl%2FWEpfeXBRrKmLkxwevySAbPmiT4ZOb7AlA6JCOdAjRGIW%2Bw0KETvaTTbnArdZXVzu9qythGpB50rX9wrnc3rSEUYfAYnJvne9sRqhmdU2d1gCPfT5oby%2Bb1YJDej%2FHaOVPGFsgAVYMTdd6o6PQpT3jUvE2b5MCW67WVjbdpGAUUQGz9MHliQo4U%2BkPP1x413w7ksGFXvM9CrhAvZkZlgEu6gQXXrg4Gq926XxH8ffxmK4r%2FfKpdD26hU0nJhhsokS0A%2BnQdusEkW%2FRCowUIrT%2FSBrU7WLXkzMQabT8PY1RLVEqjWFQQrt7v%2F20yKWMhPSz2juAq7QmZkkZozf60VG%2Bw2pDEe7HjWBYlmwRDIf2yuSNyLW1pujMhEMX%2BeZZHTnlQZt8GqNic1D6VQ0zJrODw%2BzgdmTlSu04Y3VkeWR1b44ss%2FBWaIcAccUg31vd%2FO2iZY7iZjAQVl6aJLt5WPfqhEIDg7QvZEHbaQRWMbPakz5hku9Ki6U6wQfDqWUeZ4rEB21vieD4tXOBjiPU%2BojKLqUAI9s5rJTShBq9IxdTjPom5nGOSIDQYY56%2B90RYES4JmdEHyaM8GLE4cSINpRf%2BNIsayBBBzdvbC8%2FQgzxWkRoVezIEaLxeoN57lCYO0j7g8nGHYUKMcSrENpMVe4dxbGVBf4Aldulcos%2F4K8Pim2REXuhOA%2FKvak4qmLT8JJ3T4v3sU4z27KNFeliMk%2FvK9KlMQorGcyD5P%2FbjGeLW%2BUWkhHyQScjmKn489Vy0ZKAi0Lr1Mauoxh1nuwDLaE6ik4gomO02Xe1PCCI1m6wSlJ07OiKQbTp9oJJ6qYJEThYKDlWJnmwOKL1h3Q62uQ1Yp2%2BdHQFhklAvEFAMFpjsamsChUQdNB2jZJ0lTc9VrMRdXbww5mEiNGaGXMIGIgxGYjpot0Im5dXuodQJ4gp5d3riNFFIqacZj1%2BGO1WxzSu8BIdUXhzUUWkO0%2BSIUN%2FIIeS3Qnn6XNaV1Z3rqIveung8y6MOWZlpxYgs1jNxi1GnV9%2FINMpBpwStNrs0ebrMfLDsh9mxooLjfG7gQVbjebEbujIZnG92u37acTUNRsZEYBh2chYOnwV1UXdyEpGpwU2lYVaFzodKKj1qJUQKJreWjw%2BaA84Ng3FhpLrOBmdRcHdb0GsRUFnJLvaDHkW7o757sfsRgNKL1swP7pxQnYv%2FTD4nYvuiAPf9WLgYTwe03nN4Bt4JrXdZZfN7WOHnMWlLA4sayVyA8%2FikrAkMomsbhh4AtsjEWhLgoR5mDfpzO3SGfllNjw6GnUOYFmreRh4ItWdV6ejh%2B3yFcJa%2B5yH%2FcDy9%2Fg0PHJaLgRmUu4wDC89EdYDK5%2FUSuMP%2BApmbe%2F49h2vI33SbimuM%2FrKMtdGdLIyh62z6QIJAvOji%2BeTdQrvyLdMz0dy4izrojyrvQnvvG4W38u11TaKD27a6azX5I0oLp9%2FvpI3lfIfAGNVn8Zal9BFYyK2BzgVa73dgwC8MUlgiPQgAB2TzGKQWGqECcH36iKYZvFMQr%2B7vRsM0%2B9mtT3qze8GU3r8Hbjhjk8JLeECdHzKi57CrIxLPoyYWjuKC1gZL62nRffsRW9Tufw7UMJbRAKEFpEAuogEuyexHyb%2Bhu5wKCNMGEvRB4bJVErxDpjwllIAWyhMGKUUhIN%2BuMOj63X0PAasGOJVylRE8Q6sMPoQsKksNOUB6LYDZH%2B7HCp%2FPbg7JKNiUS%2FbqJVoYWkWYM2HxIvYzddHjpccBjxreoUu6oVizewah%2FmLJUbIY6FmdjnNmkwMUbxdHVxsaz%2Bv0mgV7o7ySUXDMav9PcvSZZYNUdtYchgzQqxYfAtktAZgiK4RhxneAI5YO8KgAzjhce7GsXu%2B2SrmYnaJgOYwminmMo%2BDekVqBR%2B%2FDSI3BXqekFVdd73G4wkKtvPsczZapjSuK50BW%2BkMw7EY65Z0HdLAhr3ZO2J96Tqw67AmleY1YMPy2y8o9vGDZxn3F6oJX4A6AERyFiPty%2FUpA3j9bkOo323QfncG23tLcRZkqc89VJw7strw3lbsO8UG2SpS%2FBPdsSvZu5lQ2JzW3wJd7Z1Y5y%2FV53U14G6%2B7rJX7R%2BnNPBDVIx7bvz1H%2FgyfpoXo6hmc1DPRvvS0E578SFjrxTmOolquVH3GBbr438Tw0ArtG4JY7PE3riRzBsVMMVqczoqUKRiVMzKZnVCUQFjLj4WGvku%2F0vU%2BxWA%2BrKrVQVqyNrjvXijKWjblR7vHtu0H3OIHgkTiw6QBTefojjwaqPXwDWgNblmqsx14AKYxlrSbyoLVbEhS5NCxXYUW71oUjJik94M5OBbxTbIh1snO9ghK28X2cjijgyS6xjK4j5bjmvju%2BTYLZo74OEHciDW1eR3fOFlbS4GmchtPvJQKPMFvg%2FIbojvsKiuWhzs5E9gFhd0ltk0oXKrZq0lLHJucdZ91mPCUmydXPNarHzKQGAsPdYtg%2Bm%2F92fmT%2FUXb1f4Jm8nNlOo6W7SKx%2BqzDLFfeGRLhPoc1tvym9Wk3NjXqY3hon%2F0%2F5PiJ7Io0UJXcXyo8owQ6eSMyakVStz57zeOt7qkwh7uwjj9Vmg0EZ5Ju2z%2BCR%2FOQrxxeqYMLT4YrSbn0DCCxLI2yoPCu2cXk6zBhKinBLaofeL3D81Xis4a%2F%2F2B3b%2BP8mIN20ONbX6pzWhx7Ic4LzcK7rh9vUGPboJX%2BaZpyhMItojF09RrOMN1bn8a1DUYlDUsebaoMLMmoTZ24UZr9MChRaNQ9ppyVZnoXie%2Fb3J%2Fl%2BRriDF0kpyqO81RnRVlXAVRjuoaEONFVSsurQPgilLbO3LyDHFW2IOhQYCIF1i%2FuV0OLjx%2BY9s%2Fbh8UNFbzqbDatdnDlnwYo3G2%2Bzzfef1Gi2hXiOkvcZca0QhWhUdOqV%2F4y2n8hSFvfSQXoe9JaYrsbZVPyQMdcMdusFf73SSzl5otN79UWNiADbVuWMyck0OKyKm95ZosoQuRx552y%2BL1923hC5Htmh3P6umKwQbfuFM8rTZUKPZRa1xn5q1GGjLwH10WJ8SQcaxabXWymgqYBTDsQDVmwtv0S78mbF24keVcNQm1CqjNQhgKKQ%2BJZzQGMDYJRxv3bElNAZg0eUSRJKdSC8jiU04w76YbBVioHDHRWimf%2ByA4XXwLaEOvkU7%2BIn72OzaKxtaoN1WLsKhUubvJqi8BSq8sQFbaGzAYlTB0tYzsXNktpZt2AYPy1ZmdcLqzVYuL3ydtrKu2lSFjAzmsi1R%2Fl%2FlFGnN8L4mTqTZvAEBW2j%2B32YEBCTX%2FtjfhzRcxBsAYhPMbzEAWqtaBRoANu8G3rZQ59KmYzqZAUBa1OalGevIjb1%2FkXuTlS95JC3fOSt2w69SGwa6CkFVSF7FpRn15qwm%2B%2F2ZBnRO%2F6pMAwBeZcjQZgHDYSSv%2BVOWD55vgihBMm4gYVvN%2BIpjMXIuHakM%2FDWOSOPd6ref8DPu%2F541wQf3%2Fwc%3D"></iframe>

# 神经网络结构

## Neus网络

`in network_sdf.py NeRFNetwork(NeRFRenderer)`

Instant-NSR训练隐式模型的网络由两个串联的MLP组成：
- 一个具有2个隐藏层的SDF MLP $𝑚_{𝑠}$
    - 每个隐藏层宽度为64个神经元
    - 用Softplus替换了原始的ReLU激活函数，并且对所有隐藏层的激活函数设置了𝛽=100，SDF MLP使用哈希编码函数(NGP)将3D位置映射为32个输出值。
- 一个具有3个隐藏层的颜色MLP $𝑚_{𝑐}$
    - 每个隐藏层宽度为64个神经元

$m_{s}$训练SDF的网络表示为:$(x,F_{geo})=m_{s}(p,F_{hash}).$
- input
    - 每个三维采样点的3个输入空间位置值
    - 来自哈希编码位置的32个输出值
- output
    - sdf值，然后我们将截断的函数应用于输出SDF值，该值使用sigmoid激活将其映射到`[−1,1]`：`sigma = F.relu(h[..., 0])`
    - 15维的$F_{geo}$值

![m_s.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/m_s.png)


$m_{c}$训练颜色的网络表示为：$\hat{C}=m_{c}(\mathrm{p},\mathrm{n},\mathrm{v},F_{geo}).$
- input
    - 视角方向在球谐函数基础上分解为4阶及以下的前16个系数
    - SDF MLP的15维的输出值$F_{geo}$
    - 每个三维采样点的3个输入空间位置值
    - 用有限差分函数估计SDF梯度的3个正态值
- output
    - RGB: 3

用sigmoid激活将输出的RGB颜色值映射到[0,1]范围

![m_c.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/m_c.png)

### 训练细节

我们在论文中证明了我们约10分钟的训练结果与原始NeuS的约8小时优化结果是可比较的。在优化阶段，我们假设感兴趣的区域最初位于单位球内。
- 分层采样：我们在PyTorch实现中采用了NeRF的分层采样策略，其中粗采样和细采样的数量分别为64和64。我们每批次采样4,096条光线，并使用单个NVIDIA RTX 3090 GPU进行为期6,000次迭代的模型训练，训练时间为12分钟。
- 法线计算：为了近似梯度以进行高效的法线计算，我们采用有限差分函数$𝑓 ′ (𝑥) = (𝑓 (𝑥 + Δ𝑥) − (𝑓 𝑥 − Δ𝑥))/2Δ𝑥$。在我们的PyTorch实现中，我们将近似步长设置为Δ𝑥 = 0.005，并在训练结束时将其减小到Δ𝑥 = 0.0005。
- Loss：我们通过最小化Huber损失$L_{𝑐𝑜𝑙𝑜r}$和Eikonal损失$L_{𝑒𝑖𝑘}$来优化我们的模型。这两个损失使用经验系数𝜆进行平衡，在我们的实验中将其设置为0.1。
- Adam：我们选择Adam优化器，初始学习率为$10^{−2}$，并在训练过程中将其降低到$1.6 \times 10^{-3}$。

#### 有限差分法：

[有限差分法 - 维基百科，自由的百科全书 (wikipedia.org)](https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E5%B7%AE%E5%88%86%E6%B3%95)

```
def gradient(self, x, bound, epsilon=0.0005):
    #not allowed auto gradient, using fd instead
    return self.finite_difference_normals_approximator(x, bound, epsilon)

def finite_difference_normals_approximator(self, x, bound, epsilon = 0.0005): # 有限差分法
    # finite difference
    # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
    pos_x = x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)
    dist_dx_pos = self.forward_sdf(pos_x.clamp(-bound, bound), bound)[:,:1]
    pos_y = x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)
    dist_dy_pos = self.forward_sdf(pos_y.clamp(-bound, bound), bound)[:,:1]
    pos_z = x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)
    dist_dz_pos = self.forward_sdf(pos_z.clamp(-bound, bound), bound)[:,:1]

    neg_x = x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)
    dist_dx_neg = self.forward_sdf(neg_x.clamp(-bound, bound), bound)[:,:1]
    neg_y = x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)
    dist_dy_neg  = self.forward_sdf(neg_y.clamp(-bound, bound), bound)[:,:1]
    neg_z = x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)
    dist_dz_neg  = self.forward_sdf(neg_z.clamp(-bound, bound), bound)[:,:1]

    return torch.cat([0.5*(dist_dx_pos - dist_dx_neg) / epsilon, 0.5*(dist_dy_pos - dist_dy_neg) / epsilon, 0.5*(dist_dz_pos - dist_dz_neg) / epsilon], dim=-1)
```


## Neural Tracking Implementation Details

在第4节中，我们提出了一种神经跟踪流程，它将传统的非刚性跟踪和神经变形网络以一种由粗到精的方式结合在一起。我们通过高斯-牛顿方法解决非刚性跟踪问题，并在接下来的内容中介绍了详细信息。
跟踪细节: 在进行非刚性跟踪之前，我们通过计算规范网格上的测地距离来对ED节点进行采样。我们计算平均边长，并将其乘以一个半径比例，用于控制压缩程度，以获得影响半径𝑟。通过所有的实验，我们发现简单地调整为0.075也可以得到很好的结果。给定𝑟，我们按Y轴对顶点进行排序，并在距离𝑟之外时从现有ED节点集合中选择ED节点。此外，当ED节点影响相同的顶点时，我们可以将它们连接起来，然后提前构建ED图以进行后续优化。

$\mathrm{(c,\sigma)=\phi^{o}(p'+\phi^{d}(p',t),d).}$
网络结构: 我们改进阶段的关键包括规范辐射场$𝜙^𝑜$和变形网络$𝜙^𝑑$。
- $𝜙^𝑜$具有与Instant-NGP相同的网络结构，包括三维哈希编码和两个串联的MLP: 密度和颜色。
    - 三维坐标通过哈希编码映射为64维特征，作为密度MLP的输入。然后，密度MLP具有2个隐藏层（每个隐藏层有64个隐藏维度），并输出1维密度和15维几何特征。
    - 几何特征与方向编码连接在一起，并输入到具有3个隐藏层的颜色MLP中。最后，我们可以获得每个坐标点的密度值和RGB值。

- $𝜙^{𝑑}$包括四维哈希编码和单个MLP。
    - 四维哈希编码具有32个哈希表，将输入（p′，𝑡）映射到64维特征。通过我们的2个隐藏层变形MLP（每个隐藏层具有128个隐藏维度），最终可以得到Δp′。

### 训练细节

训练细节。我们分别训练$𝜙^𝑜$和 $𝜙^{𝑑}$ 。我们首先利用多视图图像来训练规范表示 $𝜙^𝑜$。当 PSNR 值稳定下来（通常在100个epoch之后），我们冻结 $𝜙^𝑜$ 的参数。然后，我们训练变形网络 $𝜙^{𝑑}$ 来预测每帧的变形位移。我们构建了一个PyTorch CUDA扩展库来实现快速训练。我们首先将规范帧中的ED节点转换到当前帧，然后构建一个KNN体素。具体而言，我们的体素分辨率是2563到5123，并且对于KNN体素中的每个体素，我们通过堆查询4到12个最近邻的ED节点。基于KNN体素，我们可以快速查询体素中的任何3D点，并获取邻居和对应的蒙皮权重，以通过非刚性跟踪计算坐标。

## Neural Blending Implementation Details

U-Net:
![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230706205127.png)

### 训练细节

在训练过程中，我们引入了一个称为遮挡映射的新维度，它是两个变形深度图之间的差异计算得出的。然后，我们将遮挡映射（1维）和两个变形的RGBD通道（4维）作为网络输入，进一步帮助U-net网络优化混合权重。在传统的逐像素神经纹理混合过程中，混合结果仅从两个变形图像生成。然而，如果目标视图在相邻虚拟视图中都被遮挡，将导致严重的伪影问题。因此，我们使用纹理渲染结果作为额外输入，以恢复由于遮挡而丢失的部分。为了有效地渲染，我们首先将输入图像降采样为512×512作为网络输入，然后通过双线性插值上采样权重映射以生成最终的2K图像。为了避免混合网络过度拟合额外的纹理渲染输入，我们在训练过程中应用高斯模糊操作来模拟低分辨率的纹理渲染图像。这个操作有助于网络专注于选定的相邻视图的细节，同时从额外的纹理渲染输入中恢复缺失的部分。此外，我们选择Adam [Diederik P Kingma et al.2014]优化器，初始学习率为1e-4，权重衰减率为5e-5。我们在一台单独的NVIDIA RTX 3090 GPU上使用Twindom [web.twindom.com]数据集对神经纹理混合模型进行了两天的预训练。





# 编码方式

Instant-NSR自己集成了`cuda.C++`程序，可以实现哈希编码而不需要安装tiny-cuda-nn扩展

## encoding.py

## hashencoder

## psencoder

## shencoder


