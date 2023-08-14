---
title: Encoding
date: 2023-07-20 14:22:42
tags:
    - NeRF
    - Encoding
categories: NeRF/NeRF
---

对输入x进行编码的方式
[Field Encoders - nerfstudio](https://docs.nerf.studio/en/latest/nerfology/model_components/visualize_encoders.html)

<!-- more -->

# 频率编码

## NeRF

$$\gamma(p)=\left(\sin \left(2^{0} \pi p\right), \cos \left(2^{0} \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$$
```python
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

# multires <=> L
# use: get_embedder(args.multires, args.i_embed)
```


## Instant-nsr-pl

```python
get_encoding.py:
class CompositeEncoding(nn.Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims
    
    def forward(self, x, *args):
        return self.encoding(x, *args) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)

def get_encoding(n_input_dims,conf):
    if conf.otype == 'HashGrid':
        encoding = HashGrid(n_input_dims, conf)
    elif conf.otype == 'ProgressiveBandHashGrid':
        encoding = ProgressiveBandHashGrid(n_input_dims, conf)
    elif conf.otype == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, conf)
    elif conf.otype == 'SphericalHarmonics':
        encoding = SphericalHarmonics(n_input_dims, conf)
    else:
        raise NotImplementedError
    encoding = CompositeEncoding(encoding, include_xyz = conf.get('include_xyz',False), xyz_scale =2, xyz_offset = -1)
    return encoding

frequency.py：
class VanillaFrequency(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)

    def forward(self, x):
        out = []
        for freq in zip(self.freq_bands):
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)     

models/neus.py：
class N(nn.Module):
    __init__:
    self.encoding = get_encoding(n_input_dims = 3, config)
    forward:
    h = self.encoding(x)

yaml:
    xyz_encoding: 
      otype: HashGrid
      include_xyz: true
      n_frequencies: 10
```

## IPE
Mip-NeRF集成位置编码

对多元高斯近似截锥体的均值和协方差进行编码

$$\begin{aligned}
\gamma(\mathbf{\mu},\mathbf{\Sigma})& =\mathrm{E}_{\mathbf{x}\sim\mathcal{N}(\mathbf{\mu}_\gamma,\mathbf{\Sigma}_\gamma)}[\gamma(\mathbf{x})]  \\
&=\begin{bmatrix}\sin(\mathbf{\mu}_\gamma)\circ\exp(-(1/2)\mathrm{diag}(\mathbf{\Sigma}_\gamma))\\\cos(\mathbf{\mu}_\gamma)\circ\exp(-(1/2)\mathrm{diag}(\mathbf{\Sigma}_\gamma))\end{bmatrix}
\end{aligned}$$

IPE可以将大区域的高频编码求和为0
![ipe_anim_horiz.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/ipe_anim_horiz.gif)

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230721153610.png)



# HashGrid

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230703160333.png)

```python
hashgrid.py:
class HashGrid(nn.Module):
    def __init__(self, n_input_dims, config):
        super().__init__()
        self.n_input_dims = n_input_dims
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(self.n_input_dims, config_to_primitive(config))
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config['n_levels']
        self.n_features_per_level = config['n_features_per_level']

    def forward(self, x):
        enc = self.encoding(x)
        return enc

models/neus.py：
class N(nn.Module):
    __init__:
    self.encoding = get_encoding(n_input_dims = 3, config)
    forward:
    h = self.encoding(x)

yaml:
    xyz_encoding: 
      otype: HashGrid
      n_levels: 16
      n_features_per_level: 2
      log2_hashmap_size: 19
      base_resolution: 16
      per_level_scale: 1.447269237440378
      include_xyz: true
```

# 球面谐波编码

球面基函数——球谐函数
>[球谐函数介绍（Spherical Harmonics） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/351289217)


$\{Y_\ell^m\}.$ 与二维的三角函数基类似，球面谐波为三维上的一组基函数，用来描述其他更加复杂的函数

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230810145848.png)
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230810150600.png)

$$
\begin{aligned}
y_l^m(\theta,\varphi)=\begin{cases}\sqrt{2}K_l^m\cos(m\varphi)P_l^m\big(\cos\theta\big),&m>0\\[2ex]\sqrt{2}K_l^m\sin(-m\varphi)P_l^{-m}\big(\cos\theta\big),&m<0\\[2ex]K_l^0P_l^0\big(\cos\theta\big),&m=0\end{cases} \\
P_n(x)=\frac1{2^n\cdot n!}\frac{d^n}{dx^n}[(x^2-1)^n] \\
P_l^m(x)=(-1)^m(1-x^2)^{m/2}\frac{d^m}{dx^m}(P_l(x)) \\
K_{l}^{m}=\sqrt{\frac{\left(2l+1\right)}{4\pi}\frac{\left(l-\left|m\right|\right)!}{\left(l+\left|m\right|\right)!}}
\end{aligned}
$$

```python
spherical.py:
class SphericalHarmonics(nn.Module):
    def __init__(self, n_input_dims, config):
        super().__init__()
        self.n_input_dims = n_input_dims
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(self.n_input_dims, config_to_primitive(config))
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, x):
        enc = self.encoding(x)
        return enc

models/neus.py：
class N(nn.Module):
    __init__:
    self.encoding = get_encoding(n_input_dims = 3, config)
    forward:
    h = self.encoding(x)

yaml:
    dir_encoding: 
      otype: SphericalHarmonics
      degree: 4
```

## IDE

在Ref-NeRF中，借鉴Mip-NeR中的IPE，提出了IDE，基于球面谐波，将高频部分的编码输出置为0

$\mathrm{IDE}(\hat{\boldsymbol{\omega}}_r,\kappa)=\left\{\mathbb{E}_{\hat{\boldsymbol{\omega}}\sim\mathrm{vMF}(\hat{\boldsymbol{\omega}}_r,\kappa)}[Y_\ell^m(\hat{\boldsymbol{\omega}})]\colon(\ell,m)\in\mathcal{M}_L\right\},$
$\mathcal{M}_{L}=\{(\ell,m):\ell=1,...,2^{L},m=0,...,\ell\}.$

$\mathbb{E}_{\hat{\boldsymbol{\omega}}\sim\mathrm{vMF}(\hat{\boldsymbol{\omega}}_r,\kappa)}[Y_\ell^m(\hat{\boldsymbol{\omega}})]=A_\ell(\kappa)Y_\ell^m(\hat{\boldsymbol{\omega}}_r),$
$A_{\ell}(\kappa)\approx\exp\left(-\frac{\ell(\ell+1)}{2\kappa}\right).$

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230810152303.png)


use:
```python
self.sph_enc = generate_ide_fn(5)

dir_enc = self.sph_enc(reflections, 0)
# 将reflections编码，粗糙度为0
```

```python
import math

import torch
import numpy as np



def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

      Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
      (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

      Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

      Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
          np.math.factorial(l - k - m) *
          generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
  """Compute spherical harmonic coefficients."""
  return (np.sqrt(
      (2.0 * l + 1.0) * np.math.factorial(l - m) /
      (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))



def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

      This function returns a function that computes the integrated directional
      encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

      Args:
        deg_view: number of spherical harmonics degrees to use.

      Returns:
        A function for evaluating integrated directional encoding.

      Raises:
        ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = np.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    # mat = torch.from_numpy(mat.astype(np.float32)).cuda()
    mat = torch.from_numpy(mat.astype(np.float32)).cpu()
    ml_array = torch.from_numpy(ml_array.astype(np.float32)).cpu()
    # ml_array = torch.from_numpy(ml_array.astype(np.float32)).cuda()


    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.concat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.concat([(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn

def get_lat_long():
    res = (1080, 1080*3)
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij') # [h,w]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
    return reflvec
```