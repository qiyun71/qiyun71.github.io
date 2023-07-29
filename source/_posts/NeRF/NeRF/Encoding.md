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