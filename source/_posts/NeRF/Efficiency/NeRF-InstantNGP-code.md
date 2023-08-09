---
title: InstantNGP环境配置和tiny-cuda-nn用法
date: 2023-07-04 14:40:26
tags:
    - NeRF
    - InstantNGP
    - Python
    - Code
    - Efficiency
    - Encoding
categories: NeRF/Efficiency
---

tiny-cuda-nn在python中的用法:[NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework (github.com)](https://github.com/nvlabs/tiny-cuda-nn#pytorch-extension)

InstantNGP环境配置和使用，由于需要使用GUI，且笔记本GPU配置太低，因此没有具体训练的过程，只是进行了环境的配置。

<!-- more -->

# Tiny-cuda-nn

>[tiny-cuda-nn/samples/mlp_learning_an_image_pytorch.py at master · NVlabs/tiny-cuda-nn · GitHub](https://github.com/nvlabs/tiny-cuda-nn/blob/master/samples/mlp_learning_an_image_pytorch.py)

```
model = tcnn.NetworkWithInputEncoding(n_input_dims=2, 
                        n_output_dims=n_channels, 
                        encoding_config=config["encoding"], 
                        network_config=config["network"]).to(device)

"""
# encoding_config = 
"encoding": {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 15,
    "base_resolution": 16,
    "per_level_scale": 1.5
},
# network_config = 
"network": {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2
}
"""
```

encoding: [multiresolution hash encoding](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/multiresolution-hash-encoding-diagram.png) ([technical paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf))
- n_levels: 多分辨率个数L=16
- n_features_per_level: 特征向量的维度F=2
- log2_hashmap_size: log2每个分辨率下特征向量个数$log_{2}T=15$
- base_resolution: $N_{min} = 16$
- per_level_scale: 每个分辨率下的scale=1.5？？？

network: a lightning fast ["fully fused" multi-layer perceptron](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/fully-fused-mlp-diagram.png) ([technical paper](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf))
- activation: 激活函数 "ReLU"
- output_activation: 输出层激活函数无
- n_neurons: 64
- n_hidden_layers: 隐藏层数2

```
image =  Image(args.image, device) # model

model = tcnn.NetworkWithInputEncoding(n_input_dims=2, 
                                                                    n_output_dims=n_channels, 
                                                                    encoding_config=config["encoding"], 
                                                                    network_config=config["network"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 2**18
interval = 10

print(f"Beginning optimization with {args.n_steps} training steps.")

try:
    batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
    traced_image = torch.jit.trace(image, batch) 
    # 对 `image` 进行跟踪，记录其在给定输入数据上的执行过程，并生成一个跟踪模型。
    # 生成的跟踪模型可以被保存、加载和执行，而且通常具有比原始模型更高的执行效率。
    # 只能跟踪具有固定输入形状的模型或函数
except:
    # If tracing causes an error, fall back to regular execution
    print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
    traced_image = image

for i in range(args.n_steps):
    batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
    targets = traced_image(batch)
    output = model(batch)

    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    loss = relative_l2_error.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

tcnn.free_temporary_memory()
```

# NGP环境配置及运行

配置前下载：

- clone repository 
    - `git clone https://github.com/NVlabs/instant-ngp.git`
- download instant-ngp.exe
    - [Release Development release · NVlabs/instant-ngp (github.com)](https://github.com/NVlabs/instant-ngp/releases/tag/continuous)

配置环境:

>[Updated: Making a NeRF animation with NVIDIA's Instant NGP - YouTube](https://www.youtube.com/watch?v=3TWxO1PftMc)

```
conda create -n ngp python=3.10 
conda activate ngp 
cd C:\Users\ehaines\Documents\_documents\Github\instant-ngp 
pip install -r requirements.txt
```

```
# Set environment in Anaconda
conda activate ngp

# Pull images from movie; I've put movie directory "chesterwood" in the instant-ngp directory for simplicity. Change "fps 2" to whatever is needed to give you around 100 images.
cd C:\Users\(your path here)\Github\instant-ngp
cd chesterwood
python ..\scripts\colmap2nerf.py --video_in IMG_9471.MOV --video_fps 2 --run_colmap --overwrite
# NOTE! This line is a bit different than shown in the video, as advice on aabb_scale's use has changed. Also, I usually want to delete a few images after extracting them, so I don't do an exhaustive match at this point. In fact, I usually hit break (Control-C) when I see "Feature extraction" starting, as the images have all been extracted at that point.

#After you delete any blurry or useless frames, continue below to match cameras.

# Camera match given set of images. Do for any set of images. Run from directory containing your "images" directory.
python C:\Users\(your path here)\Github\instant-ngp\scripts\colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --overwrite
# For videos or closely related sets of shots, you can take out the "--colmap_matcher exhaustive" from the line above, since your images are in order. This saves a few minutes. You could also leave off "--aabb_scale 16" or put 64, the new default; the docs say it is worth playing with this number, see nerf_dataset_tips.md for how (short version: edit it in transforms.json). In my limited testing, I personally have not seen a difference.

# run interactive instant-ngp - run from the main directory "instant-ngp"
cd ..
instant-ngp chesterwood
```

GPU配置太低，无法运行

```
出现错误：CUDA_ERROR_OUT_OF_MEMORY
Uncaught exception: D:/a/instant-ngp/instant-ngp/dependencies/tiny-cuda- nn/include\tiny-cuda-nn/gpu_memory.h:590 cuMemSetAccess(m_base_address + m_size, n_bytes_to_allocate, &access_desc, 1) failed with error CUDA_ERROR_OUT_OF_MEMORY

原因：GPU硬件配置太低1050Ti and only 4GB of VRAM
nvidia-smi  
Tue Jul 4 14:47:36 2023  
+---------------------------------------------------------------------------------------+  
| NVIDIA-SMI 531.79 Driver Version: 531.79 CUDA Version: 12.1 |  
|-----------------------------------------+----------------------+----------------------+  
| GPU Name TCC/WDDM | Bus-Id Disp.A | Volatile Uncorr. ECC |  
| Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute M. |  
| | | MIG M. |  
|=========================================+======================+======================|  
| 0 NVIDIA GeForce GTX 1050 Ti WDDM | 00000000:01:00.0 Off | N/A |  
| N/A 42C P8 N/A / N/A| 478MiB / 4096MiB | 2% Default |  
| | | N/A |  
+-----------------------------------------+----------------------+----------------------+
```

