# start tensorboard

默认在6006端口上
tensorboard --logdir /root/tf-logs

tensorboard --port 6007 --logdir /root/tf-logs

# shutdown after running python

python example.py && /usr/bin/shutdown 


# Pytorch with cuda in linux

安装pycuda时，gcc一直找不到lcuda：添加软连接

`sudo ln -s /usr/local/cuda/lib64/libcuda.so /usr/lib/libcuda.so`
`sudo ln -s /usr/local/cuda/lib64/libcuda.so /home/yq/miniconda3/envs/retr/lib/libcuda.so`


安装带cuda版本的Pytorch：[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)
> [高版本CUDA能否安装低版本PYTORCH？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/450523810)

