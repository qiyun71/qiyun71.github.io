> [python - subprocess.CalledProcessError: ... returned non-zero exit status 255 - Stack Overflow](https://stackoverflow.com/questions/60355996/subprocess-calledprocesserror-returned-non-zero-exit-status-255)

```bash
# error
subprocess.CalledProcessError: Command 'cmd /u /c "D:\MicrosoftVS\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64 && set' returned non-zero exit status 255.
```


> [出现错误“subprocess.CalledProcessError: Command ‘\[‘ninja‘, ‘-v‘\]‘ returned non-zero exit status 1”解决方法_subprocess.calledprocesserror: command '\['ninja', -CSDN博客](https://blog.csdn.net/fq9200/article/details/125362088)

```bash
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

# solution
## 如果可以找到setup.py
将setup.py中的“cmdclass={'build_ext': BuildExtension}”这一行改为“cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}”，pytorch默认使用ninjia作为backend，这里把它禁用掉就好了；

## 可能是版本的问题，pip package
使用其他版本的package
```


> [bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats · Issue #156 · bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/156)

```bash
# error
bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats

# solution
Manual copy of the .so file worked. I have version cuda version 11.7 so the following command in the conda environment directory ensured that it worked correctly again.
cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so
```


> [solved: Could not load the Qt platform plugin "xcb" · NVlabs/instant-ngp · Discussion #300](https://github.com/NVlabs/instant-ngp/discussions/300)

no screen 

`export QT_QPA_PLATFORM=offscreen`