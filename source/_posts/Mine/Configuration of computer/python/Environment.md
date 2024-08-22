# python setup.py install 
cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}

# pip 清华源
-i https://pypi.tuna.tsinghua.edu.cn/simple
## 只使用ipv6的清华源
-i https://mirrors6.tuna.tsinghua.edu.cn/pypi/web/simple/ 

## 配置文件
win11：C:\Users\Qiyun\AppData\Roaming\pip\pip.ini
由于已经使用代理，把mirrors6删掉
index-url = https://mirrors6.tuna.tsinghua.edu.cn/pypi/web/simple/

## 添加requirements.txt文件

pip freeze > requirements.txt
