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


# import library

## 方案三：将你的项目安装为可编辑包 (最专业、最强大的方法)

这是最规范、最一劳永逸的方法，尤其适用于需要长期维护或多人协作的项目。你将把你的项目文件夹（如 `utils`）变成一个 Python 可以识别的“包”。

**操作步骤:**

1.  **在 `utils` 文件夹中创建一个空文件**，名为 `__init__.py`。这个文件的存在告诉 Python，`utils` 文件夹是一个可以被导入的包 (Package)。
    ```
    project_root/
    ├── utils/
    │   ├── __init__.py  <-- 新建这个空文件
    │   └── dataprocess.py
    └── data/
        └── data.ipynb
    ```

2.  **在项目根目录 `project_root/` 中创建一个名为 `setup.py` 的文件。**
    ```
    project_root/
    ├── setup.py         <-- 新建这个文件
    ├── utils/
    │   ├── __init__.py
    │   └── dataprocess.py
    └── data/
        └── data.ipynb
    ```    在 `setup.py` 文件中写入以下最基本的内容：
    ```python
    from setuptools import setup, find_packages

    setup(
        name='my_project_utils',  # 给你的包起个名字
        version='0.1',
        packages=find_packages(),
    )
    ```

3.  **以可编辑模式安装你的包。**
    打开终端，导航到项目根目录（`setup.py` 所在的目录），然后运行：
    ```bash
    pip install -e .
    ```
    *   `-e` 代表 "editable" (可编辑)。这意味着你对 `utils/dataprocess.py` 文件的任何修改都会立即生效，无需重新安装。
    *   `.` 代表当前目录。

完成这三步后，你的 `utils` 包就被“注册”到了你当前的 Python 环境中。现在，无论你的 notebook 在哪个目录下，也无论你从哪里启动 Jupyter，你都可以直接、可靠地导入它。

```python
# 在 data.ipynb 中，现在这样写就可以了
from utils.dataprocess import normalize_frf
```

*   **优点:**
    *   **一劳永逸**，解决了所有路径问题。
    *   项目结构清晰，非常专业。
    *   IDE 和代码编辑器的自动补全、代码跳转等功能会完美工作。
    *   方便未来打包和分发你的代码。
*   **缺点:**
    *   需要一点点初次设置。