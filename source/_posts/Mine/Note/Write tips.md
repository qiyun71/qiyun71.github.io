> [科学网—How to Supervise Yourself (怎么自导博士论文) - 何毓琦的博文](https://blog.sciencenet.cn/blog-1565-242182.html) 广域搜索+深度发掘


# Write Paper

> [害怕写论文？你需要通过这个练习来训练你的写论文技能_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1tE421A7rM/?vd_source=1dba7493016a36a32b27a14ed2891088)

结构化写作，因果关系(描述一个大的现象-->由于这个现象导致结果-->对这个结果有哪些处理方法)，观点和相关支撑的论据

# Tools

## Colors

[# 论文配色 | 顶刊科研绘图高级配色汇总！](https://mp.weixin.qq.com/s/iAPY89fbYJkd5hBZ3I9dlw)

Blue: 1E4C9C 345D82 3371B3 5795C7 81B5D5 AED4E5
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240801201206.png)

## Latex

> [guanyingc/latex_paper_writing_tips: Tips for Writing a Research Paper using LaTeX](https://github.com/guanyingc/latex_paper_writing_tips)
> [Latex 定义新命令 - 知乎](https://zhuanlan.zhihu.com/p/383336622)

```latex
\documentclass{article}
\newcommand{\ILL}{I Love Latex!}       % 定义不带参数的命令
\newcommand{\WhoLL}[1]{#1 Love Latex!} % 定义带一个参数的命令
\newcommand{\WhoLWhat}[2]{#1 Love #2!} % 定义带两个参数的命令
\begin{document}
    \ILL                % 不带参数的命令的使用
    \WhoLL{I}           % 带一个参数的命令的使用
    \WhoLWhat{I}{Latex} % 带两个参数的命令的使用
\end{document}
```

$\approx$ $\thickapprox$

```latex
\documentclass{article}
\usepackage{amssymb}
\begin{document}
   \[ p \approx q \]
   \[ p \thickapprox q \]
\end{document}
```

### mathxx

`\mathrm{\mathbf I}` 加粗+正体

[What are all the font styles I can use in math mode? - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/58098/what-are-all-the-font-styles-i-can-use-in-math-mode)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240626091625.png)

### latex2word

latex转换成word格式: [🔁 将 LaTeX 格式的文档转换为 Word | BIThesis](https://bithesis.bitnp.net/guide/converting-to-word.html)

windows:
`scoop install pandoc`

普通转换: `pandoc main.tex -o main.docx`
转换成带格式的word: `pandoc main.tex --reference-doc=template.docx -o main.docx`
带参考文献的word: `pandoc main.tex --bibliography=refs.bib --reference-doc=template.docx -o main.docx` (不是很好用)

### Aurora伪代码

**(for word)**

> [使用Aurora在Word中插入算法伪代码教程，亲测有效，写论文必备_aurora word-CSDN博客](https://blog.csdn.net/jucksu/article/details/116307244)
> [使用Aurora+Algorithm2e在word中输入伪码 - 知乎](https://zhuanlan.zhihu.com/p/367884765)

algorithm2e语法 [mlg.ulb.ac.be/files/algorithm2e.pdf](https://mlg.ulb.ac.be/files/algorithm2e.pdf)


## Word

- 方框打勾，2611选中，alt+s快捷键

- 英文单词自动换行 [word中英文单词间距过大——换行或断字加横杠_英文单词换行断开加一横怎么加-CSDN博客](https://blog.csdn.net/Netceor/article/details/126480000)

- code 代码块粘贴到Word/PPT https://github.com/Lord-Turmoil/CodePaste



## PPT


 PPT风格文字
 ![](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240813172055.png)