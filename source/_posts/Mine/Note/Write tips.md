> [科学网—How to Supervise Yourself (怎么自导博士论文) - 何毓琦的博文](https://blog.sciencenet.cn/blog-1565-242182.html) 广域搜索+深度发掘

# Latex

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

## mathxx

[What are all the font styles I can use in math mode? - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/58098/what-are-all-the-font-styles-i-can-use-in-math-mode)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240626091625.png)


# Word

方框打勾，2611选中，alt+s快捷键

英文单词自动换行
[word中英文单词间距过大——换行或断字加横杠_英文单词换行断开加一横怎么加-CSDN博客](https://blog.csdn.net/Netceor/article/details/126480000)