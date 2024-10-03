# Tools

## Colors

| Reference                                                                                                         | Color                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![image.png\|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240801201206.png) | Blue: [# 论文配色 \| 顶刊科研绘图高级配色汇总！](https://mp.weixin.qq.com/s/iAPY89fbYJkd5hBZ3I9dlw)<br>1E4C9C<br>345D82 <br>3371B3 <br>5795C7 <br>81B5D5 <br>AED4E5 |
| ![image.png\|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240917194520.png) | Miku : [Hatsune Miku Color Palette](https://www.color-hex.com/color-palette/19601)<br>蓝色 37C8D4<br>红色 C92930<br>黑色 3A3E46                          |
|                                                                                                                   |                                                                                                                                                    |
|                                                                                                                   |                                                                                                                                                    |
|                                                                                                                   |                                                                                                                                                    |


## Latex

[公式 - 科学空间|Scientific Spaces](https://kexue.fm/latex.html)

### Basic

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


### Cheatsheet 

约等于 $\approx$ $\thickapprox$

```latex
\documentclass{article}
\usepackage{amssymb}
\begin{document}
   \[ p \approx q \]
   \[ p \thickapprox q \]
\end{document}
```


求和符号上下标位置：
- `\sum\nolimits_{j=1}^{M}`   上下标位于求和符号的水平右端，$\sum\nolimits_{j=1}^{M}$
- `\sum\limits_{j=1}^{M}`   上下标位于求和符号的上下处， $\sum\limits_{j=1}^{M}$

加粗：
- `\mathbf{I}` 加粗+正体 $\mathbf{I}$
  - 粗体正体一般用于表示**矩阵**或其他不变的数学对象，也用于强调特定的物理量或者符号，特别是在多维数组、矩阵或张量的场景下
- `$\boldsymbol{I}$` 加粗+斜体 $\boldsymbol{I}$
  - 粗体斜体常用于表示**向量**或**张量**。在许多领域（尤其是物理学和工程学），向量通常使用斜体加粗的符号来区别于标量

符号花体：[What are all the font styles I can use in math mode? - TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/questions/58098/what-are-all-the-font-styles-i-can-use-in-math-mode)

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240626091625.png)

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
> [使用Aurora写伪代码遇到的问题（安装、overleaf配合Aurora的使用方法）_aurora安装教程-CSDN博客](https://blog.csdn.net/TycoonL/article/details/115586651)

algorithm2e语法 [mlg.ulb.ac.be/files/algorithm2e.pdf](https://mlg.ulb.ac.be/files/algorithm2e.pdf)

宏包:
```latex
\makeatletter
\newif\if@restonecol
\makeatother
\let\algorithm\relax
\let\endalgorithm\relax
\usepackage[linesnumbered,ruled]{algorithm2e}%[ruled,vlined]{
\usepackage[linesnumbered,lined,boxed,commentsnumbered]{algorithm2e}
\usepackage{algpseudocode}
\usepackage{amsmath}
\usepackage{algorithm}
\renewcommand{\algorithmicrequire}{\textbf{Input:}}  % Use Input in the format of Algorithm
\renewcommand{\algorithmicensure}{\textbf{Output:}} % Use Output in the format of Algorithm 
```

- 修改算法标题为中文`\renewcommand{\algorithmcfname}{算法}`
- 修改算法编号`\renewcommand{\thealgorithm}{1:}`
- 消除竖线 `\SetAlgoNoLine`

```latex
{$$;} \par
```

## Word

- 方框打勾，2611选中，alt+s快捷键

- 英文单词自动换行 [word中英文单词间距过大——换行或断字加横杠_英文单词换行断开加一横怎么加-CSDN博客](https://blog.csdn.net/Netceor/article/details/126480000)

- code 代码块粘贴到Word/PPT https://github.com/Lord-Turmoil/CodePaste

- endnote 个别word无法插入编号问题
  - [Reference error hash (#) + number. - EndNote / EndNote How To - Discourse](https://community.endnote.com/t/reference-error-hash-number/310353)
  - [how can I change the citing format back? - EndNote / EndNote How To - Discourse](https://community.endnote.com/t/how-can-i-change-the-citing-format-back/310352/8)
  - [(2 条消息) endnote插入文献时出现{，#}这样的乱码，怎么解？ - 知乎](https://www.zhihu.com/question/44969655) [endnote插入文献时出现{，#}这样的乱码_endnote大括号和井号-CSDN博客](https://blog.csdn.net/qq_43739296/article/details/114420524) 

## PPT

 PPT风格文字
 ![](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240813172055.png)

# Write Paper

## Write Approach

### 自定义loss

> [An Interval Neural Network Method for Identifying Static Concentrated Loads in a Population of Structures](https://www.mdpi.com/2226-4310/11/9/770)

损失函数可以通过其他的评价指标来构建，并且通过几个缩放因子，来使得loss值当指标变好的时候下降的平缓gradual decreas，当指标变差的时候，loss飞速地上涨 

> [S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields | PDF](https://arxiv.org/pdf/2308.07032)

针对NeRF随机光线采样，导致SSIM评价指标无法捕捉图像中的区块相似度

## Basic Framework

> [科学网—How to Supervise Yourself (怎么自导博士论文) - 何毓琦的博文](https://blog.sciencenet.cn/blog-1565-242182.html) 广域搜索+深度发掘


> [害怕写论文？你需要通过这个练习来训练你的写论文技能_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1tE421A7rM/?vd_source=1dba7493016a36a32b27a14ed2891088)

结构化写作，因果关系(描述一个大的现象-->由于这个现象导致结果-->对这个结果有哪些处理方法)，观点和相关支撑的论据

> [获得清华博士学位的条件之一：教会导师一点东西](https://mp.weixin.qq.com/s/SnOTlwqWNIwW6z0mDGdgCA) 研究、写作 宏观

七个问题：如何做研究
● What’s the problem?
● Why is the problem important? 
● Why is the problem difficult ? 
● What is the state of the art?
● What is your magic touch？Idea
● What are the main results?
● What are the assumptions?

七股文：如何写论文
- Introduction：4段回答问题
  - What’s the problem?
  - Why is the problem difficult？
  - What is the state of the art? 简要提一下综述
  - What is your magic touch?
- Related Work (Literature  Review) 文献综述
  - 详细回答“What is the state of the art? ”
  - show how smart you are
- Problem Formulation 数学公式 (背景工作，参考了哪些人的理论)
  - 完整地回答“what is the problem”
- Main Results 应当占你全文2/3左右的内容 **主要工作** 
  - “what are the main results?”的完整阐述
  - 更应该讲清楚“what are the assumptions？”
- Numerical Experiments 实验部分
  - Main Results是让你们写定理，而“Numerical Experiments”是说实际应用
- Conclusion
  - 简要地回顾你解决了什么问题，你是怎么解决的，你主要做了什么样的贡献。后面的人如果在这个方向接着做，应当做什么？千万不要跟引言和摘要相同，他们有不同的作用。引言是让人家读完之后能知道你的问题是否重要，摘要是读完之后知道这篇文章值不值得往下读，而总结是我读完了会决定要不要在你这个方向继续往下研究。

## 科研习惯

1. 动手之前先动脑。

有了新的idea时，不要着急做，先查文献了解有没有别人做过。

如果符合基本原理，则要提前设计好实验方案，准备好相应的实验设备、材料等，并规划好实验时间。切记不要用行动的勤奋掩盖思维的懒惰！

2. 搞课题研究要先定框架，再填内容。

读研阶段你会发现无论是开题报告、组会汇报、实验方案、大论文和小论文的撰写等工作都是先列框架，再填充内容。然后，再不断修改和完善框架。

3. 每天坚持写一些东西。

无论是大论文、小论文，项目申报书，文献综述或是实验总结，给自己定个目标每天至少500字，坚持下来后你会发现写东西并不难，而且前期写的多的话后期完成毕业论文和小论文就可以直接复制粘贴了。

4. 文献不建议零散的看，一定要批量看。

文献下载下来先做好分类，每次打算看多少就下多少，比如下载了30篇文献，那就集中时间几天内把30篇文献全部看完，并且做一个文献综述的总结。

将这30篇文献进行梳理都总结到这篇文献综述中来，这样整理完写完，你对这些文献的理解可以上升至少一个层级。

5. 规划好实验数据用途。

做完实验，马上整理数据并作图分析，留好原始图 (origin PS等)，方便以后使用，把图片、数据表格，数据分析的文字填充到大论文相应的章节里，再填充到列好框架的小论文里。



[用随机梯度下降来优化人生 - 李沐的文章 - 知乎](https://zhuanlan.zhihu.com/p/414009313)


## English细节

图片：
As demonstrated in Figure 3,
- demonstrated
- shown
- illustrated
- depicted

因果
- 为此：To this end

loss function 惩罚penalize

### which

前有逗号，指代最前面整句的主语
前无逗号，指代前面的事物(可以是宾语)

which 的限定用法与非限定用法。 which当作关係代名词时，要特别注意限定用法与非限定用法。

例：Tom has a cat which can sleep all day. 汤姆有一隻可以睡整天的猫。

说明：which前面没逗号，所以which指的是”猫”，表示汤姆可能有很多猫，而这裡指的是整天在睡觉的那隻猫。这是限定用法。

例：Tom has a cat, which can sleep all day. 汤姆有一隻猫，可以睡整天。

## GPT (Write more, read less)

> [Chatbots in science: What can ChatGPT do for you?](https://www.nature.com/articles/d41586-024-02630-z)
> [The Perfect Prompt: A Prompt Engineering Cheat Sheet | by Maximilian Vogel | The Generator | Medium](https://medium.com/the-generator/the-perfect-prompt-prompt-engineering-cheat-sheet-d0b9c62a2bba)
> [The Perfect Prompt: Cheat Sheet With 100+ Best Practice Examples - PART 1](https://www.linkedin.com/pulse/perfect-prompt-engineering-cheat-sheet-snippets-part-vogel-mxkcf/)


The basic principles of good prompt：
- **Be clear about what you want the model to do**. (use commands such as ‘Summarize’ or ‘Explain’)
- Ask the model to **adopt a role or persona** (‘You are a professional copy editor’).
- **Provide examples** of real input and output, potentially covering tricky ‘corner’ cases, that show the model what you want it to do.
- Specify **how the model should answer** (‘Explain it to someone who has a basic understanding of epigenetics’) or even the exact output format (for instance, as an analysis-friendly JSON or CSV file).
- Optionally, specify a word limit, whether the text should use the active or passive voice, and any other requirements. Check out the ‘[Prompt Engineering Cheat Sheet](https://medium.com/the-generator/the-perfect-prompt-prompt-engineering-cheat-sheet-d0b9c62a2bba)’ for more tips.

Here is a prompt that we use to **revise manuscript abstracts**, which we crafted on the basis of guidelines[1](https://www.nature.com/articles/d41586-024-02630-z#ref-CR1) published in 2017：
- You are a professional copy editor with ample experience handling scientific texts. Revise the following abstract from a manuscript so that it follows a context–content–conclusion scheme. (1) The context portion communicates to the reader the gap that the paper will fill. The first sentence orients the reader by introducing the broader field. Then, the context is narrowed until it lands on the open question that the research answers. A successful context section distinguishes the research’s contributions from the current state of the art, communicating what is missing in the literature (that is, the specific gap) and why that matters (that is, the connection between the specific gap and the broader context). (2) The content portion (for example, ‘here, we ...’) first describes the new method or approach that was used to fill the gap, then presents an executive summary of results. (3) The conclusion portion interprets the results to answer the question that was posed at the end of the context portion. There might be a second part to the conclusion portion that highlights how this conclusion moves the broader field forward (for example, ‘broader significance’).

你是一名专业的中翻英和学术润色专家，请帮我把以下中文翻译成英文，并进行专业的学术润色：


