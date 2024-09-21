[www.jiqizhixin.com](https://www.jiqizhixin.com/articles/2022-05-24-6)机器之心翻译2022/05/24 17:36

![image|666](https://image.cubox.pro/cardImg/2023102119092917844/43445.jpg?imageMogr2/quality/90/ignore-error/1)

#### **1958 年：感知机的兴起**

1958 年，弗兰克 · 罗森布拉特发明了感知机，这是一种非常简单的机器模型，后来成为当今智能机器的核心和起源。

感知机是一个非常简单的二元分类器，可以确定给定的输入图像是否属于给定的类。为了实现这一点，它使用了单位阶跃激活函数。使用单位阶跃激活函数，如果输入大于 0，则输出为 1，否则为 0。

下图是感知机的算法。-
![](https://image.cubox.pro/cardImg/2023102119093261976/53685.jpg?imageMogr2/quality/90/ignore-error/1)

_感知机_

Frank 的意图不是将感知机构建为算法，而是构建成一种机器。感知机是在名为 Mark I 感知机的硬件中实现的。Mark I 感知机是一台纯电动机器。它有 400 个光电管（或光电探测器），其权重被编码到电位器中，权重更新（发生在反向传播中）由电动机执行。下图是 Mark I 感知机。

![](https://image.cubox.pro/cardImg/2023102119093389880/67432.jpg?imageMogr2/quality/90/ignore-error/1)

_Mark I 感知机。图片来自美国国家历史博物馆_

就像你今天在新闻中看到的关于神经网络的内容一样，感知机也是当时的头条新闻。《纽约时报》报道说，“\[海军\] 期望电子计算机的初步模型能够行走、说话、观察、书写、自我复制并意识到它的存在”。今天，我们都知道机器仍然难以行走、说话、观察、书写、复制自己，而意识则是另一回事。

Mark I 感知机的目标仅仅是识别图像，而当时它只能识别两个类别。人们花了一些时间才知道添加更多层（感知机是单层神经网络）可以使网络具有学习复杂功能的能力。这进一步产生了多层感知机 (MLP)。

#### **1982~1986 : 循环神经网络 (RNN)**

在多层感知机显示出解决图像识别问题的潜力之后，人们开始思考如何对文本等序列数据进行建模。

循环神经网络是一类旨在处理序列的神经网络。与多层感知机 (MLP) 等前馈网络不同，RNN 有一个内部反馈回路，负责记住每个时间步的信息状态。

![](https://image.cubox.pro/cardImg/2023102119093344397/37419.jpg?imageMogr2/quality/90/ignore-error/1)

_前馈网络与循环神经网络_

第一种 RNN 单元在 1982 年到 1986 年之间被发现，但它并没有引起人们的注意，因为简单的 RNN 单元在用于长序列时会受到很大影响，主要是由于记忆力短和梯度不稳定的问题。

#### **1998：LeNet\-5：第一个卷积神经网络架构**

LeNet\-5 是最早的卷积网络架构之一，于 1998 年用于文档识别。LeNet\-5 由 3 个部分组成：2 个卷积层、2 个子采样或池化层和 3 个全连接层。卷积层中没有激活函数。

正如论文所说，LeNet\-5 已进行商业化部署，每天读取数百万张支票。下面是 LeNet\-5 的架构。该图像取自其原始论文。

![](https://image.cubox.pro/cardImg/2023102119093430188/36763.jpg?imageMogr2/quality/90/ignore-error/1)

LeNet\-5 在当时确实是一个有影响力的东西，但它（常规的卷积网络）直到 20 年后才受到关注！LeNet\-5 建立在早期工作的基础上，例如福岛邦彦提出的第一个卷积神经网络、反向传播（Hinton 等人，1986 年）和应用于手写邮政编码识别的反向传播（LeCun 等人，1989 年）。

#### **1998：长短期记忆（LSTM）**

由于梯度不稳定的问题，简单 RNN 单元无法处理长序列问题。LSTM 是可用于处理长序列的 RNN 版本。LSTM 基本上是 RNN 单元的极端情况。

LSTM 单元的一个特殊设计差异是它有一个门机制，这是它可以控制多个时间步长的信息流的基础。

简而言之，LSTM 使用门来控制从当前时间步到下一个时间步的信息流，有以下 4 种方式：

*   输入门识别输入序列。
    
*   遗忘门去掉输入序列中包含的所有不相关信息，并将相关信息存储在长期记忆中。
    
*   LTSM 单元更新更新单元的状态值。
    
*   输出门控制必须发送到下一个时间步的信息。
    

![](https://image.cubox.pro/cardImg/2023102119093533331/40033.jpg?imageMogr2/quality/90/ignore-error/1)

_LSTM 架构。图片取自 MIT 的课程《6.S191 Introduction to Deep Learning》_

LSTM 处理长序列的能力使其成为适合各种序列任务的神经网络架构，例如文本分类、情感分析、语音识别、图像标题生成和机器翻译。

LSTM 是一种强大的架构，但它的计算成本很高。2014 年推出的 GRU（Gated Recurrent Unit）可以解决这个问题。与 LSTM 相比，它的参数更少，而且效果也很好。

#### **2012 年：ImageNet 挑战赛、AlexNet 和 ConvNet 的兴起**

如果跳过 ImageNet 大规模视觉识别挑战赛 (ILSVRC) 和 AlexNet，就几乎不可能讨论神经网络和深度学习的历史。

ImageNet 挑战赛的唯一目标是评估大型数据集上的图像分类和对象分类架构。它带来了许多新的、强大的、有趣的视觉架构，我们将简要回顾这些架构。

挑战赛始于 2010 年，但在 2012 年发生了变化，AlexNet 以 15.3% 的 Top 5 低错误率赢得了挑战，这几乎是之前获胜者错误率的一半。AlexNet 由 5 个卷积层、随后的最大池化层、3 个全连接层和一个 softmax 层组成。AlexNet 提出了深度卷积神经网络可以很好地处理视觉识别任务的想法。但当时，这个观点还没有深入到其他应用上！

在随后的几年里，ConvNets 架构不断变得更大并且工作得更好。例如，有 19 层的 VGG 以 7.3% 的错误率赢得了挑战。GoogLeNet(Inception-v1) 更进一步，将错误率降低到 6.7%。2015 年，ResNet（Deep Residual Networks）扩展了这一点，并将错误率降低到 3.6%，并表明通过残差连接，我们可以训练更深的网络（超过 100 层），在此之前，训练如此深的网络是不可能的。之前人们发现更深层次的网络工作得更好，这导致了其他新架构，如 ResNeXt、Inception-ResNet、DenseNet、Xception 等。

读者可以在这里找到这些架构和其他现代架构的总结和实现：https://github.com/Nyandwi/ModernConvNets

![](https://image.cubox.pro/cardImg/2023102119093520758/66981.jpg?imageMogr2/quality/90/ignore-error/1)

_ModernConvNets 库。_

![](https://image.cubox.pro/cardImg/2023102119093629677/30317.jpg?imageMogr2/quality/90/ignore-error/1)

_ImageNet 挑战赛。图片来自课程《 CS231n》_

#### **2014 年 : 深度生成网络**

生成网络用于从训练数据中生成或合成新的数据样本，例如图像和音乐。

生成网络有很多种类型，但最流行的类型是由 Ian Goodfellow 在 2014 年创建的生成对抗网络 (GAN)。GAN 由两个主要组件组成：生成假样本的生成器和区分真实样本和生成器生成样本的判别器。生成器和鉴别器可以说是互相竞争的关系。他们都是独立训练的，在训练过程中，他们玩的是零和游戏。生成器不断生成欺骗判别器的假样本，而判别器则努力发现那些假样本（参考真实样本）。在每次训练迭代中，生成器在生成接近真实的假样本方面做得更好，判别器必须提高标准来区分不真实的样本和真实样本。

GAN 一直是深度学习社区中最热门的事物之一，该社区以生成伪造的图像和 Deepfake 视频而闻名。如果读者对 GAN 的最新进展感兴趣，可以阅读 StyleGAN2、DualStyleGAN、ArcaneGAN 和 AnimeGANv2 的简介。如需 GAN 资源的完整列表，请查看 Awesome GAN 库：https://github.com/nashory/gans-awesome-applications。下图说明了 GAN 的模型架构。

![](https://image.cubox.pro/cardImg/2023102119093695465/88852.jpg?imageMogr2/quality/90/ignore-error/1)

_生成对抗网络（GAN）_

GAN 是生成模型的一种。其他流行的生成模型类型还有 Variation Autoencoder (变分自编码器，VAE)、AutoEncoder （自编码器）和扩散模型等。

#### **2017 年：Transformers 和注意力机制**

时间来到 2017 年。ImageNet 挑战赛结束了。新的卷积网络架构也被制作出来。计算机视觉社区的每个人都对当前的进展感到高兴。核心计算机视觉任务（图像分类、目标检测、图像分割）不再像以前那样复杂。人们可以使用 GAN 生成逼真的图像。NLP 似乎落后了。但是随后出现了一些事情，并且在整个网络上都成为了头条新闻：一种完全基于注意力机制的新神经网络架构横空出世。并且 NLP 再次受到启发，在随后的几年，注意力机制继续主导其他方向（最显著的是视觉）。该架构被称为 Transformer 。

在此之后的 5 年，也就是现在，我们在这里谈论一下这个最大的创新成果。Transformer 是一类纯粹基于注意力机制的神经网络算法。Transformer 不使用循环网络或卷积。它由多头注意力、残差连接、层归一化、全连接层和位置编码组成，用于保留数据中的序列顺序。下图说明了 Transformer 架构。

![](https://image.cubox.pro/cardImg/2023102119093715342/46695.jpg?imageMogr2/quality/90/ignore-error/1)

_图片来自于《Attention Is All You Need》_

Transformer 彻底改变了 NLP，目前它也在改变着计算机视觉领域。在 NLP 中，它被用于机器翻译、文本摘要、语音识别、文本补全、文档搜索等。

读者可以在其论文 《Attention is All You Need》 中了解有关 Transformer 的更多信息。

#### **2018 年至今**

自 2017 年以来，深度学习算法、应用和技术突飞猛进。为了清楚起见，后来的介绍是按类别划分的。在每个类别中，我们都会重新审视主要趋势和一些最重要的突破。

**Vision Transformers**

Transformer 在 NLP 中表现出优异的性能后不久，一些勇于创新的人就迫不及待地将注意力机制放到了图像上。在论文《[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650798990&idx=1&sn=a38ee3e1b9260b37a332c94ba5ebdff8&scene=21#wechat_redirect)》中，谷歌的几位研究人员表明，对直接在图像块序列上运行的正常 Transformer 进行轻微修改可以在图像分类数据集上产生实质性的结果。他们将他们的架构称为 Vision Transformer (ViT)，它在大多数计算机视觉基准测试中都有体现（在作者撰写本文时，ViT 是 Cifar-10 上最先进的分类模型）。

ViT 设计师并不是第一个尝试在识别任务中使用注意力机制的人。我们可以在论文 Attention Augmented Convolutional Networks 中找到第一个使用的记录，这篇论文试图结合自注意力机制和卷积（摆脱卷积主要是由于 CNN 引入的空间归纳偏置）。另一个例子见于论文《Visual Transformers: Token-based Image Representation and Processing for Computer Vision，这篇论文在基于滤波器的 token 或视觉 token 上运行 Transformer。这两篇论文和许多其他未在此处列出的论文突破了一些基线架构（主要是 ResNet）的界限，但当时并没有超越当前的基准。ViT 确实是最伟大的论文之一。这篇论文最重要的见解之一是 ViT 设计师实际上使用图像 patch 作为输入表示。他们对 Transformer 架构没有太大的改变。

![](https://image.cubox.pro/cardImg/2023102119093825566/56812.jpg?imageMogr2/quality/90/ignore-error/1)

_Vision Transformer(ViT)_

除了使用图像 patch 之外，使 Vision Transformer 成为强大架构的结构是 Transformer 的超强并行性及其缩放行为。但就像生活中的一切一样，没有什么是完美的。一开始，ViT 在视觉下游任务（目标检测和分割）上表现不佳。

在引入 [Swin Transformers ](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650813346&idx=3&sn=6c71dae96eb11c353de04fa531422084&scene=21#wechat_redirect)之后，Vision Transformer 开始被用作目标检测和图像分割等视觉下游任务的骨干网络。Swin Transformer 超强性能的核心亮点是由于在连续的自注意力层之间使用了移位窗口。下图描述了 Swin Transformer 和 Vision Transformer (ViT) 在构建分层特征图方面的区别。

![](https://image.cubox.pro/cardImg/2023102119093997216/24881.jpg?imageMogr2/quality/90/ignore-error/1)

_图片来自 Swin Transformer 原文_

Vision Transformer 一直是近来最令人兴奋的研究领域之一。我们可以在这里讨论许多 Vision Transformers 论文，但读者可以在论文《Transformers in Vision: A Survey》中了解更多信息。其他最新视觉 Transformer 还有 CrossViT、ConViT 和 SepViT 等。

**视觉和语言模型**

视觉和语言模型通常被称为多模态。它们是涉及视觉和语言的模型，例如文本到图像生成（给定文本，生成与文本描述匹配的图像）、图像字幕（给定图像，生成其描述）和视觉问答（给定一个图像和关于图像中内容的问题，生成答案）。Transformer 在视觉和语言领域的成功很大程度上促成了多模型作为一个单一的统一网络。

实际上，所有视觉和语言任务都利用了预训练技术。在计算机视觉中，预训练需要对在大型数据集（通常是 ImageNet）上训练的网络进行微调，而在 NLP 中，往往是对预训练的 BERT 进行微调。要了解有关 V-L 任务中预训练的更多信息，请阅读论文《A Survey of Vision-Language Pre-Trained Models》。有关视觉和语言任务、数据集的一般概述，请查看论文《Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods》。

前段时间，OpenAI 发布了 [DALL·E 2](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650842587&idx=2&sn=f6043375215d97793d5669c7436a16e7&scene=21#wechat_redirect)（改进后的 DALL·E），这是一种可以根据文本生成逼真图像的视觉语言模型。现有的文本转图像模型有很多，但 DALL·E 2 的分辨率、图像标题匹配和真实感都相当出色。

DALL·E 2 尚未对公众开放，但你可以加入候补名单。以下是 DALL·E 2 创建的一些图像示例。

![](https://image.cubox.pro/cardImg/2023102119094071068/13946.jpg?imageMogr2/quality/90/ignore-error/1)

上面呈现的 DALL·E 2 生成的图像取自一些 OpenAI 员工，例如 @sama、@ilyasut、@model\_mechanic 和 openaidalle。

**大规模语言模型 (LLM)**

语言模型有多种用途。它们可用于预测句子中的下一个单词或字符、总结一段文档、将给定文本从一种语言翻译成另一种语言、识别语音或将一段文本转换为语音。

开玩笑地说，发明 Transformers 的人必须为语言模型在朝着大规模参数化方向前进受到指责（但实际上没有人应该受到责备，Transformers 是 2010 年代十年中最伟大的发明之一，大模型令人震惊的地方在于：如果给定足够的数据和计算，它总能更好地工作）。在过去的 5 年中，语言模型的大小一直在不断增长。

在引入论文《Attention is all you need》一年后，大规模语言模型开始出现。2018 年，OpenAI 发布了 GPT（Generative Pre-trained Transformer），这是当时最大的语言模型之一。一年后，OpenAI 发布了 GPT-2，一个拥有 15 亿个参数的模型。又一年后，他们发布了 GPT-3，它有 1750 亿个参数。GPT-3 用了 570GB 的 文本来训练。这个模型有 175B 的参数，模型有 700GB 大。根据 lambdalabs 的说法，如果使用在市场上价格最低的 GPU 云，训练它需要 366 年，花费 460 万美元！

GPT-n 系列型号仅仅是个开始。还有其他更大的模型接近甚至比 GPT-3 更大。如：NVIDIA Megatron-LM 有 8.3B 参数。最新的 DeepMind Gopher 有 280B 参数。2022 年 4 月 12 日，DeepMind 发布了另一个名为 Chinchilla 的 70B 语言模型，尽管比 Gopher、GPT-3 和 Megatron-Turing NLG（530B 参数）小，但它的性能优于许多语言模型。Chinchilla 的论文表明，现有的语言模型是训练不足的，具体来说，它表明通过将模型的大小加倍，数据也应该加倍。但是，几乎在同一周内又出现了具有 5400 亿个参数的 Google Pathways 语言模型（PaLM）！

![](https://image.cubox.pro/cardImg/2023102119094196457/10916.jpg?imageMogr2/quality/90/ignore-error/1)

_Chinchilla 语言模型_

**代码生成模型**

代码生成是一项涉及补全给定代码或根据自然语言或文本生成代码的任务，或者简单地说，它是可以编写计算机程序的人工智能系统。可以猜到，现代代码生成器是基于 Transformer 的。

我们可以确定地说，人们已经开始考虑让计算机编写自己的程序了（就像我们梦想教计算机做的所有其他事情一样），但代码生成器在 OpenAI 发布 Codex 后受到关注。Codex 是在 GitHub 公共仓库和其他公共源代码上微调的 GPT-3。OpenAI 表示：“OpenAI Codex 是一种通用编程模型，这意味着它基本上可以应用于任何编程任务（尽管结果可能会有所不同）。我们已经成功地将它用于编译、解释代码和重构代码。但我们知道，我们只触及了可以做的事情的皮毛。” 目前，由 Codex 支持的 [GitHub Copilot](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650819923&idx=1&sn=45764d880cb7ce30a403d40290fd0387&scene=21#wechat_redirect) 扮演着结对程序员的角色。

在我使用 Copilot 后，我对它的功能感到非常惊讶。作为不编写 Java 程序的人，我用它来准备我的移动应用程序（使用 Java）考试。人工智能帮助我准备学术考试真是太酷了！

在 OpenAI 发布 Codex 几个月后，DeepMind 发布了 [AlphaCode](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650837433&idx=1&sn=08f9979eda2a9ecfdc8e52a47f5a0195&scene=21#wechat_redirect)，这是一种基于 Transformer 的语言模型，可以解决编程竞赛问题。AlphaCode 发布的博文称：“AlphaCode 通过解决需要结合批判性思维、逻辑、算法、编码和自然语言理解的新问题，在编程竞赛的参与者中估计排名前 54%。” 解决编程问题（或一般的竞争性编程）非常困难（每个做过技术面试的人都同意这一点），正如 Dzmitry 所说，击败 “人类水平仍然遥遥无期”。

前不久，来自 Meta AI 的科学家发布了 InCoder，这是一种可以生成和编辑程序的生成模型。

更多关于代码生成的论文和模型可以在这里找到：https://paperswithcode.com/task/code-generation/codeless

**再次回到感知机**

在卷积神经网络和 Transformer 兴起之前的很长一段时间里，深度学习都围绕着感知机展开。ConvNets 在取代 MLP 的各种识别任务中表现出优异的性能。视觉 Transformer 目前也展示出似乎是一个很有前途的架构。但是感知机完全死了吗？答案可能不是。

在 2021 年 7 月，两篇基于感知机的论文被发表。一个是 [MLP-Mixer: An all-MLP Architecture for Vision](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650814947&idx=1&sn=7cce32919afc573f1ddc8090ca90a74d&scene=21#wechat_redirect)，另一个是 [Pay Attention to MLPs(gMLP) ](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650816385&idx=1&sn=9374d47f39c972894d7c4f4e3f75e9f9&scene=21#wechat_redirect).

MLP-Mixer 声称卷积和注意力都不是必需的。这篇论文仅使用多层感知机 (MLP)，就在图像分类数据集上取得了很高的准确性。MLP-Mixer 的一个重要亮点是它包含两个主要的 MLP 层：一个独立应用于图像块（通道混合），另一个层跨块应用（空间混合）。

gMLP 还表明，通过避免使用自注意和卷积（当前 NLP 和 CV 的实际使用的方式），可以在不同的图像识别和 NLP 任务中实现很高的准确性。

![](https://image.cubox.pro/cardImg/2023102119094111264/43966.jpg?imageMogr2/quality/90/ignore-error/1)

读者显然不会使用 MLP 去获得最先进的性能，但它们与最先进的深度网络的可比性却是令人着迷的。

**再次使用卷积网络：2020 年代的卷积网络**

自 Vision Transformer（2020 年）推出以来，计算机视觉的研究围绕着 Transformer 展开（在 NLP 中，transformer 已经是一种规范）。Vision Transformer (ViT) 在图像分类方面取得了最先进的结果，但在视觉下游任务（对象检测和分割）中效果不佳。随着 Swin Transformers 的推出， Vision Transformer 很快也接管了视觉下游任务。

很多人（包括我自己）都喜欢卷积神经网络。卷积神经网络确实能起效，而且放弃已经被证明有效的东西是很难的。这种对深度网络模型结构的热爱让一些杰出的科学家回到过去，研究如何使卷积神经网络（准确地说是 ResNet）现代化，使其具有和 Vision Transformer 同样的吸引人的特征。特别是，他们探讨了「Transformers 中的设计决策如何影响卷积神经网络的性能？」这个问题。他们想把那些塑造了 Transformer 的秘诀应用到 ResNet 上。

Meta AI 的 Saining Xie 和他的同事们采用了他们在论文中明确陈述的路线图，最终形成了一个名为 [ConvNeXt](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650835782&idx=1&sn=7cbf78a35c624f0ff4c76b1760fed68f&scene=21#wechat_redirect) 的 ConvNet 架构。ConvNeXt 在不同的基准测试中取得了可与 Swin Transformer 相媲美的结果。读者可以通过 ModernConvNets 库（现代 CNN 架构的总结和实现）了解更多关于他们采用的路线图。

#### **结论**

深度学习是一个非常有活力、非常宽广的领域。很难概括其中所发生的一切，作者只触及了表面，论文多到一个人读不完。很难跟踪所有内容。例如，我们没有讨论强化学习和深度学习算法，如 AlphaGo、蛋白质折叠 AlphaFold（这是最大的科学突破之一）、深度学习框架的演变（如 TensorFlow 和 PyTorch），以及深度学习硬件。或许，还有其他重要的事情构成了我们没有讨论过的深度学习历史、算法和应用程序的很大一部分。

作为一个小小的免责声明，读者可能已经注意到，作者偏向于计算机视觉的深度学习。可能还有其他专门为 NLP 设计的重要深度学习技术作者没有涉及。

此外，很难确切地知道某项特定技术是什么时候发表的，或者是谁最先发表的，因为大多数奇特的东西往往受到以前作品的启发。如有纰漏，读者可以去原文评论区与作者讨论。

_原文链接：https://www.getrevue.co/profile/deeprevision/issues/a-revised-history-of-deep-learning-issue-1-1145664_

[跳转到 Cubox 查看](https://cubox.pro/my/card?id=7115363917827670534)