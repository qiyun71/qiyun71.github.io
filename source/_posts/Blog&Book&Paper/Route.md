Life:

(数字孪生)通过有限元/数学建模，在计算机上(虚拟世界中)模拟真实世界的一些规律/现象，尽可能的与实际物体（实验测量）有相同的物理属性
- 有限元分析将**构造实体几何CAD模型**通过网格划分变为**多边形网格Mesh**([3D Model](../3DReconstruction/3D%20Model.md))，然后进行求解，得到FRFs/模态频率(动力学)。
- 对实际物体工作过程中的测量依靠各种传感器完成，eg:[图像传感器](https://zh.wikipedia.org/wiki/%E5%9B%BE%E5%83%8F%E4%BC%A0%E6%84%9F%E5%99%A8)-->照片，[压电式传感器](https://baike.baidu.com/item/%E5%8E%8B%E7%94%B5%E5%BC%8F%E4%BC%A0%E6%84%9F%E5%99%A8/8835700)-->振动(d/v/a)...

仿真 [CAX](../Other%20Interest/CAX.md)
- 原理 [Learn-FEA](../Learn/Finite%20Element/Learn-FEA.md)
  - [Structural Dynamics](../Learn/Finite%20Element/Structural%20Dynamics.md)
    - [Mechanical Vibrations Theory and Application to Structural Dynamics](Read/Book/Mechanical%20Vibrations%20Theory%20and%20Application%20to%20Structural%20Dynamics.md)
- 软件操作
  - 网格划分 Hypermesh
  - [Pa&Nastran](../Learn/Finite%20Element/Pa&Nastran.md)
  - [Abaqus](../Learn/Finite%20Element/Abaqus.md)
  - 结合编程语言二次开发
    - [PyNastran](../Project/PyNastran.md)

实验 [Vibration Measurement](../Other%20Interest/Vibration%20Measurement.md)
- Modal Testing (LMS)
  - [模态试验实用技术.excalidraw](Read/Book/模态实验/模态试验实用技术.excalidraw.md)  --> [Modal Testing](../Learn/Finite%20Element/Modal%20Testing.md) 原理公式
  - [模态实验.excalidraw](Read/Book/模态实验/模态实验.excalidraw.md) 实验设备/软件
- [(WX) Dynamics experiment](Read/Interest%20Papers/(WX)%20Dynamics%20experiment.md)

---

***Must***： [Math](../Learn/Math/Math.md) 
Other Learn：
- [ ] [Book Lists](Read/Book%20Lists.md)
  - [ ] [How to Read a Paper](Read/Book/How%20to%20Read%20a%20Paper.md)
  - [ ] [Writing Science](Read/Book/Writing%20Science.md)
- [ ] 语言学习
  - [ ] [Japanse](Read/Book/Japanse.md)
  - [ ] [Everyone-can-use-English](Read/Book/Everyone-can-use-English.md)
  - [ ] [Write tips](Write/Write%20tips.md) paper写作工具/语言

---

研究兴趣/方向 (Master, Phd or Life) [Interest.excalidraw](Write/Interest.excalidraw.md)

Research Interest：
- [Uncertainty](../Other%20Interest/Uncertainty.md)
  - [(BSF)Uncertainty](Read/Interest%20Papers/(BSF)Uncertainty.md)
- [Reliability](../Other%20Interest/Reliability.md)
  - [核领域可靠性](Read/Interest%20Papers/核领域可靠性.md)
- [PHM](../Other%20Interest/PHM.md)
- Model Updating
  - [ModelUpdating.excalidraw](Write/Write%20Paper/Model%20Updating/ModelUpdating.excalidraw.md) 思考 --> Write Paper:
    - [Stochastic Model Calibration with Image Encoding——Converting High-Dimensional Frequency-Domain Responses into RGB Images for Neural Network Inversion](Write/Write%20Paper/Model%20Updating/Stochastic%20Model%20Calibration%20with%20Image%20Encoding——Converting%20High-Dimensional%20Frequency-Domain%20Responses%20into%20RGB%20Images%20for%20Neural%20Network%20Inversion.md)
    - [A fast interval model updating method based on MLP neural network](Write/Write%20Paper/Model%20Updating/A%20fast%20interval%20model%20updating%20method%20based%20on%20MLP%20neural%20network.md)
    - [Self-superviside Model Updating through inverse calibration model](Write/Write%20Paper/Model%20Updating/Self-superviside%20Model%20Updating%20through%20inverse%20calibration%20model.md)
  - [Basics about ModelUpdating](../ModelUpdating/Basics%20about%20ModelUpdating.md)
  - [Paper about ModelUpdating](../ModelUpdating/Paper%20about%20ModelUpdating.md)
  - [Case about ModelUpdating](../ModelUpdating/Case%20about%20ModelUpdating.md)
    - [折叠翼](Read/Interest%20Papers/折叠翼.md)
- 3D Reconstruction
  - [Basics about 3D Reconstruction](../3DReconstruction/Basics%20about%203D%20Reconstruction.md)
  - [Paper About 3D Reconstruction](../3DReconstruction/Paper%20About%203D%20Reconstruction.md)
  - [NeuS-based 3D Reconstruction](../3DReconstruction/NeuS-based%203D%20Reconstruction.md) For 硕士毕业论文 [Master Paper](Write/Write%20Paper/3D%20Reconstruction/Master%20Paper.md)
  - [Code of Multi-view 3D Reconstruction based on SDF and volume rendering](../3DReconstruction/Code%20of%20Multi-view%203D%20Reconstruction%20based%20on%20SDF%20and%20volume%20rendering.md)
  - [Datasets](../3DReconstruction/Datasets.md)
  - Practical
    - [Finite Element Model 3D Reconstruction](../3DReconstruction/Practical/Finite%20Element%20Model%203D%20Reconstruction.md) **有限元模型重建**
    - [Multi-view Human Body Reconstruction](../3DReconstruction/Practical/Multi-view%20Human%20Body%20Reconstruction.md)
    - [Dimensions  Measurement](../3DReconstruction/Practical/Dimensions%20%20Measurement.md) 测量物体的尺寸
    - [Anime Image 3D Reconstruction](../3DReconstruction/Practical/Anime%20Image%203D%20Reconstruction.md)
- Others
  - [Time Series Data](../Other%20Interest/Time%20Series%20Data.md) 时间序列数据 --> 回归/分类
  - [Small Sample Learning](../Other%20Interest/Small%20Sample%20Learning.md) 小子样问题

---

编程语言
- [Learn-Python](../Learn/Python/Learn-Python.md)
  - [Figure](../Learn/Python/Figure.md)
  - [Data Analysis](../Learn/Python/Data%20Analysis.md)
  - [Algorithm](../Learn/Python/Algorithm.md)
  - [Linux OS](../Learn/Python/Linux%20OS.md)
- [Learn-Matlab](../Learn/Other%20Interest/Learn-Matlab.md)
- [Learn-Csharp](../Learn/Other%20Interest/Learn-Csharp.md)
- [Learn-Cpp](../Learn/Other%20Interest/Learn-Cpp.md)
- [Learn-Rust](../Learn/Other%20Interest/Learn-Rust.md)

神经网络
- [DeepLearning](../Learn/Neural%20Network/DeepLearning.md)
  - [Loss Functions](../Learn/Neural%20Network/Loss%20Functions.md)
  - [PyTorch](../Learn/Neural%20Network/PyTorch.md)


---

兴趣：
- 硬件 
  - [Amazing Machine](Write/Blog/Amazing%20Machine.md) 
  - [Computer](Write/Source/Computer.md)
  - [Six Legged Spider Robot](../Project/Six%20Legged%20Spider%20Robot.md)
- 软件 
  - [Tools](Write/Source/Tools.md) 
  - [IPV6](Write/Source/IPV6.md) 
  - [Unpack Snowbreak File](../Project/Unpack%20Snowbreak%20File.md)
  - [Visualize Interval uncertainty quantification metrics](../Project/Visualize%20Interval%20uncertainty%20quantification%20metrics.md)

---

人生 [Life](Write/Life.md)
世界 [World](Write/World.md) 
- [Travel to Japan 2025](Write/Blog/Travel%20to%20Japan%202025.md)
价值 [Blockchain](Read/Blockchain.md)



---

职位：

[【可靠性工程师-理论方向招聘】_上海宇量昇科技有限公司招聘信息-猎聘](https://www.liepin.com/job/1972290995.shtml)

"岗位介绍： 
1、 应用可靠性工程方法（如FMEA、FTA、可靠性建模）识别潜在风险并提出改进方案。 
2、制定可靠性测试计划（如寿命测试、加速老化测试、环境应力筛选），并负责专业测试台架搭建。 
3、对现场故障数据进行根因分析，提出纠正措施。 
4、制定企业内部的可靠性工程规范及流程、行业可靠性标准。 
5、与研发、生产、质量、售后团队协作，推动可靠性目标落地。 
任职要求： 1、硕士及以上学历，主修可靠性工程、机械工程、电子工程、材料科学等相关专业。 2、精通可靠性分析方法（FMEA、FTA、FRACAS）。 3、掌握数据分析工具（Reliasoft、Minitab、JMP）及相关可靠性测试设备。 4、5年以上可靠性工程或质量工程相关经验，复杂系统可靠性经验优先，博士可放宽。 5、沟通协调能力强，能推动跨部门合作。"