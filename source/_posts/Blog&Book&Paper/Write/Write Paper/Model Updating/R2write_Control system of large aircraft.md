
## 项目的立项依据

### 立项背景及必要性

*技术难题：*
*多重不确定因素耦合下系统容错控制难*
*恶劣环境下多重耦合故障的实时精确诊断难*
*“人-机-环”多源混合不确定性的传播与量化难*
*高维响应下非线性气动参数快速精确辨识难*

1. 人-机-环多源混合不确定性统一表征与量化建模难
2. 不确定性影响下非线性气动参数实时高精度高效率与高鲁棒辨识难
3. 恶劣环境下人-机-环耦合及人因主导的多重故障叠加预测难
4. 多重耦合故障下大飞机控制系统实时自重构与性能容错难

大型民用客机因其载客量大、航程远、飞行速度快等特征，在现代全球交通体系中扮演着核心角色，在支撑远程跨洲际运输与促进区域经济互联方面发挥着关键作用。在全球航空竞争格局中，波音（Boeing）与空客（Airbus）长期引领大型客机控制技术的迭代方向，其主力机型如波音787、空客A350已全面应用高度集成的全电传操纵系统，并正加速向智能化、高自主化演进。例如，空客公司于2023年通过其Dragonfly项目，在A350演示机上成功验证了基于人工智能的自主紧急备降、自主着陆及地面滑行辅助技术，旨在极端工况或飞行员失能情况下通过智能决策提升整机的生存能力；波音公司则在2025年启动的ecoDemonstrator计划中，密集测试新一代数字化通信与飞行路径优化系统，以应对日益严苛的航行安全与效率要求。我国大型客机事业虽起步较晚，但已取得历史性突破，C919客机自2023年5月正式投入商业运营以来，截至2025年5月已累计执行商业航班超1.1万班，承运旅客突破200万人次，标志着国产大飞机已进入规模化运营的新阶段。与此同时，C929远程宽体客机的研制工作正全面展开，其对**飞行控制系统的安全性、可靠性及智能化水平**提出了更高要求。随着国产大飞机逐步进入高频次、多航线、复杂气象环境的商业运营序列，如何确保系统在**各类不确定因素干扰下**的飞行安全，已成为我国航空工业实现自主可控与高质量发展的首要任务。

尽管现代民航通过严苛的安全标准维持着极低的事故率，但在**极端运行环境、关键设备偶发故障与高频次人机交互深度耦合**的复杂工况下，依然存在着重大安全风险。近年来，多起典型航空事故揭示了系统内部故障与外部扰动在特定工况下相互叠加并引发级联失效的深层演化机理。例如，2018年与2019年发生的波音737 MAX系列事故，其核心诱因在于单一迎角传感器故障触发了存在逻辑缺陷的机动特性增强系统，导致控制律对飞行员的正向操纵产生持续抑制，在故障传播过程中演变为整机失控；2019年阿特拉斯航空3591号航班坠毁事故中，则是由于在恶劣气象条件与复飞模式非预期激活的共同作用下，最终飞行员产生的空间定向障碍导致了严重的错误操作；2024年3月拉塔姆航空800号班机在巡航阶段的突发骤降，进一步展示了驾驶舱人机交互干扰诱发飞行控制律异常响应的高度敏感性。据国际航空运输协会及波音安全报告统计，进近与着陆阶段的时长虽仅占总航程的1%左右，但其涉及的致命事故比例却高达37%，且大多呈现出明显的多因素耦合特征。上述案例表明，现有的控制系统在应对**非预期扰动、硬件设备异常及人机操纵冲突耦合并发**这类恶劣条件时，仍难以有效保障飞机安全。

大型民用客机飞行控制系统作为整机姿态调节与轨迹跟踪的决策中枢，在维持飞行稳定性及保障航行安全方面发挥着决定性作用。现代大飞机普遍采用数字电传操纵系统，其核心架构通过高可靠、多余度的飞行控制计算机，将飞行员的操纵输入或自动飞行系统的决策指令，结合实时反馈的飞行状态参数进行综合运算，进而驱动各类执行机构动作。如图 1所示，该系统是一个典型的高维、非线性、强耦合的多输入多输出复杂系统，其运行深度依赖于多源感知信息的实时交换与多执行机构的协调配合。控制系统不仅承担着基础的飞行任务，还内嵌了严密的飞行安全包线保护逻辑，旨在通过实时监控飞行姿态与运动包线，防止飞机进入失速、超速或极端过载等危险状态。

然而随着大型民用客机实际航行环境的日益复杂，其控制系统的性能受制于由“人-机-环（Man-Machine-Environment, MME）”三个维度交织构成的多源混合不确定性**。根据不确定性的来源与表现性质，可划分为由建模认知局限导致的认知不确定性以及由随机扰动引发的随机不确定性。在机体维度，大型客机因其复杂的柔性结构特征，在跨声速或大迎角等复杂包线内，气动衍生参数受马赫数与雷诺数耦合影响呈现出明显的非线性漂移，且传感器量测噪声与执行器在长期服役中存在的性能退化、速率限幅进一步加剧了控制效果的降低；在环境维度，突发风切变、微下击暴流以及大气温度与密度的骤变等随机不确定因素会瞬时改变飞行器的升力特性与操纵效能，导致系统状态变量出现非预期突变；在人机交互维度，飞行员在处理突发工况时的认知偏差或生理迟滞会引入高度不确定的随机操纵输入，不仅干扰了控制律的正常解算，甚至导致飞行员诱发振荡的出现。

在上述多源混合不确定性的深度影响下，大型民用客机飞行控制系统所面临的风险往往由非预期的随机扰动进一步转变为复杂的物理层故障，呈现出显著的多重耦合特征。不同于常见的单点孤立故障，大飞机高度集成的系统架构使得传感器、执行器及气动舵面等关键部件的异常在恶劣工况下表现出极强的时空相关性与级联失效机理。例如，在复杂气象环境或剧烈大气扰动中，传感器的微小量测偏置可能在非线性控制律的传递下，诱发执行机构出现速率限幅或动力输出偏差，进而导致舵面效能的动态衰减与气动布局的非对称受损。这种多重耦合故障具有极强的隐蔽性与动态演变特征，其破坏力远超单一故障的简单线性叠加。尤其在恶劣飞行条件下多重耦合故障出现更为频繁，飞行控制系统难以实现高精度的故障辨识与实时的控制律重构，直接制约了整机在极端场景下的安全保障能力。

综合上述分析，针对大型民用客机在人-机-环多源混合不确定性及多重耦合故障下的安全保障困境，开展智能容错控制研究已成为攻克复杂系统控制瓶颈、提升国产大飞机极端工况下自主运行水平的必要路径。这一研究体系主要由四个核心环节构成：一是多源混合不确定性的量化**，旨在系统性识别并分析来自机体特性、飞行员操纵及外部环境的各种不确定性来源，建立具有高鲁棒性的量化数学模型，为控制算法提供坚实的理论基准；二是不确定参数的动态辨识，重点针对气动模型参数等关键不确定参数，考虑其随飞行工况及系统状态演化的特征，实现在强噪声干扰环境下的高精度在线模型修正，以精准捕捉飞行器动力学特性的异常漂移；三是多重耦合故障的实时监测与演化预测，通过对传感器、执行器等部件并发故障的深度识别与特征提取，模拟并预判故障在闭环系统内部的传播路径与后果演化，为后续决策预留充足的安全裕度；四是智能自重构容错控制系统设计，基于前述感知、辨识与预测结果，实现控制策略在极端受损工况下的自主重构与性能补偿，确保整机在非预期、非确定性场景下的稳定航行与安全包线维持。上述四个环节互为支撑、深度耦合，其实施过程时刻受到大飞机自身非线性特征、飞行员认知负载及复杂气象环境的动态制约，因此，相关的研究过程仍存在以下技术难题：

##### （1）人-机-环多源混合不确定性统一表征与量化建模难

在大飞机智能容错控制研究中，构建人-机-环多源混合不确定性量化模型，是实现高精度状态感知与控制律自主重构的基础前提。然而，实际飞行过程中的不确定性因素表现出极其显著的异构性与多维性，既涉及由建模认知局限导致的气动参数摄动、结构性能退化等认知不确定性，也涵盖由于大气湍流、传感器随机噪声等本质随机过程引发的随机不确定性。在大飞机复杂的航行场景下，飞行员的操纵规律、机载设备的物理状态以及外部气象环境的瞬时演化在时间尺度与空间分布上存在巨大差异，使得不同属性的不确定性要素难以在统一的数学框架下进行一致性描述。由此可知，多源不确定性的跨域耦合及其本质异构特征，导致当前难以建立一套能够全面覆盖认知与随机混合类型、具备高保真特性的统一不确定性量化建模体系，无法输出精准的系统状态感知结果，直接限制了后续故障辨识的鲁棒性与容错控制决策的有效性。

##### （2）不确定性影响下非线性气动参数实时高精度高效率与高鲁棒辨识难

在大飞机飞行过程中，实时、高精度地辨识气动模型参数等关键不确定参数，是准确捕捉飞行器动力学特性偏移、支撑系统状态评估与控制律重构的关键。然而，大型民用客机在跨声速、大迎角等极端飞行条件下，其气动特性表现出极强的非线性以及强烈的时空耦合特征，且传感器测量过程不可避免地受到机体结构振动、大气紊流以及电磁环境等复杂噪声干扰，引发观测数据质量下降，导致不确定参数的辨识极为困难。且由于大飞机对控制系统的实时性要求极高，而高精度辨识往往依赖于计算量巨大的高维非线性估计算法或复杂的迭代优化过程，这与机载飞行控制计算机有限的硬件算力资源形成了不可避免的技术冲突。 同时，气动参数存在明显的动态演化趋势，使得传统辨识算法难以在维持快速跟踪响应，在面对突发扰动或极端工况时极易出现辨识结果严重不准。由此可知，在强不确定性与高动态场景下，气动参数辨识过程面临着精度、效率与鲁棒性之间的矛盾，难以实现对时变非线性参数的快速精准修正，无法为智能容错控制决策提供可靠的模型依据。

##### （3）恶劣环境下人-机-环耦合及人因主导的多重故障叠加预测难

在大型民用客机遭遇突发异常状态时，实现对多重耦合故障的实时精准监测与演化态势预测，是提前预判系统失效风险、为智能决策提供关键前馈信息的核心。然而，在恶劣航行环境与系统状态退化的双重作用下，大飞机的故障特征往往表现出极强的隐蔽性以及显著的非线性级联特征，尤其是当传感器、执行器故障与恶劣气象扰动共同出现时，故障监测系统往往难以通过常规信号 特征识别出底层的多重并发模式。由于大飞机高度依赖人机共驾逻辑，飞行员在高压环境下的认知偏差或生理性滞后会引入具有高度随机性的控制律补偿动作，这种人因不确定性与机体故障在反馈回路中深度耦合，使得故障的传播路径呈现出极高的时空不确定性，传统的基于固定阈值的监测方法难以准确、快速提取故障特征，从而引发任务风险评估的严重滞后与预测结果的判断错误。由此可知，恶劣工况下人-机-环要素的复杂耦合特性，导致多重并发故障的动态规律难以解析，无法实现对系统级联失效后果的超前、精准预测，直接制约了控制系统对极端安全风险的防御与规避能力。

##### （4）多重耦合故障下大飞机控制系统实时自重构与性能容错难

在大型民用客机多重耦合故障的极端场景下，构建具备实时自重构能力的智能容错控制系统，是维系动力学稳定、保障安全着陆的最后防线。然而，当传感器偏置、执行器卡死及舵面受损等故障同时出现时，飞机的飞行安全包线急剧收缩，导致传统控制律完全失效。因此，迫切需要通过智能重构实现控制资源的动态最优分配，但该过程面临着严苛的实时性与人机耦合难题。首先，重构算法需在毫秒级硬实时约束下完成高维非线性解算，机载计算机有限的硬件算力难以支撑。其次，重构过程会本质性地改变飞行器的操纵特性，若控制反馈偏离飞行员的认知预期，可能导致物理层故障演变为灾难性的人机冲突。此外，受损机体呈现的高度非对称动力学特征，要求系统在极窄的剩余包线内精准兼顾稳定性与操纵品质，以最大限度降低机组的认知负荷。由此可知，飞行安全包线缩减、人机交互复杂性及实时响应要求，共同构成了大飞机在多重耦合故障下的容错控制难题。

### 国内外研究现状

#### 不确定性分析在大飞机控制系统研究现状

不确定性分析通过量化系统输入的变异性并评估其对系统输出的影响，从而保证系统的可靠性和鲁棒性，其在航空航天[^aircraft]、机械[^mechanical]和土木等多个领域[^civil]得到了广泛应用。在大飞机控制系统中，由于复杂的操作环境和多变的飞行条件，系统面临着诸多不确定性因素，不确定性分析变得至关重要。不确定性可以分为认知不确定性（Epistemic Uncertainty）和随机不确定性（Aleatory Uncertainty）。认知不确定性源自于对系统或环境的认知有限，例如传感器精度不足或模型简化带来的误差，通过增加信息或改进模型可以减少或消除；随机不确定性则源自于系统固有的随机性，如气象条件的变化或飞行载荷的波动。在大飞机控制系统中，不确定性来源包括但不限于气动模型过度简化、传感器测量误差、气动参数变化、飞行环境扰动以及操作人员行为等，这些不确定性因素共同影响着控制系统的性能和安全性。

不确定性分析主要包括不确定性量化（Uncertainty Quantification）和不确定性传播（Uncertainty Propagation）。不确定性量化旨在通过数学模型描述系统输入的不确定性特征，常用的方法包括概率方法、区间方法和混合方法。概率方法[^probabilistic]通过概率分布函数来描述输入参数的随机性，适用于具有充分统计数据的情况；区间方法[^interval]则使用上下界来表示参数的不确定范围，适用于缺乏统计信息但有明确界限的情况；混合方法[^pbox]结合了概率和区间方法的优点，能够更全面地表征复杂系统中的不确定性。为了推动本领域不确定性量化的发展，NASA兰利研究中心[^nasa]为飞行器控制和结构设计提出了不确定性量化挑战问题，其特点在于输入参数同时包含偶然性和认知性不确定性，观测数据稀少且不规则，要求在有限信息下实现高精度的正向传播与反向校准，具有重要的理论价值和工程意义。为此，Bi等[^pbox1]针对该类问题，开发了系统性的正向与反向不确定性处理方法，提出基于概率盒（Probability-box）的量化框架，统一表征偶然性和认知性不确定性，并通过解耦方法设计双循环流程分别传播两类不确定性。

不确定性传播旨在评估输入不确定性对系统输出的影响，最经典的方法是蒙特卡洛方法（Monte Carlo Method），其通过大量随机采样来生成输出分布[^mc]。例如闫国良等[^mc1]利用蒙特卡洛法实现了不同速度下失效概率的估算，揭示了控制机翼展弦长及刚度参数的期望值与变异性对降低颤振风险的重要性。Sankararaman等人[^mc2]采用概率方法来描述飞行计划与气象信息等输入参数的随机性，并应用蒙特卡洛等不确定性传播方法量化了多种不确定性源对飞机轨迹预测的综合影响。然而，蒙特卡洛方法计算量大，尤其在高维问题中，计算成本显著增加。为此，研究者们发展了多种改进采样方法，如马尔可夫链蒙特卡洛（Markov Chain Monte Carlo）[^tmcmc]、重要抽样（Importance Sampling）[^is]和子集抽样（Subset Simulation）[^ss]，以提高采样效率，同时保持了结果的统计准确性。

除了对采样技术的改进，代理模型方法通过构建简化的数学模型来近似复杂系统的行为，进一步降低不确定性分析的计算成本。主流代理模型方法包括响应面模型（Response Surface Model）[^rsm]、Kriging模型[^kriging]、多项式混沌展开（Polynomial Chaos Expansion, PCE）[^pce_zhao]以及数据驱动的代理模型。Tartaruga等人[^blindkriging]提出了一种基于奇异值分解和盲克里金代理模型的方法，用于民用飞机气动弹性模型相关载荷的预测与不确定性量化，在精确捕捉载荷包络线特征的同时显著降低了计算成本。Mota等人[^pce]采用概率方法来描述集合天气预报中风速预测的不确定性，并借助任意多项式混沌等代理模型方法进行不确定性传播，从而以高计算效率实现了对飞机间冲突概率的估计。Lazzara等[^lstm]提出了基于长短时记忆网络的代理模型，并结合自编码器降维实现了对飞机动力学高维时间响应的高效预测。这些方法的核心在于通过少量仿真数据构建高效的代替模型，在蒙特卡洛采样中快速生成大量输出样本，从而实现高效准确的不确定性传播。

综上所述，不确定性分析目前在航空航天领域已从传统的蒙特卡洛方法，逐步发展出多种高效的采样技术和代理模型方法，以应对复杂系统中多源不确定性的挑战。而大飞机控制系统作为典型的复杂工程系统，在恶劣环境和人因等多不确定性因素的影响下，其不确定性分析方法的研究与应用仍面临输入输出强非线性、人-机-环耦合下多源混合不确定性表征和跨机理传播复杂及人因不确定性量化评估困难等诸多挑战，亟需进一步探索和创新。

#### 大飞机气动参数辨识研究现状

飞机气动参数辨识是飞行控制系统设计与优化的关键环节，旨在通过分析飞行试验数据或仿真数据以估计飞机的气动特性参数，其准确性直接影响飞机的飞行性能和安全性。因此，发展高精度、高鲁棒性且适应复杂飞行条件的气动参数辨识方法，始终是航空航天领域的关键课题。纵观其发展历程，辨识方法经历了从完全依赖物理模型到基于数据驱动的方法，并逐渐发展到两者深度融合的阶段[^algorithm_nn]。

传统气动参数辨识主要建立在基于物理模型的参数估计方法之上。该方法以经典飞行动力学为框架，依托风洞试验与计算流体力学模拟建立参数化模型，进而采用系统辨识算法从数据中反演参数。经典的方法包括最小二乘法（Least Squares Method）[^LeastSquare]、极大似然法（Maximum Likelihood Method）[^MaximumLikelihood]及卡尔曼滤波（Kalman Filter）[^kalman]等，其在模型准确、状态线性的条件下理论完备且辨识结果可靠。此外还有以遗传算法（Genetic Algorithm）[^ga]、粒子群优化（Particle Swarm Optimization）[^pso]等智能优化算法为基础的参数辨识方法，这些方法通过全局搜索策略克服了传统优化方法易陷入局部最优的缺点，提升了辨识的鲁棒性和适应性。Raymundo等[^ga1]提出基于遗传算法的物理辨识方法，通过设计基于状态转移矩阵的适应度函数并约束参数搜索范围，辨识飞机纵向线性模型参数。Yang等[^pso1]提出了基于Sobol序列的融合自粒子群与蛇优化的算法，通过自适应权重与局部搜索机制增强全局寻优能力。汪清等[^srukf]通过将启动系数的时间导数建模为一阶 Gauss-Markov 过程，然后基于平方根无迹卡尔曼滤波器实现了飞行器气动参数的在线辨识。然而，随着飞行任务的复杂化和飞行环境的多变性，传统物理模型方法在处理非线性气动特性和高维响应时面临诸多挑战。物理模型简化导致的模型误差积累，以及高维参数空间下计算效率低下，均限制了其在实际应用中的效果。

近年来，随着传感器技术和数据分析方法的发展，基于数据驱动的气动参数辨识方法逐渐兴起。这些方法利用飞行试验数据，通过机器学习和统计分析技术，实现对气动参数的在线修正[^svm]。夏悠然等[^pso_elm]提出一种基于改进极限学习机（Extreme Learning Machine）和集成学习的气动参数辨识算法，通过混沌初始化策略、自适应速度更新策略以及裁切阈值集成框架，在降低对动力学建模依赖的同时提升了模型泛化能力。Gao等[^dbo_kelm]采用改进蜣螂优化算法优化核极限学习机的核参数与正则化系数，并应用于弹丸气动参数辨识。Xu等[^ls_svm_ukf]提出一种基于最小二乘支持向量机与无损卡尔曼滤波的在线建模方法，用于估计大机动飞行中高速飞行器的时变气动参数。Winter等[^mlp]提出一种结合递归局部线性神经模糊模型与多层感知器神经网络的非线性系统辨识方法，对时间序列响应进行非线性准静态修正，可以准确捕捉系统的线性和非线性特征。张家铭等[^bp]提出了基于BP神经网络的智能修正方法，通过对真实与标称气动数据偏差进行建模与拟合，实现了对气动参数的高精度修正。Li等[^cnn]提出了融合大卷积核与密集块结构的网络结构，通过大卷积核抑制高频噪声，利用密集块减少样本依赖并缓解经验模态分解（Empirical Mode Decomposition）的模态混叠，有效提升辨识精度、鲁棒性与稳定性。HUI等[^lstm_for_identify]提出一种结合堆叠长短时记忆网络与Levenberg-Marquardt法的气动参数估计方法，通过数据驱动建模与参数优化，实现了无需显式动力学假设的有效辨识。这些数据驱动方法在处理复杂非线性特性和高维响应方面表现出色，但其对大量高质量数据的依赖，以及缺乏物理约束导致的可解释性不足，仍然是亟需解决的问题。

最近几年，随着物理建模与数据驱动技术的融合，混合辨识方法逐渐成为研究热点。这类方法结合了物理模型的先验知识与数据驱动的灵活性，既能利用物理规律约束参数空间，又能通过数据学习捕捉复杂非线性特性。付军泉等[^pinn]提出了基于物理信息神经网络（Physics Informed Neural Network）的气动参数辨识方法，通过引入飞行动力学方程作为损失函数约束，实现了对气动参数的高精度辨识。刘磊等[^pinn2]提出了在闭环条件下基于物理信息神经网络的气动参数辨识方法，通过结合飞行控制系统动态特性，有效提升了辨识的鲁棒性和适应性。Lin等[^pinn3]提出一种基于物理信息神经网络的飞行器气动参数辨识方法，以六自由度运动方程作为物理约束，以待辨识气动参数作为网络变量，构建了替代传统飞行器模型的神经网络代理模型，通过联合优化神经网络参数和气动参数，实现了高精度的气动参数辨识。这些混合方法在提升辨识精度和鲁棒性方面展现出巨大潜力，但如何有效融合物理模型与数据驱动技术，仍需进一步探索。

综上所述，气动参数辨识方法经历了从传统物理模型方法到数据驱动方法，再到混合辨识方法的发展过程。当前大飞机气动参数辨识面临着处理复杂非线性特性、高维响应以及多源不确定性等挑战，亟需发展高效、鲁棒且适应复杂飞行条件的辨识方法。未来的研究方向可能包括进一步融合物理模型与数据驱动技术，提升辨识方法的可解释性和泛化能力，以及开发适应实时在线辨识需求的高效算法。

#### 大飞机“人-机-环”耦合故障识别及监测研究现状

随着现代航空技术的发展，大飞机已演变为典型的“人-机-环”（Man-Machine-Environment, MME）复杂系统，其安全运行不再仅依赖于单一软硬件的可靠性，更取决于驾驶员、自动化系统与外部环境之间动态交互的稳定性[^mme]。对于大飞机控制系统，故障主要可以分为执行器故障和传感器故障[^falut_type]。执行器作为飞机控制面的直接驱动装置，其故障可能导致控制失效甚至飞行事故，传感器则负责提供关键的飞行状态信息，其故障可能引发错误的控制指令，进而影响飞行安全。在复杂的飞行环境中，驾驶员的操作行为和环境因素（如气象条件、地形等）也会对系统性能产生显著影响。因此，研究大飞机“人-机-环”多重耦合故障识别及监测方法，对于提升飞行安全性和系统鲁棒性具有重要意义。

故障诊断技术是保障大飞机安全运行的关键手段，主要包括故障检测、识别和估计三个环节，即判断有无故障发生，对发生的故障进行定位以及估计故障的严重程度[^fault_diagnosis]。目前主流的故障诊断方法分为基于分析模型[^diagnosis_classification1]和基于数据驱动[^diagnosis_classification2]的方法。 基于分析模型的方法依赖于对系统物理特性的深入理解，通过建立精确的数学模型来描述系统的动态行为，从而实现故障的检测与识别。经典的方法包括参数估计方法[^parameter_estimation]和状态观测器方法等，其中状态观测器法是最经典的方法，其通过构建系统状态的估计器，比较估计值与实际测量值之间的残差来检测故障[^diagnosis_model]。例如Wang等[^event-triggered]基于正系统理论与李雅普诺夫稳定性理论设计了事件触发区间观测器，通过构造残差区间实现故障检测，在提升故障敏感度的同时降低了通信资源占用。Nejati等[^unknown_input_observe]针对直升机无人机悬停模式下执行器故障的耦合问题，设计了基于双未知输入观测器的鲁棒检测架构，实现了横滚/俯仰与偏航执行器故障的解耦与隔离。Chang等[^sliding-mode-observer]设计了多变量滑模观测器，在无需系统不确定性与故障先验信息条件下实现了执行器故障的有限时间重构，并通过无扰策略最小化扰动对故障估计的影响，实现了故障重构与系统不确定性的近似解耦。这类方法在诊断时通常需要精确的系统模型，且对模型不确定性和外部扰动较为敏感，限制了其在复杂飞行环境中的应用。

基于数据驱动的方法通过分析大量飞行数据，利用机器学习和统计分析技术实现故障的检测与识别，其优势在于不依赖于精确的物理模型，能够适应复杂非线性系统的动态特性[^diagnosis_data]。常用的方法包括支持向量机（Support Vector Machine）、随机森林（Random Forest）[^svm_fault]以及循环神经网络（Recurrent Neural Network）、卷积神经网络（Convolutional Neural Network）等各种深度学习方法[^deeplearning_fault]。院老虎等[^densenet-svm]针对旋转机械故障样本少、特征提取困难的问题，提出基于连续小波变换与密集连接卷积网络的特征提取方法，并结合支持向量机进行分类，该方法在少量样本条件下仍能取得更高的诊断准确率。黄鹏飞等[^random_forest]针对高超声速飞行器迎角传感器在强非线性与强噪声环境下的故障诊断难题，通过设计分数阶混沌系统实现噪声免疫与故障特征提取，并融合系统模型与支持向量机和随机森林等多种机器学习分类器构建诊断模型。何东钰等[^residual_network]针对飞机复杂运动机构故障与运动特征关联机制解析不足的问题，该研究提出了基于数据生成、特征转换与深度学习的三层诊断体系，通过二维图像映射构建多维特征张量，并结合注意力机制的残差网络实现了高精度故障诊断。Xiao等[^cnn-fault]将卫星推进器故障诊断转化为二值图像分类问题，提出了一种纯数据驱动的在线诊断方法，无需依赖推进器动力学与卫星姿态控制模型，即可实现对卡开与卡关故障的高精度检测与定位。荣光等[^cnn-lstm-rnn]提出了基于仿真数据驱动的多模型融合方法，通过构建CNN、LSTM与RNN组成的深度学习模型库对各单元进行模块化诊断，为飞行器智能故障诊断提供了可扩展的系统化解决方案。这些数据驱动方法在处理复杂非线性特性和高维数据方面表现出色，但其对大量高质量训练数据的依赖，以及缺乏物理约束导致的可解释性不足，仍然是亟需解决的问题。

综上所述，大飞机故障诊断技术虽已呈现出从物理机理主导向数据驱动转型的演进趋势，但在面对人-机-环要素深度耦合的极端工况时仍面临严峻挑战。受恶劣环境扰动与人因不确定性的双重影响，故障传播往往表现出极强的隐蔽性与非线性级联特征，导致现有方法难以在动态反馈回路中精准解耦多重并发故障。因此，未来的研究亟需打破单一技术范式的局限，致力于构建物理约束与数据挖掘深度融合的混合诊断架构。这不仅需要提升算法在强时空不确定性下的特征提取与泛化能力，更应着力突破系统级联失效的态势演化预测难题，从而为实现全飞行包线内的超前预警与精准决策提供坚实支撑。


## 项目的研究内容、研究目标，以及拟解决的关键科学问题

### 研究内容

#### 不确定性下非线性气动参数实时在线随机修正方法研究

##### 基准气动参数失准机理分析与动态演化建模方法研究

鉴于真实飞行环境与地面试验边界的显著差异，气动参数呈现出强非线性、时变性及多源不确定性耦合特征。传统建模方法通常将这些偏差笼统处理，难以区分物理建模误差与环境随机干扰。本研究旨在构建多源不确定性解耦与量化框架，通过剖析认知不确定性（如风洞壁干扰、模型简化）与随机不确定性（如大气湍流、测量噪声）的耦合机理，对两者进行分离并厘清不同不确定性源的传播路径。在此基础上，引入非定常时间项与高阶状态量，建立含时变系数的气动参数动态演化方程，并利用能量守恒与对称性原理施加物理约束，以规避稀疏数据下的过拟合风险。

i）多源不确定性解耦与分层量化框架构建。建立分层量化体系，严格区分源于理论简化与物理认知不足的认知不确定性，以及源于环境随机性与测量底噪的随机不确定性。通过关联分析风洞/CFD数据与飞行试验数据，识别由雷诺数效应、激波/边界层干扰引发的系统性偏差，以及由大气湍流、阵风扰动及传感器噪声导致的随机误差，实现对气动参数偏差来源的精准溯源与解耦。

ii）非线性参数演化建模与全局灵敏度分析。针对大迎角、过失速等复杂流态，构建包含迎角变化率及角速度高阶项的动态演化方程，以捕捉气动力的非定常迟滞效应。基于全局灵敏度分析，在全飞行包络内定量计算马赫数、迎角及舵面偏转等输入变量对气动参数输出方差的贡献权重。根据灵敏度排序结果，剔除冗余变量，确立关键修正参数集，在保证模型物理保真度的前提下显著降低在线辨识的计算维度与搜索空间。

##### 基于生成式人工智能的气动参数实时在线修正方法研究

针对传统确定性辨识方法无法表征参数分布且难以处理一对多反问题的局限，本研究提出利用条件可逆神经网络（cINN）解决气动参数反演难题。cINN利用其双射特性，学习参数在飞行观测数据条件下的完整后验分布，将复杂的参数空间映射为潜在空间的标准分布，实现不确定性条件下的实时随机修正。本研究旨在构建以飞行观测为条件输入的逆向参数映射架构，利用生成式人工智能的概率建模能力，赋予模型在飞行状态缓变或突变下的在线自适应能力，并配合轻量化推理设计，确保毫秒级的实时响应性能。

i）基于cINN的离线流场概率建模与潜在空间监督训练。构建可逆耦合层网络结构，设计正向流场预测与逆向参数推断的对偶流程。离线训练时cINN将复杂的气动参数分布映射为潜在空间的标准高斯分布，通过严格监督潜在变量分布，使得网络能够捕捉气动参数在不同飞行状态下的完整概率特征与多解性，从而为在线修正提供包含均值与不确定性信息的基准先验输出。

ii）在线增量学习与轻量化推理机制。针对飞行状态缓变或突变引起的参数动态演化，设计基于少量新观测数据的在线增量学习机制。通过冻结大部分网络权重，仅微调最后几层参数，实现对新工况下气动参数分布的快速适应与更新。同时，采用模型剪枝与量化技术，构建轻量化推理架构，确保在机载计算资源受限的条件下实现毫秒级的实时响应能力。

##### 气动参数辨识结果的可信度定量评估与鲁棒性增强机制研究
 
修正后的气动参数若缺乏可信度评估，将难以安全应用于闭环控制系统。本研究旨在建立输出结果的置信度量化指标，并研究复杂干扰下的系统鲁棒性增强策略。利用cINN生成概率密度函数的特性，通过计算微分熵、方差及置信区间，实时量化模型预测的不确定性水平，为控制律增益调参提供理论依据。同时，针对机载传感器普遍存在的高频测量噪声，设计基于噪声注入的鲁棒性增强机制，确保辨识算法在低信噪比环境下仍能输出稳定的参数估计，并基于Lyapunov理论验证闭环系统的综合效能。

i）基于后验分布的可信度定量评估方法。利用cINN生成的参数分布计算微分熵作为信心指标，当熵值超限时触发模型置信度预警。实时构建参数的95%置信区间，建立其与控制律增益的动态映射关系：在置信区间收敛时采用高增益高性能控制，在区间发散时自动切换至保守鲁棒控制，实现辨识-控制的动态协同。

ii）基于噪声注入训练的鲁棒性增强与闭环验证。为了提高辨识算法对真实飞行中传感器噪声的容忍度，提出随机噪声注入的鲁棒性增强策略。在cINN的离线训练阶段，显式地在输入端（如角速度、过载信号）叠加符合真实传感器特性的高斯白噪声与有色噪声。通过这种对抗式的训练与设计，迫使模型学习忽略高频噪声干扰，专注于提取低频气动特性。最终，通过六自由度全量非线性仿真，验证该机制在叠加真实量级传感器噪声及阵风干扰下，气动参数辨识的收敛速度、稳定性及对轨迹跟踪控制的支撑作用。


### 参考文献

---


[^civil]: “a city centre building of architectural” ([Macdonald和Strachan, 2001, p. 225](zotero://select/library/items/484EFSEN)) ([pdf](zotero://open-pdf/library/items/9RDLSBUV?page=7&annotation=4AY7ABIN))
[^aircraft]: “flight control systems” ([Zhou 等, 2024, p. 1247](zotero://select/library/items/JWLYGDE6)) ([pdf](zotero://open-pdf/library/items/B2766C8Y?page=3&annotation=2NKA4I2I))
[^mechanical]: “Mechanical Systems” ([Fu 等, 2025, p. 1](zotero://select/library/items/TNUS2EC5)) ([pdf](zotero://open-pdf/library/items/5GFYF8U5?page=1&annotation=PVEP6GZ4))
[^interval]: “Non-probabilistic” ([Luo 等, 2019, p. 663](zotero://select/library/items/MFS5APZB)) ([pdf](zotero://open-pdf/library/items/IRW35FJC?page=1&annotation=7USI7VCC))
[^pbox]: “probability box” ([Red-Horse和Benjamin, 2004, p. 186](zotero://select/library/items/GAWUUF27)) ([pdf](zotero://open-pdf/library/items/PPCVP7WQ?page=4&annotation=BWJJGEDM))
[^probabilistic]: “A probabilistic approach” ([Panzeri 等, 2018, p. 3](zotero://select/library/items/FBW49APF)) ([pdf](zotero://open-pdf/library/items/YTRAI7LT?page=3&annotation=344D42TW))
[^mc]: “Monte Carlo Techniques” ([Hurtado和Barbat, 1998, p. 3](zotero://select/library/items/5RZHR7YG)) ([pdf](zotero://open-pdf/library/items/DVIVL56A?page=1&annotation=NEG5FMFD))
[^blindkriging]: “Blind Kriging” ([Tartaruga 等, 2015, p. 7](zotero://select/library/items/7736XKA2)) ([pdf](zotero://open-pdf/library/items/IIP5C8IL?page=8&annotation=64JMRFYK))
[^mc1]: “考虑结构参数不确定性的概率颤振分析” ([闫国良 等, 2017, p. 323](zotero://select/library/items/N63IIUVJ)) ([pdf](zotero://open-pdf/library/items/RP7DJ2RH?page=1&annotation=IVXQCKH4))
[^mc2]: “Uncertainty Quantification in Trajectory Prediction for Aircraft Operations” ([Sankararaman和Daigle, 2017, p. 1](zotero://select/library/items/8ABG23KP)) ([pdf](zotero://open-pdf/library/items/LEBE4T3K?page=1&annotation=NC7WSLVP))
[^pce_zhao]: “基于 PCE 方法的翼型不确定性分析及稳健设计” ([赵轲 等, 2014, p. 10](zotero://select/library/items/5TSITXX6)) ([pdf](zotero://open-pdf/library/items/KSFKIN6U?page=1&annotation=AN7JKG3T))
[^pce]: “Aircraft Conflict Detection Under Wind Uncertainty” ([Mota 等, 2023, p. 1](zotero://select/library/items/567GTI87)) ([pdf](zotero://open-pdf/library/items/M38H267H?page=1&annotation=SNKJMXGP))
[^pbox1]: “the probability-box (P-box)” ([Bi 等, 2022, p. 2](zotero://select/library/items/SX7G52NV)) ([pdf](zotero://open-pdf/library/items/XZ38NMDT?page=2&annotation=DBAJES54))
[^tmcmc]: “transitional Markov chain Monte Carlo TMCMC” ([Ching和Chen, 2007, p. 816](zotero://select/library/items/ASU2FRM7)) ([pdf](zotero://open-pdf/library/items/CZBH5BEV?page=1&annotation=QJGUXBYD))
[^is]: “Importance Sampling” ([Wu 等, p. 1](zotero://select/library/items/QK565ZX2)) ([pdf](zotero://open-pdf/library/items/QGJBBZQ2?page=1&annotation=RIUCN6EN))
[^ss]: “Subset Simulation” ([DiazDelaO 等, 2017, p. 1102](zotero://select/library/items/8274EHQV)) ([pdf](zotero://open-pdf/library/items/X8J8AV4H?page=1&annotation=UK4VKSH5))
[^rsm]: “response surface models” ([Fang 等, 2012, p. 83](zotero://select/library/items/NVVC3ABP)) ([pdf](zotero://open-pdf/library/items/59GHPA9R?page=1&annotation=JRL7WFFA))
[^kriging]: “自适应 Kriging” ([宋周洲 等, 2024, p. 762](zotero://select/library/items/DQ9FJ4GM)) ([pdf](zotero://open-pdf/library/items/N7RPW48T?page=1&annotation=LDER9DC7))
[^lstm]: “Surrogate modelling for an aircraft dynamic landing loads simulation using an LSTM AutoEncoder-based dimensionality reduction approach” ([Lazzara 等, 2022, p. 1](zotero://select/library/items/UQG6FYGZ)) ([pdf](zotero://open-pdf/library/items/ZZ99MZCT?page=1&annotation=LVB4EIZM))
[^nasa]: “The NASA langley challenge on optimization under uncertainty” ([Crespo和Kenny, 2021, p. 1](zotero://select/library/items/GWGF5KAJ)) ([pdf](zotero://open-pdf/library/items/95EBQQAD?page=1&annotation=SUL5RX4Q))

---

[^LeastSquare]: “least squares method” ([Wang 等, 2022, p. 3](zotero://select/library/items/IQTI3T6X)) ([pdf](zotero://open-pdf/library/items/PE9594NW?page=3&annotation=5H7ULTW4))
[^MaximumLikelihood]: “PD[LPXP OLNHOLKRRG HVWLPDWLRQ]” ([Dai 等, 2022, p. 923](zotero://select/library/items/3CZYVFE5)) ([pdf](zotero://open-pdf/library/items/ZMD6H5R4?page=1&annotation=8DFUDV28))
[^kalman]: “ExtendedKalman Filter” ([Ljung, 1979, p. 36](zotero://select/library/items/LJRM47MN)) ([pdf](zotero://open-pdf/library/items/MJLITYK6?page=1&annotation=J4T5RQJU))
[^algorithm_nn]: “智能算法” ([崔瀚 等, 2025, p. 15](zotero://select/library/items/M2PEG69M)) ([pdf](zotero://open-pdf/library/items/P8LUWCR3?page=3&annotation=E9KR3X53))
[^ga]: “多种群遗传算法” ([周圆明 等, 2007, p. 1](zotero://select/library/items/3TM7S2ZF)) ([pdf](zotero://open-pdf/library/items/AEQPYFVZ?page=1&annotation=A2AIVXE6))
[^pso]: “粒子群算法” ([张天姣 等, 2010, p. 1](zotero://select/library/items/IJZ3I84X)) ([pdf](zotero://open-pdf/library/items/7PIUA7QL?page=1&annotation=RWRZTPBV))
[^pso_elm]: “改进粒子群优化” ([夏悠然 等, 2025, p. 2612](zotero://select/library/items/Z7ZTHB6U)) ([pdf](zotero://open-pdf/library/items/FZ9GXVVW?page=1&annotation=M8LJ5295))
[^ga1]: “Genetic Algorithms” ([Peña-García 等, 2024, p. 1](zotero://select/library/items/Z3NY3SNY)) ([pdf](zotero://open-pdf/library/items/4S6EEWYJ?page=1&annotation=EAGUUJUA))
[^pso1]: “PSO-SO algorithm” ([Yang 等, 2024, p. 1](zotero://select/library/items/9SVLVKGQ)) ([pdf](zotero://open-pdf/library/items/35U5X649?page=2&annotation=CUFVXCAH))
[^svm]: “基 于 SVM 的神经网络方法” ([浦甲伦 等, 2018, p. 1](zotero://select/library/items/VLFM5XE8)) ([pdf](zotero://open-pdf/library/items/WLND75IM?page=1&annotation=SDZKF4TI))
[^pinn]: “基于物理信息神经网络” ([付军泉 等, 2023, p. 30](zotero://select/library/items/4U2MQWCN)) ([pdf](zotero://open-pdf/library/items/ZJJF7C74?page=1&annotation=RBA4PTAB))
[^lstm_for_identify]: “Long Short-Term Memory (LSTM) network model” ([Hui 等, 2024, p. 123](zotero://select/library/items/C4WXDJ3I)) ([pdf](zotero://open-pdf/library/items/4HBV5GLD?page=1&annotation=SCRUZ4IN))
[^pinn2]: “基于物理信息神经网络” ([刘磊 等, 2025, p. 1](zotero://select/library/items/BJ7XAP83)) ([pdf](zotero://open-pdf/library/items/TBJN79T5?page=1&annotation=R5K8ZC7U))
[^mlp]: “connected neural networks” ([Winter和Breitsamter, 2018, p. 1](zotero://select/library/items/S6IP3SVU)) ([pdf](zotero://open-pdf/library/items/YJT7CVN4?page=1&annotation=RRTPGABC))
[^pinn3]: “Physics-informed Neural Networks (PINNs)” ([Lin 等, 2025](zotero://select/library/items/XGP9HSM4)) ([snapshot](zotero://open-pdf/library/items/4IR72IYE?sel=section%20%3E%20p&annotation=KLBV8PP7))
[^ls_svm_ukf]: “least square support vector machine (LS-SVM) and unscented Kalman filter (UKF)” ([Xu 等, 2025, p. 86](zotero://select/library/items/VK5ZZY8G)) ([pdf](zotero://open-pdf/library/items/TBDXWHQP?page=1&annotation=7HM3FY6H))
[^dbo_kelm]: “Dung Beetle Optimization (DBO) algorithm” ([Gao和Yi, 2025, p. 1](zotero://select/library/items/7IBSNG27)) ([pdf](zotero://open-pdf/library/items/5ND87CKV?page=1&annotation=7RGWUUQB))
[^srukf]: “平 方 根 无 迹 Kalman 滤 波 器(SRUKF)和 无 迹 Rauch-Tung-Striebel 平 滑 器 (URTSS)” ([汪清 等, 2025, p. 1](zotero://select/library/items/D8IA867Q)) ([pdf](zotero://open-pdf/library/items/BKQNQIG7?page=1&annotation=Q2M9UXN8))
[^cnn]: “large convolutional kernel and dense block” ([Li 等, 2024, p. 12](zotero://select/library/items/VBJUTN59)) ([pdf](zotero://open-pdf/library/items/MC584BFG?page=1&annotation=MT7NMYDA))
[^bp]: “基于机器学习的气动参数智能修正方法 张家铭 钟鸿豪 白文艳 孙 友 曹玉腾 北京航天自动控制研究所,北京 100854” ([张家铭 等, 2021, p. 49](zotero://select/library/items/6ZUT9EDH)) ([pdf](zotero://open-pdf/library/items/GWH6X4AE?page=1&annotation=8KIR4JNM))

---

[^falut_type]: “both sensor and actuator faults” ([Witczak, 2014, p. 168](zotero://select/library/items/UXWQKPIH)) ([pdf](zotero://open-pdf/library/items/B8FIB65H?page=179&annotation=BDZHF5VL))
[^mme]: “Risk assessment of complex system based on manmachine-environment” ([Guo 等, 2019, p. 228](zotero://select/library/items/I78CAD7H)) ([pdf](zotero://open-pdf/library/items/NZPATL8L?page=1&annotation=FSWU62KU))
[^fault_diagnosis]: “故障诊断技术” ([马亮 等, 2022, p. 1651](zotero://select/library/items/75UICPY2)) ([pdf](zotero://open-pdf/library/items/PTBI2Q3I?page=2&annotation=Q8QPFY7P))
[^diagnosis_classification1]: “Model-Based and Signal-Based” ([Gao 等, 2015, p. 3757](zotero://select/library/items/2LPABTLT)) ([pdf](zotero://open-pdf/library/items/VK7C27EK?page=1&annotation=BV4NERL6)) 
[^diagnosis_classification2]: “Knowledge-Based” ([Gao 等, 2015, p. 3768](zotero://select/library/items/V7PYT73L)) ([pdf](zotero://open-pdf/library/items/ENJ3UPL8?page=1&annotation=LTWLAE4E))
[^diagnosis_model]: [Ding, Model-Based Fault Diagnosis Techniques: Design Schemes, Algorithms and Tools, 2013](zotero://select/library/items/XWL7X4HA)
[^diagnosis_data]: [李晗 et al., 基于数据驱动的故障诊断方法综述, 2011, 控制与决策](zotero://select/library/items/9ZZVHKAS)
[^parameter_estimation]: “Parameter Estimation Using Adaptive Kalman Filter” ([Bagheri 等, 2007, p. 72](zotero://select/library/items/Z78KWY9K)) ([pdf](zotero://open-pdf/library/items/V9DD5CW8?page=1&annotation=JTQFKZK5))
[^event-triggered]: “Observer-based fault detection for large-scale systems with event-triggered protocolsq” ([Wang 等, 2025, p. 23](zotero://select/library/items/MTM6ETCL)) ([pdf](zotero://open-pdf/library/items/M5RLGNB6?page=1&annotation=W27ZW9IL))
[^unknown_input_observe]: “unknown input observer” ([“Actuator Fault Detection and Isolation for Helicopter Unmanned Arial Vehicle in the Present of Disturbance”, 2021, p. 676](zotero://select/library/items/UBG79LRF)) ([pdf](zotero://open-pdf/library/items/PNMYJPA7?page=1&annotation=NBEI6WLJ))
[^sliding-mode-observer]: “sliding-mode-observer-based” ([Chang 等, 2021, p. 8](zotero://select/library/items/IDF46HS2)) ([pdf](zotero://open-pdf/library/items/62HM4ZM8?page=1&annotation=NWJ68PTE))
[^svm_fault]: “Fisher 判别法、逻辑回归、随机森林和支持向量机” ([朱兴动 等, 2020, p. 1](zotero://select/library/items/PBDYPPHX)) ([pdf](zotero://open-pdf/library/items/HCWK69E2?page=1&annotation=8NTH7UYN))
[^densenet-svm]: “基于密集连接卷积网络(DenseNet)和支 持 向 量 机(SVM)” ([院老虎 等, 2021, p. 1635](zotero://select/library/items/YW92AK26)) ([pdf](zotero://open-pdf/library/items/UIEAJVFY?page=1&annotation=ZDRKQEDE))
[^cnn-lstm-rnn]: “通过构建包含卷积神经网络(CNN)、长短期记忆网络(LSTM)、循环神经网络(RNN)的深度学习 模型库” ([荣光 等, 2025, p. 73](zotero://select/library/items/WMV4MBA8)) ([pdf](zotero://open-pdf/library/items/9XSVQ89L?page=1&annotation=NBTGVXMB))
[^cnn-fault]: “CNN” ([Xiao和Yin, 2021, p. 10166](zotero://select/library/items/9JC7MYMK)) ([pdf](zotero://open-pdf/library/items/5FQHSR6G?page=5&annotation=H73LI86Q))
[^random_forest]: “随机森林” ([黄鹏飞 等, 2023, p. 1208](zotero://select/library/items/QTYFTEX2)) ([pdf](zotero://open-pdf/library/items/G8AVIQGQ?page=6&annotation=M4ISLZJF))
[^deeplearning_fault]: “Deep Learning for Fault Diagnostics” ([Sundaram 等, 2021, p. 41246](zotero://select/library/items/5CND64D7)) ([pdf](zotero://open-pdf/library/items/2JXDQQV8?page=1&annotation=V5TF6AIQ))
[^residual_network]: “残 差 网 络” ([何东钰 等, 2025, p. 1172](zotero://select/library/items/56RH5MTH)) ([pdf](zotero://open-pdf/library/items/KES55ETF?page=6&annotation=ZVRKQPXN))
