Glass fibre epoxy composites / glass fibre reinforced thermoplastic composites
- Scarselli, G.; Quan, D.; Prasad, V.; Rao, P.S.; Hardiman, M.; Reid, I.; O’Dowd, N.P.; Murphy, N.; Ivankovic, A. (2023): Mode I fracture toughness of glass fibre reinforced thermoplastic composites after UV and atmospheric plasma treatments, Composites Science and Technology, 236, Article 109982. DOI: [10.1016/j.compscitech.2023.109982](https://doi.org/10.1016/j.compscitech.2023.109982);
- Scarselli, G.; Prasad, V.; Quan, D.; Maffezzoli, A.; Murphy, N.; Ivankovic, A. (2025): Interlaminar fracture properties of UV and plasma‐treated glass fiber epoxy composites, Polymer Composites, 46, Article 1871–1883. DOI: [10.1002/pc.29079](https://doi.org/10.1002/pc.29079);


通过Push out test 测量获得The interfacial shear strength (IFSS) $IFSS=\frac{P_{max}}{2\pi rt}$ (MPa)
- $P_{max}$ is the maximum force
- $r$ is **the radius of the single fiber** observed using the optical microscope before the test
- $t$ is the thickness (μm) of the composite slice tested

通过拉伸和断裂测试评估纯环氧树脂(基体matrix材料)的机械特性

## Three point bend tests

The Mode I fracture energy ($G_{IC}$) of the neat epoxy was evaluated by conducting a single-edge notched bend (SENB) test26,27 following the ASTM D5045-14 standard.
通过three-point bending tests，得到Elium 树脂的 I 型断裂能量为 $G_{IC}=815 \pm 19 J/m^{2}$。The corresponding plane strain fracture toughness 为$1.53 \pm 0.01 MPa~m^{1/2}$. 这些值证实了 Elium 热塑性树脂出色的断裂性能，是热固性树脂典型值的两倍多。

specimen geometry：
- thickness B in cm
- width W = 2B
- length L = 2.2W
- span S = 4W
- a is the crack length
  - the pre-crack length ‘a’ is such that $0.45 < \frac{a}{W} < 0.55$.

specimen material：
- $\nu$ is the Poisson’s ratio of the resin: 0.32 (taken from the existing literature)
- E is the tensile modulus of the resin: 2.59 GPa (experimentally evaluated)

$G_{IC}=\frac{(1-\nu^2){K_{IC}}^2}{E}$ in $J/m^{2}$
$K_Q=\left(\frac{P_Q}{B\sqrt{W}}\right)f(x)$
$f(x)=6\sqrt{x}\frac{(1.99-x(1-x)(2.15-3.93x+2.7x^2))}{(1+2x)\sqrt[3]{(1-x)}}$
- $x = a/W$
- $K_Q$ is the conditional or trial $K_{IC}$ （is the plain strain fracture energy）value with the unit as $MPa~m^{1/2}$
- $P_{Q}$ is the maximum load value in $kN$

The Mode II fracture energy ($G_{IIC}$) was evaluated by performing the asymmetric four-point bend (A4PB) test.

$G_{IIC}=\frac{K_{IIC}^2}{E}(1-\nu^2)$
$K_{IIC}=\frac{Q}{B\sqrt{W}}f\left(\frac{a}{W}\right)$
$\mathrm{Q=}\frac{P(L_1-L_2)}{L_1+L_2}$
$f{\left(\frac{a}{W}\right)}=9.763\left(\frac{a}{W}\right)^4-15.036\left(\frac{a}{W}\right)^3+8.667\left(\frac{a}{w}\right)^2+1.695\left(\frac{a}{W}\right)-0.037,where\left(\frac{a}{W}\right)\leq0.7$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250526165410.png)



此外通过 3 point bend tests还可以得到 composite 的Flextural modulus 
- 弹性模量有拉伸模量、压缩模量、弯曲模量、剪切模量和体积模量之分，取决于材料的形变模式


## DCB tests

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250529143001.png)


### Simulation tutorial about DCB and cohesive model

![Interlaminar fracture toughness test (DCB/Mode I) - YouTube](https://www.youtube.com/watch?v=ob-3Coej8rs)

[Fracture Mechanics Concepts: Micro→Macro Cracks; Tip Blunting; Toughness, Ductility & Yield Strength - YouTube](https://www.youtube.com/watch?v=SD6qITe3-Xo)

[Double Cantilever Beam Test - an overview | ScienceDirect Topics](https://www.sciencedirect.com/topics/engineering/double-cantilever-beam-test)
[ISO 15024:2023 DCB test (.pdf)](https://cdn.standards.iteh.ai/samples/84263/70d2f1d8ba3c4ead98f0aaad30863c48/ISO-15024-2023.pdf)

[Abaqus Cohesive粘聚力模型模拟双悬臂梁DCB试件分层开胶_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Y64y1n7z3/?spm_id_from=333.337.search-card.all.click&vd_source=1dba7493016a36a32b27a14ed2891088)

[ABAQUS中Cohesive粘聚力模型的2种定义方式（附案例操作步骤） - 知乎](https://zhuanlan.zhihu.com/p/522975182)

基于Traction-Separation Law的粘聚力模型：
- **粘聚力单元**（cohesive element）
- **粘聚力接触**（cohesive surface interaction）

![v2-a482e6b856115fbc21a4313480a7fbdf_1440w.jpg (666×215)](https://pic2.zhimg.com/v2-a482e6b856115fbc21a4313480a7fbdf_1440w.jpg)

一般来说，在使用内聚力模型时，需要给出刚度，极限强度，临界能量释放量（或者失效时的位移）。
纯I型（张开型）、纯II型（滑开型）和纯III型（撕开型）

![v2-379f99bf222e57ad193bd835a4771173_1440w.jpg (731×350)](https://pic2.zhimg.com/v2-379f99bf222e57ad193bd835a4771173_1440w.jpg)



### Simulation Target: Match $G_{p}$ and $G_{f}$

Fracture toughness: $G_C=G_m+G_p+G_f+G_{deb}$
- $G_{C}$ is obtained by experiment **已知**
- $G_{m}$ stands for the energy involved in the deformation and fracture of the matrix **已知**
- $G_{p}$ is the energy for additional matrix deformation associated with **fibre bridging**
- $G_{f}$ is the work needed for the **fibre fracture**
- $G_{deb}$ is the energy associated with the fibre/matrix interfacial debonding. **已知**

