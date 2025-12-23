# Ishigami function

[Ishigami Function - UQ with UQLab / Benchmarks - UQWorld](https://uqworld.org/t/ishigami-function/55)
[Ishigami Function — gsa-module 0.5.3 documentation](https://gsa-module.readthedocs.io/en/stable/test_gallery/ishigami.html)

### 第一步：定义 Ishigami 函数

首先，我们写出 Ishigami 函数的形式：
$$
Y = f(X_1, X_2, X_3) = \sin(X_1) + a \sin^2(X_2) + b X_3^4 \sin(X_1)
$$
其中，输入参数 $X_1, X_2, X_3$ 是相互独立的，并且都服从在 $[-\pi, \pi]$ 上的均匀分布，即 $X_i \sim U[-\pi, \pi]$。
标准的参数值为 $a=7, b=0.1$。

### 第二步：理论基础 - Sobol-Hoeffding 分解

Sobol 敏感性分析的核心思想是将函数 $f$ 分解为对单个输入、输入对、输入三元组等贡献的项之和。这种分解称为 Sobol-Hoeffding 分解：
$$
Y = f_0 + \sum_{i} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + f_{1,2,3}(X_1, X_2, X_3)
$$
其中：
*   $f_0 = E[Y]$ 是函数的均值。
*   $f_i = E[Y|X_i] - f_0$ 是只由 $X_i$ 引起的主效应。
*   $f_{ij} = E[Y|X_i, X_j] - f_i - f_j - f_0$ 是由 $X_i$ 和 $X_j$ 共同引起的交互效应。

总方差 $D = \text{Var}(Y)$ 可以被分解为各项的方差之和：
$$
D = \sum_i \text{Var}(f_i) + \sum_{i<j} \text{Var}(f_{ij}) + \text{Var}(f_{1,2,3})
$$
我们用 $D_i = \text{Var}(f_i)$ 和 $D_{ij} = \text{Var}(f_{ij})$ 来表示这些方差分量。您的目标就是求出这些 $D$ 值。

---

### 第三步：推导过程

在开始之前，我们需要记住一些在 $[-\pi, \pi]$ 上的积分/期望结果：
*   $E[\sin(X_i)] = \frac{1}{2\pi}\int_{-\pi}^{\pi}\sin(x)dx = 0$
*   $E[\cos(X_i)] = 0$
*   $E[\sin^2(X_i)] = \frac{1}{2\pi}\int_{-\pi}^{\pi}\sin^2(x)dx = \frac{1}{2\pi} \cdot \pi = \frac{1}{2}$
*   $E[X_i^n]$ 当 n 为奇数时为 0。
*   $E[X_i^4] = \frac{1}{2\pi}\int_{-\pi}^{\pi}x^4dx = \frac{1}{2\pi} \left[\frac{x^5}{5}\right]_{-\pi}^{\pi} = \frac{\pi^4}{5}$
*   $E[X_i^8] = \frac{1}{2\pi}\int_{-\pi}^{\pi}x^8dx = \frac{\pi^8}{9}$

#### 1. 计算均值 $f_0$

$$
f_0 = E[Y] = E[\sin(X_1) + a \sin^2(X_2) + b X_3^4 \sin(X_1)]
$$
利用期望的线性性质和变量的独立性：
$$
f_0 = E[\sin(X_1)] + a E[\sin^2(X_2)] + b E[X_3^4]E[\sin(X_1)]
$$
$$
f_0 = 0 + a \cdot \frac{1}{2} + b \cdot \frac{\pi^4}{5} \cdot 0 = \frac{a}{2}
$$

#### 2. 计算一阶效应项 $f_i$

*   **$f_1(X_1)$:**
    $E[Y|X_1] = E[\sin(X_1) + a \sin^2(X_2) + b X_3^4 \sin(X_1) | X_1]$
    $= \sin(X_1) + a E[\sin^2(X_2)] + b E[X_3^4] \sin(X_1)$
    $= \sin(X_1) + \frac{a}{2} + \frac{b\pi^4}{5} \sin(X_1)$
    $f_1(X_1) = E[Y|X_1] - f_0 = \left(\sin(X_1) + \frac{a}{2} + \frac{b\pi^4}{5} \sin(X_1)\right) - \frac{a}{2} = (1+\frac{b\pi^4}{5})\sin(X_1)$

*   **$f_2(X_2)$:**
    $E[Y|X_2] = E[\sin(X_1)] + a \sin^2(X_2) + b E[X_3^4 \sin(X_1)]$
    $= 0 + a \sin^2(X_2) + 0 = a \sin^2(X_2)$
    $f_2(X_2) = E[Y|X_2] - f_0 = a \sin^2(X_2) - \frac{a}{2}$

*   **$f_3(X_3)$:**
    $E[Y|X_3] = E[\sin(X_1)] + a E[\sin^2(X_2)] + b X_3^4 E[\sin(X_1)]$
    $= 0 + \frac{a}{2} + 0 = \frac{a}{2}$
    $f_3(X_3) = E[Y|X_3] - f_0 = \frac{a}{2} - \frac{a}{2} = 0$

#### 3. 计算二阶效应项 $f_{ij}$

*   **$f_{1,3}(X_1, X_3)$:**
    $E[Y|X_1, X_3] = \sin(X_1) + a E[\sin^2(X_2)] + b X_3^4 \sin(X_1) = \sin(X_1) + \frac{a}{2} + b X_3^4 \sin(X_1)$
    $f_{1,3} = E[Y|X_1, X_3] - f_1 - f_3 - f_0$
    $f_{1,3} = (\sin(X_1) + \frac{a}{2} + b X_3^4 \sin(X_1)) - (1+\frac{b\pi^4}{5})\sin(X_1) - 0 - \frac{a}{2}$
    $f_{1,3} = b \sin(X_1) (X_3^4 - \frac{\pi^4}{5})$

*   **其他交互项:** 可以证明 $f_{1,2}$, $f_{2,3}$ 和 $f_{1,2,3}$ 均为 0。

#### 4. 计算方差分量 $D_i$ 和 $D_{ij}$

现在我们计算每个效应项的方差。对于任意函数 $g(X)$，$\text{Var}(g(X)) = E[g^2(X)] - (E[g(X)])^2$。由于 $E[f_i] = E[f_{ij}]=0$，所以我们只需要计算 $E[f_i^2]$。

*   **$D_1 = \text{Var}(f_1)$:**
    $D_1 = \text{Var}((1+\frac{b\pi^4}{5})\sin(X_1)) = (1+\frac{b\pi^4}{5})^2 \text{Var}(\sin(X_1))$
    $\text{Var}(\sin(X_1)) = E[\sin^2(X_1)] - (E[\sin(X_1)])^2 = \frac{1}{2} - 0^2 = \frac{1}{2}$
    $D_1 = \frac{1}{2}(1+\frac{b\pi^4}{5})^2 = \frac{1}{2}(1 + \frac{2b\pi^4}{5} + \frac{b^2\pi^8}{25}) = \frac{1}{2} + \frac{b\pi^4}{5} + \frac{b^2\pi^8}{50}$
    **这与您给出的 $D_1$ 公式完全匹配。**

*   **$D_2 = \text{Var}(f_2)$:**
    $D_2 = \text{Var}(a \sin^2(X_2) - \frac{a}{2}) = a^2 \text{Var}(\sin^2(X_2))$
    $\text{Var}(\sin^2(X_2)) = E[\sin^4(X_2)] - (E[\sin^2(X_2)])^2$
    $E[\sin^4(X_2)] = \frac{1}{2\pi}\int_{-\pi}^{\pi}\sin^4(x)dx = \frac{3}{8}$
    $\text{Var}(\sin^2(X_2)) = \frac{3}{8} - (\frac{1}{2})^2 = \frac{3}{8} - \frac{1}{4} = \frac{1}{8}$
    $D_2 = a^2 \cdot \frac{1}{8} = \frac{a^2}{8}$
    **这与您给出的 $D_2$ 公式完全匹配。**

*   **$D_3 = \text{Var}(f_3) = \text{Var}(0) = 0$**
    **这与您给出的 $D_3$ 公式完全匹配。**

*   **$D_{1,3} = \text{Var}(f_{1,3})$:**
    $D_{1,3} = \text{Var}(b \sin(X_1) (X_3^4 - \frac{\pi^4}{5}))$
    由于 $X_1$ 和 $X_3$ 独立，$\text{Var}(AB) = E[A^2B^2] - (E[AB])^2 = E[A^2]E[B^2] - (E[A]E[B])^2$。
    这里 $A = \sin(X_1), B = X_3^4 - \frac{\pi^4}{5}$。$E[A]=0, E[B]=0$。所以 $\text{Var}(AB) = E[A^2]E[B^2] = \text{Var}(A)\text{Var}(B)$。
    $D_{1,3} = b^2 \text{Var}(\sin(X_1)) \text{Var}(X_3^4 - \frac{\pi^4}{5})$
    $\text{Var}(X_3^4) = E[(X_3^4)^2] - (E[X_3^4])^2 = E[X_3^8] - (\frac{\pi^4}{5})^2 = \frac{\pi^8}{9} - \frac{\pi^8}{25} = \frac{16\pi^8}{225}$
    $D_{1,3} = b^2 \cdot \frac{1}{2} \cdot \frac{16\pi^8}{225} = \frac{8b^2\pi^8}{225}$
    **这与您给出的 $D_{1,3}$ 公式完全匹配。**

*   **其他 $D_{ij}$ 和 $D_{1,2,3}$ 均为 0。**

#### 5. 计算总方差 D

总方差是所有方差分量之和：
$$
D = D_1 + D_2 + D_3 + D_{1,2} + D_{1,3} + D_{2,3} + D_{1,2,3}
$$
$$
D = \left(\frac{1}{2} + \frac{b\pi^4}{5} + \frac{b^2\pi^8}{50}\right) + \frac{a^2}{8} + 0 + 0 + \frac{8b^2\pi^8}{225} + 0 + 0
$$
整理一下关于 $b^2$ 的项：
$$
\frac{b^2\pi^8}{50} + \frac{8b^2\pi^8}{225} = b^2\pi^8 \left(\frac{1}{50} + \frac{8}{225}\right) = b^2\pi^8 \left(\frac{9+16}{450}\right) = b^2\pi^8 \frac{25}{450} = \frac{b^2\pi^8}{18}
$$
所以，总方差为：
$$
D = \frac{a^2}{8} + \frac{b\pi^4}{5} + \frac{b^2\pi^8}{18} + \frac{1}{2}
$$
**这也与您给出的 D 公式完全匹配。**

### 结论

通过对 Ishigami 函数进行严格的 Sobol-Hoeffding 分解，并利用输入变量在 $[-\pi, \pi]$ 上均匀分布的特性进行积分（求期望），我们可以一步步地推导出每一个方差分量的解析表达式，最终得到您提供的完整解析解。