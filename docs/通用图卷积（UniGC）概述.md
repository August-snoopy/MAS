### 通用图卷积（UniGC）概述

**通用图卷积（UniGC）**是一种新颖的概念，通过将各种类型的图卷积（GC）表示为通用形式的特殊情况来统一它们。这种方法旨在整合现有图卷积的优势，并增强基于图的人体运动预测的灵活性。

#### 数学公式

在最通用的形式中，UniGC 设计用于处理具有维度 $ (T, J, C) $ 的 3D 运动序列 $ X $，其中：
- $ T $ 是帧数，
- $ J $ 是关节数，
- $ C $ 是每个关节的通道数或坐标数。

运动序列 $ X $ 被视为一个具有 $ T \times J \times C $ 个节点的图。这些节点之间的关系编码在一个具有形状 $ (T, J, C, T, J, C) $ 的 6 维全局邻接矩阵 $ A $ 中。通用 UniGC 操作表达式为：

$$
Y = G(X; A) \implies
\begin{cases}
x = R_{TJC}(X), \\
A = R_{TJC, TJC}(A), \\
y \leftarrow Ax, \\
Y = R_{T, J, C}(y),
\end{cases}
$$

其中：
- $ R_{TJC} $ 和 $ R_{TJC, TJC} $ 是将数组展平的重塑操作，
- $ y \leftarrow Ax $ 表示图卷积操作。

在此公式中：
- $ X $ 被重塑为一维数组 $ x $，
- $ A $ 被重塑为二维矩阵，
- $ y $ 计算为 $ A $ 和 $ x $ 的乘积，
- $ y $ 被重塑回原始维度以生成输出 $ Y $。

#### UniGC 的特化

UniGC 可以通过引入各种邻接掩码和捆绑策略来专注于数据的不同方面：

1. **空间-通道图卷积（$ G_{sc} $）**：
   
   - 引入掩码 $ M_{sc} $ 以专注于同一帧内的关系：
     $$
     M_{sc}[t_1, :, :, t_2, :, :] = \begin{cases}
     1 & \text{若 } t_1 = t_2, \\
     0 & \text{否则}.
     \end{cases}
     $$
   - 这导致：
     $$
     Y = G_{sc}(X; A, M_{sc}) \implies
     \begin{cases}
     x = R_{TJC}(X), \\
     M_{sc} = R_{TJC, TJC}(M_{sc}), \\
     y \leftarrow (A \odot M_{sc}) x, \\
     Y = R_{T, J, C}(y),
     \end{cases}
     $$
     其中 $ \odot $ 表示元素级乘法。
   
2. **空间-时间图卷积（$ G_{st} $）**：
   
   - 引入掩码 $ M_{st} $ 以专注于同一通道内的关系：
     $$
     M_{st}[:, :, c_1, :, :, c_2] = \begin{cases}
     1 & \text{若 } c_1 = c_2, \\
     0 & \text{否则}.
     \end{cases}
     $$
   - 这导致：
     $$
     Y = G_{st}(X; A, M_{st}) \implies
     \begin{cases}
     x = R_{TJC}(X), \\
     M_{st} = R_{TJC, TJC}(M_{st}), \\
     y \leftarrow (A \odot M_{st}) x, \\
     Y = R_{T, J, C}(y),
     \end{cases}
     $$
   
3. **空间图卷积（$ G_s $）**：
   - 使用组合的空间-通道和空间-时间掩码 $ M_s = M_{sc} \odot M_{st} $：
     $$
     Y = G_s(X; A) \implies
     \begin{cases}
     x = R_{TJC}(X), \\
     M_s = M_{sc} \odot M_{st}, \\
     y \leftarrow (A \odot M_s) x, \\
     Y = R_{T, J, C}(y).
     \end{cases}
     $$

其他特化如时间-通道 ($ G_{tc} $)、时间 ($ G_t $) 和通道 ($ G_c $) 图卷积通过使用适当的掩码来隔离沿特定维度的关系，遵循类似的方法。

### GCNext: 动态网络框架

**GCNext** 利用 UniGC 的适应性，通过在每一层和每个样本上动态选择最适合的图卷积类型，从而创建一个更高效、更有效的 GCN。它包括：

1. **选择器模块**：
   - 使用可学习的过程 $ S $ 确定最佳图卷积操作：
     $$
     v = \text{GumbelSoftmax}(S(X; \theta_s)),
     $$
     其中 $ v $ 是一个表示选择操作的独热向量。

2. **层结构**：
   
   - 每一层包含多个图卷积块和一个轻量级选择器模块。
   
3. **使用案例**：
   
   - **从头开始训练**：GCNext 为每个样本动态学习最佳架构。
   - **改进现有的 GCN**：通过动态集成不同图卷积的优势来增强现有的 GCN。

该框架减少了计算成本，同时提高了在人类运动预测等任务中的性能，实验已在 Human3.6M、AMASS 和 3DPW 数据集上得到了证明。

### 结论

UniGC 和 GCNext 提供了一种使用图卷积进行人体运动建模的多功能方法。通过在通用框架下统一不同的图卷积类型，它们允许对复杂的时空数据进行灵活和高效的建模。

**参考文献**：

- Wang, Xinshun, et al. "GCNext: Towards the Unity of Graph Convolutions for Human Motion Prediction." *The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)*, 2024.