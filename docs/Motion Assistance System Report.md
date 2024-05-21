# :walking: Motion Assistance System :walking_woman:

## 数据格式

本项目的数据来源于10名不同的测试者，他们穿戴IMU设备进行数据收集。共收集了24个IMU节点的数据，对于每一个IMU节点，我们获取一个长度为12的特征向量：包含3x3的位置旋转矩阵和长度为3的加速度向量。将以上矩阵和向量展平后拼接起来，再将所有节点的特征向量进行拼接，最终拼接测试者的12个身体特征信息（例如躯干长度、臂围、胸围等），最后加上标签，标签是当前正在进行的动作。数据示例如下：

| head_position_11 | head_position_12 | ...  | body_length | ...  | label           |
| ---------------- | ---------------- | ---- | ----------- | ---- | --------------- |
| -0.27487681      | -0.12415444      | ...  | 47          | ...  | "引体向上-举手" |

每一行为一个时间步，每一列为一个特征。

### 后处理

需要删除一些冗余列(以下为在Excel中显示的列标)：

```
GW -- PP
IS -- RL
```

这些冗余列对应的是手指的节点位置.

### 数据集扩充问题

针对数据集扩充问题，我们现在已经有了一些样例数据，大约有500个样本，可以先使用这些样本来进行简单的试验。

后续根据需求考虑添加样例，横向扩充与纵向扩充:

1. 横向扩充: 添加更多的样本类别，不局限于引体向上动作.

   <table style="width:100%; table-layout:fixed;">
     <tr>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/S001C001P001R001A044_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/S003C001P008R001A008_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/S002C001P010R001A017_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/S003C001P008R001A002_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/S001C001P001R001A051_w.gif?raw=true"></td>
     </tr>
     <tr>
       <td><font size="1">Touch head<font></td>
       <td><font size="1">Sitting down<font></td>
       <td><font size="1">Take off a shoe<font></td>
       <td><font size="1">Eat meal/snack<font></td>
       <td><font size="1">Kick other person<font></td>
     </tr>
     <tr>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/hammer_throw_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/clean_and_jerk_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/pull_ups_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/tai_chi_w.gif?raw=true"></td>
       <td><img width="150px" src="https://github.com/yysijie/st-gcn/blob/master/resource/info/juggling_balls_w.gif?raw=true"></td>
     </tr>
     <tr>
       <td><font size="1">Hammer throw<font></td>
       <td><font size="1">Clean and jerk<font></td>
       <td><font size="1">Pull ups<font></td>
       <td><font size="1">Tai chi<font></td>
       <td><font size="1">Juggling ball<font></td>
     </tr>
   </table>

2. 纵向扩充: 添加更多的样本数量，拟扩充到万级样本量(500个样本采集所需时间约1小时).

## 任务

1. **3D Pose Estimation**：给出一段时间序列，识别出当前时间序列对应的动作。
2. **Pose Tracking**：给定一段时间序列，预测下一个时间窗口中的动作。
3. **特征选择任务(特征缺失任务)**: 在上述任务基础上，尝试抛弃一些IMU节点特征，在缺失特征的情况下进行实验.

### 分类任务

![image-20240521153018744](C:\Users\zeyan\AppData\Roaming\Typora\typora-user-images\image-20240521153018744.png)

对于分类任务，每一个样本为一个动作对应的一段连续时间内的特征向量。构建出一个元组`(Matrix， Label)`。

`Matrix`：

| 位置矩阵(展平) | 加速度向量 | 人体特征信息 | ...  |
| :------------: | ---------- | ------------ | ---- |
|      -1-1      | -1-1       | 0-100        | ...  |

`Label`：

例如："引体向上-举手"、"引体向上-放下"

### 序列到序列任务

![img](https://raw.githubusercontent.com/una-dinosauria/human-motion-prediction/master/imgs/walking.gif)

我们期望使用这些数据完成一个序列到序列任务，具体来说：

1. 根据一段时间步的特征向量来预测当前时间步的标签。
2. 生成下一段时间步的特征向量和标签。

考虑采用深度学习模型，如LSTM或Transformer，STGNN，进行特征提取和动作预测。具体模型设计可以根据任务的复杂性和数据量来调整。

预处理需要考虑 *相对位置编码，节点信息嵌入* 等技术.

### 评价指标

针对不同的任务应该设置不同的评价指标，例如：

- 分类任务：准确率（Accuracy）、精确率（Precision）、召回率（Recall）和F1值（F1-Score）。
- 生成任务：均方误差（MSE）、平均绝对误差（MAE）等。

### 已有可参考研究

1. **ST-GCN (Spatial Temporal Graph Convolutional Networks)**
   - 论文：[Spatio-Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
     AAAI 2018 (CCF A)
   - 方法论：ST-GCN将人体姿态表示为图结构，通过图卷积网络（GCN）来学习时空特征，实现动作识别。
   - 评价指标：Accuracy
   - 启发式设计：可以基于ST-GCN的思想构建ST-GNN模型，将节点表示为人体关节点，边表示关节点之间的空间关系，通过GCN层和时间维度上的卷积层来学习时空特征。ST-GNN能够有效地捕获人体结构的拓扑关系，并结合时序信息进行建模，从而实现对运动行为的建模和预测。
2. **TGNN (Temporal Graph Neural Networks)**
   - 论文：[Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2005.11650)
     KDD 2020 (CCF A)
   - 方法论：TGNN专注于处理动态图数据，可以有效地处理时序关系，并结合图神经网络来进行建模。
   - 评价指标：RSE、CORR（回归任务）
   - 启发式设计：借鉴TGNN的思想，设计一个能够动态更新图结构的ST-GNN模型，以适应不同时间步的拓扑关系变化。
3. **Hybrid Deep Learning**
   - 论文：[A new hybrid deep learning model for human action recognition](https://www.sciencedirect.com/science/article/pii/S1319157819300412)
   - 方法论：使用一种基于GRU（门控循环神经网络）的混合深度学习方法来进行人体动作识别，完成了分类任务。
   - 评价指标：Accuracy

4. 综述
   - 论文: [Recent developments in human motion analysis](https://www.sciencedirect.com/science/article/pii/S0031320302001000)
     Pattern recognition (SCI Q1， IF 8.0)

5. **Human Motion Prediction**
   - 论文：[On Human Motion Prediction Using Recurrent Neural Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Martinez_On_Human_Motion_CVPR_2017_paper.html)
     CVPR 2017 (CCF A)
   - 方法论：该论文提出了均方误差（MSE）和平均绝对误差（MAE）作为动作预测任务的评价指标，能够有效地评估预测结果与真实值之间的差异。


### 对比实验设计

为了验证模型的有效性，我们可以设计对比实验，例如：

1. **Baseline模型**：使用传统机器学习模型（如SVM、随机森林）进行初步实验。
2. **深度学习模型**：如LSTM、GRU、Transformer等，进行深入实验和优化。
3. **数据增强**：使用数据增强技术（如噪声加入、时间偏移）扩充数据集，观察其对模型性能的影响。