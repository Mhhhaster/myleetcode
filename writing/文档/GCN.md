# GCN

## 一、GCN概述

### 1.1 概念引入

#### **特征工程vs表示学习**

特征工程：依靠专家提取显示特征，工程量大，特征选取的好坏直接决定后续任务的性能

表示学习：采用模型自动学习数据的隐式特征，不依赖专家，但需要庞大的训练数据集

#### **Graph Embedding(GE)**

图嵌入（Graph Embedding/Network Embedding，GE），属于表示学习的范畴，通常有两个层次的含义：

1. **将图中的节点表示成低维、实值、稠密的向量形式**，再将向量用于下游的具体任务中。例如用户社交网络得到就是每个用户的表示向量，可用于节点分类等；
2. **将整个图表示成低维、实值、稠密的向量形式**，用来对整个图结构进行分类

图嵌入的方式主要有三种：

1. **矩阵分解：**将节点间的关系用矩阵的形式加以表达，然后分解该矩阵以得到嵌入向量。
2. **DeepWalk：**DeepWalk 是基于 word2vec 词向量提出来的。word2vec  在训练词向量时，将语料作为输入数据，而图嵌入输入的是整张图。
3. **Graph Neural Network：**图结合deep learning方法搭建的网络统称为图神经网络GNN

#### Graph Neural Network

图神经网络(Graph Neural Network,  GNN)是指神经网络在图上应用的模型的统称。

从传播的方式来看，图神经网络可以分为图卷积神经网络（GCN），图注意力网络（GAT，缩写为了跟GAN区分），Graph LSTM等等

**图中的每个结点无时无刻不因为邻居和更远的点的影响而在改变着自己的状态直到最终的平衡，关系越亲近的邻居影响越大。**

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211226164312187.png" alt="image-20211226164312187" style="zoom: 50%;" />



图神经网络有三个过程：**聚合、更新、循环**，以结点A为例

**聚合邻居信息**：N=a[2,2,2,2,2]+b[3,3,3,3,3]+c[4,4,4,4,4]

**更新A信息**：σ(W[1,1,1,1,1]+α*N)   

​	σ是激活函数，relu或sigmoid等等

​	W是模型需要训练的参数

**循环**：重复上述过程两次，A结点即可聚合到E结点信息

### 1.2 Graph Convolutional Network

GCN主要在**聚合邻居结点信息**过程做了改进：

#### 卷积VS图卷积

卷积：对数字图像做卷积操作其实就是利用卷积核（卷积模板）在图像上滑动，将图像点上的像素灰度值与对应的卷积核上的数值相乘，然后将所有相乘后的值相加作为卷积核中间像素对应的图像上像素的灰度值，并最终滑动完所有图像的过程。

**用卷积核得到像素点的加权和从而提取到某种特定的特征，然后用反向传播来优化卷积核参数就可以自动的提取特征，是CNN特征提取的基石**

图卷积：任何一个图卷积层都可以写成这样一个非线性函数：
$$
H^{l+1}=f(H^l,A)
$$
不同模型的差异点在于函数 ***f*** 的实现不同，其中
$$
第一层的输入：H^0=X                    
\\X∈R^{N*D}
$$
***N***为图的节点个数， ***D***为每个节点特征向量的维度， ***A***为邻接矩阵。

普通形式拉普拉斯矩阵
$$
L=D-A
$$
对称归一化的拉普拉斯矩阵更新
$$
L^{sys}=\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}
$$
通常采用下式聚合节点信息
$$
H^{(l+1)}=\sigma(L^{sys}H^{(l)}W^{(l)})
$$

### 1.3 数据集文件

| **数据集** | 样本数 | 边数   | 特征数 | 图数 | 类别数 |
| ---------- | ------ | ------ | ------ | ---- | ------ |
| Cora       | 2708   | 10556  | 1433   | 1    | 7      |
| Citeseer   | 3327   | 9928   | 3703   | 1    | 6      |
| Ppi        | 56944  | 818716 | 50     | 24   | 121    |

### 1.4 dgl库概述

**使用dgl.graph创建一个图**

```
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))   #图中四条边0->1  0->2  0->3  1->3 
```

**直接读取scipy中的稀疏矩阵创建图**

```
g=dgl.from_scipy(adj_matrix)
```

**对结点和边赋予特征**

```
g.ndata['feat'] = feat
g.edata['weight']=weight
```

## 二、节点分类——以cora数据集为例

#### 自环(2层，relu)

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227212440885.png" alt="image-20211227212440885" style="zoom: 67%;" />

无自环：maxvalacc= 0.8845

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227212543730.png" alt="image-20211227212543730" style="zoom:67%;" />

有自环：maxvalacc= 0.8845

#### 层数(无自环，relu)

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227212917561.png" alt="image-20211227212917561" style="zoom:67%;" />

2层网络：maxvalacc= 0.8845

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227212948799.png" alt="image-20211227212948799" style="zoom:67%;" />

3层网络：maxvalacc= 0.8722

#### 激活函数(2层，无自环)

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227212917561.png" alt="image-20211227212917561" style="zoom:67%;" />

relu：maxvalacc= 0.8845

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227213153130.png" alt="image-20211227213153130" style="zoom:67%;" />

sigmoid：maxvalacc= 0.8845

## 三、 链路预测——以citeseer和ppi数据集为例

### citeseer

#### 自环

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227232908198.png" alt="image-20211227232908198" style="zoom:50%;" />

#### 层数

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227232845667.png" alt="image-20211227232845667" style="zoom: 50%;" />

#### 激活函数

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227232923634.png" alt="image-20211227232923634" style="zoom:50%;" />

### ppi数据集

自环

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227233341233.png" alt="image-20211227233341233" style="zoom:50%;" />

层数

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227233404505.png" alt="image-20211227233404505" style="zoom:50%;" />

激活函数

<img src="C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211227233325234.png" alt="image-20211227233325234" style="zoom:50%;" />
