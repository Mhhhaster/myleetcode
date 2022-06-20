## transformer定位：

在encoder-decoder架构中，用multi-head self-attention机制替代RNN

## **训练trick：**

#### 一、优化器：

1. 传统梯度优化

BGD:用到了所有的训练数据，朝着全局最优迭代

SGD:考虑每个样本

MBGD：两者综合，现用SGD同样代表MBGD

2. 防止陷入局部最优

Momentum：考虑了前一次下降的累积，避免 ravines /跳不出局部最优

NAG：超前梯度，先按历史梯度走一小步再考虑当前梯度方向

3. 稀疏数据、稀疏参数大幅更新，频繁参数小幅更新，**自适应调整****lr**

Adagrad：累计梯度平方，作为分母

**Adam优化器√：**momentum一阶梯度+RMSprop指数加权平均梯度

β1 = 0.9, β2 = 0.98 and ε = 10−9.

NAdam：Adam+NAG动量项

#### 二、Warmup（针对自适应优化器）：

由于模型的权重(weights)是随机初始化的，若刚开始训练就选择一个较大的学习率，可能带来模型的不稳定(振荡)。选择Warmup预热学习率的方式，可以使模型可以慢慢趋于稳定，等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快，模型效果更佳。

Residual Dropout 我们将 dropout [33] 应用于每个子层的输出，然后将其添加到子层输入并进行归一化。此外，我们将 dropout 应用于编码器和解码器堆栈中嵌入和位置编码的总和。对于基本模型，我们使用 Pdrop = 0.1 的比率

Label Smoothing During training, we employed label smoothing of value ?ls = 0.1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

标签平滑 在训练过程中，我们采用了值 ?ls = 0.1 [36] 的标签平滑。这会伤害困惑，因为模型会变得更加不确定，但会提高准确性和 BLEU 分数。

#### 三、正则化：

**Residual Dropout：**

每个子层（multi-head self-attention、MLP）输入前、进入LN之前，drop out。

**Label Smoothing(inception v3)：**

对于正确的标签，softmax的输出到0.1就行。

## attention特点：

**LSTM缺点：顺序计算**

隐藏状态ht时序生成，是t时刻输入已经前t-1个时刻隐藏层的函数，计算难以并行；

难以长时记忆

**CNN缺点：感受野小**

每次只关注邻域 kernel size 的区域

**attention：**

对关系进行建模时无需考虑它们在输入或输出序列中的距离，传递全部隐藏状态，计算权重

## Scaled Dot-Product Attentino机制

![image-20220610141919026](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206101419454.png)

QKV都为向量，长度为dk。但是Q的数量不一定等于KV对的向量

**计算QKV**：

QKV都是yi

**计算相似度（权重）：**

微观上一个词的Q和所有词的K点乘，宏观上矩阵乘法。除根号dk（dk=512，防止计算结果过大，导致softmax结果更两极分化）。Mask部分对t时刻之后的结果全部赋值为大负数，则经过softmax之后都为0，则权重为0。

**result=权重*V**

## Multi-Head Attentino机制

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206101435314.png" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206101648444.png" alt="image-20220610164837353" style="zoom: 67%;" />

为什么叫自注意力机制：QKV本质相同，都来源于同一组词向量embedding之后的结果，只不过乘了不同的