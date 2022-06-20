# Yolo v1-v5

## 0. U need to know in advance:

#### **图像处理三大任务**：

**分类：**输出一张图片的类别）

**检测：**输出一个列表，每一项使用一个数据组给出检出目标的类别和位置（常用矩形检测框的坐标表示）

**分割：**如车道线分割。分割是对图像的像素级描述，它赋予每个像素类别（实例）意义

​	传统语义分割基于灰度值的不连续特性，基于深度学习的图像分割用CNN理解每个像素

​	**语义分割：**抠图，并分离开具有不同语义的图像部分，为图像中的每个像素分配一个类别，但是同一类别之间的对象不会区分。

​	**实例分割：**检测任务的拓展，要求描述出目标的轮廓（相比检测框更为精细），目标检测输出目标的边界框和类别，实例分割输出的是目标的Mask和类别。

<img src="https://pic2.zhimg.com/80/v2-f5aa16e33a671378bb49b5e866c3d3a5_720w.jpg" alt="img" style="zoom: 67%;" />

#### 目标检测算法类别：

**Two-stage**（基于区域region-based）：先生成样本候选框(有可能包含待检测物体)，再通过CNN进行样本分类

​	**R-CNN：**将检测任务转化为区域上的分类任务

​		使用选择性搜索来从一张图片中创建多个边界框。

​	**Fast R-CNN：**

**One-stage：**

​	特点：直接回归物体的类别概率和位置坐标值，无RPN网络（region proposal），准确度低，快。

​	模型：v1，v2，v3，SSD，SqueezeDet以及DetectNet

#### 常见缩写

#### 	**mAP**：

​		mAP：mean Average Precision，即各类别AP的平均值
​		AP：PR曲线下面积

#### 	**FPS：**Frames Per Second，每秒处理帧数

#### 	**NMS：非极大值抑制（Non-Maximum suppression）**

目标检测算法中一个必要的后处理过程，**目的是消除同一个物体上的冗余预测框。**

- NMS算法的主要思想是：**以置信度最大**的边界框为target，分别计算target与其余剩下的预测框的重叠程度（用IoU来衡量），若重叠程度大于某一预先设定的阈值，则认为该预测框与target是同时负责预测同一个物体的，所以将该边界框删除，否则予以保留。接着在未被删除的预测框中选择分数最高的预测框作为新target，重复以上过程，直至判断出所有的框是否应该删除。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201107143950531.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU2MDQwMg==,size_1,color_FFFFFF,t_70#pic_center)
- Soft-NMS：不是直接根据IOU删去，而是为其他预测框重新打分
- Adaptive NMS：IOU阈值随环境密集程度变换
- IOU-Net：认为排序时单纯按照置信度不合理，需要综合考虑其与ground-truth的IOU

## 1. v1：

**核心思想：**(将分类问题转化为回归问题)

1.将一幅图像划分为 S x S 个网格(grid cell)，如果某个物体的中心落在某网格内，则其预测任务由该网格负责

2.每个网格预测： B 个bounding box（每个bounding box预测位置和confidence）+C个类别的分数

​	4个位置：v1中没有anchor概念，因此都是直接相对于图像，而不是相对anchor，均(0,1)

​	confidence: Yolo独有，网格中无目标为0，有目标即预测边界框与真实边界框IOU

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206131009882.png" alt="image-20220613100858716" style="zoom:50%;" />

**损失函数：**

![image-20220613111255727](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206131112825.png)

**特点：** anchor-free、45FPS

**缺点：**

群体小目标检测效果差（一个网格只能预测两个bounding box，且属于同一类别）

对于新出现的目标或者新尺寸的目标效果差

定位不准确（没有anchor）

## 2. V2(YOLO9000):

**特点：**能检测超过9000个类别

**backbone：**Darknet-19(19个卷积层)

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206131449586.png" alt="image-20220613144924499" style="zoom:50%;" />

**网络结构：**

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206131451215.png" alt="image-20220613145132069" style="zoom:67%;" />

（位置+confidence+类别概率）* 5个anchor

#### **改进：**

**BN：**每个卷积层添加BN层，正则化作用，可以移除dropout层，提升了2%

**High Resolution classifier：**224 x 224 —>448 x 448，提升了4%

**anchor boxes：**基于anchor偏移的预测能够使网络更容易收敛，提高了召回率

**dimension cluster(anchor聚类)：**基于k-means方法得到anchor

**direct location prediction：**

**fine-grained features：**将低维特征和高维特征融合

**multi-Scale training：**每迭代十个batch，对输入图像尺寸随机缩放{320，352，....，608}，增加鲁棒性

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206131144392.png" alt="image-20220613114433300" style="zoom:50%;" />

#### 实现细节：

如何匹配正负样本？

如何计算误差？

权重衰减

数据增强处理同v1：random crops，color shifting

## v3：

**特点：**

**backbone：**darknet-53（用卷积层替换pooling层的resnet）

<img src="https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206132001507.png" alt="image-20220613200119430" style="zoom:50%;" />

**网络结构：**

![image-20220613200449763](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206132004844.png)

同样每层卷积包括三个部分：conv2d+BN+LeakyReLU

![image-20220613185827095](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206131858144.png)

残差连接：

![image-20220613200538624](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206132005674.png)

**思想：**在三个预测特征层scales上进行预测，每层上预测3个boxes （通过k-means得到）

其中boxes =bounding box ≈anchor（与SSD和RCNN区别：）

N x N x[3*(4+1+80)]:

4表示基于anchor offset

1表示独有的confidence

80表示类别概率

**正负样本匹配：**

针对每一个GT都会分配一个bounding box prior（一个正样本），如果一个bounding box prior与GT的重叠值最大，那么令其为一；如果与GT的重叠值不是最大但超过某个阈值，直接丢弃；其余的视作负样本。

**损失计算：**置信度损失+分类损失+定位损失



定位损失

## SPP：

**mosaic图像增强：**随机选取四张图像进行组合

​	增加了数据的多样性

​	增加目标个数

​	BN能一次性统计多张图片的参数

**SPP模块：**借鉴于SPPnet，拆开了convolutional set，插入一个SPP结构。

stride都为1，需要填充，且卷积前后尺寸不变，concat后实现不同尺度特征的融合（深度四倍）

![image-20220613204132991](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206132041097.png)

**CloU Loss**

​	IOU loss=1-IOU  or -ln IOU

​	GIOU loss=

​	todo

**Focal Loss** todo

