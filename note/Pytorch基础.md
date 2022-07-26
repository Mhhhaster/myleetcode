# Pytorch基础

## tensor

可以利用gpu的numpy

- ### tensor属性

  - dtype![img](https://api2.mubu.com/v3/document_image/6daadb0a-bffe-4af0-81d5-3f0f208ef1da-7811670.jpg)

  - device：cuda or cpu

  - layout

- #### **tensor创建：**

  - 初始化

  ```python
  x=torch.empty(n,m) #全0
  
  x=torch.rand(n,m) #全部在0~1范围
  
  x=torch.ones(n,m) # 全1
  ```

  - 由已有转换：

  ```python
  #指定每个元素
    x=torch.tensor([list])
   #从numpy中获取数据
  	a=np.ones(5)
  	b=torch.from_numpy(a)
  ```

- #### **tensor操作：**

  - **基本运算**

  ```python
  加法：x+y
  	torch.add(x,y)
      y.add_(x)
  减
  乘
  除
  幂
  指数
  对数
  三角函数
  四舍五入
  ```

  - **resize/reshape**

  ```python
  x=torch.randn(4,4)
  y=x.view(-1,8)  
  #y.size()=torch.size([2,8])
  #y为x的视图，内存地址不变
  ```

  - **降维操作**

  ```python
  torch.argmax(input, dim=None, keepdim=False) #返回最大值排序的索引值
  torch.argmin(input, dim=None, keepdim=False)  #返回最小值排序的索引值
  
  torch.cumprod(input, dim, out=None)  #y_i=x_1 * x_2 * x_3 *…* x_i
  torch.cumsum(input, dim, out=None)  #y_i=x_1 + x_2 + … + x_i
  
  torch.dist(input, out, p=2)       #返回input和out的p式距离
  torch.mean()                      #返回平均值
  torch.sum()                       #返回总和
  torch.median(input)               #返回中间值
  torch.mode(input)                 #返回众数值
  torch.unique(input, sorted=False) #返回1-D的唯一的tensor,每个数值返回一次.
  ```

  - **tensor导出**

  ```python
  #获取单元素tensor值
  x=torch.rand(1)
  print(x.item())   
  
  #tensor转numpy
  b=a.numpy()
  
  #注意：转换前后共享内存空间，a.add_(1)后，a为全2 tensor，b为全2 array
  ```

  - **y对x求导**

  ```python
  x=torch,xx(...,requires_grad=True)
  # 默认为false，设为True之后为x.grad额外放置一个空间
  y=f(x)
  
  y.backward()
  
  print(x.grad)
  ```

  - **Join：cat续接 & stack叠加 && gather**

  ```python
  a=torch.Tensor([1,2,3])
  torch.stack((a,a)).size()
  # torch.size(2,3)
  torch.cat((a,a)).size()
  # torch.size(6)
  
  torch.gather(input, dim, index)
  '''
  根据dim维度和索引，获取对应的值
  https://zhuanlan.zhihu.com/p/352877584
  '''
  ```

  - **split & chunk**

  ```python
  >>> a = torch.Tensor([1,2,3])
  >>> torch.split(a,1)  # 每组大小为1
  (tensor([1.]), tensor([2.]), tensor([3.]))
  >>> torch.chunk(a,1)   #切分为1组
  (tensor([ 1., 2., 3.]),)
  ```

## Autograd



![image-20220614171232536](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206141712733.png)
