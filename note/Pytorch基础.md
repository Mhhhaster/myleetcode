# Pytorch基础

### tensor=可以利用gpu的numpy

**创建：**

```
全0：x=torch.empty(n,m)

全0~1：x=torch.rand(n,m)

全1：x=torch.ones(n,m)

由已有转换：
	指定每个元素: x=torch.tensor([list])
	a=np.ones(5)
	b=torch.from_numpy(a)
```

**计算：**

```python
加法：x+y
	torch.add(x,y)
    y.add_(x)
```

**resize/reshape**：

```python
x=torch.randn(4,4)
y=x.view(-1,8)  
#y.size()=torch.size([2,8])
```

**tensor转换**

```
#获取单元素tensor值
x=torch.rand(1)
print(x.item())   

#tensor转numpy
b=a.numpy()

注意：转换前后共享内存空间，a.add_(1)后，a为全2 tensor，b为全2 array
```

**y对x求导**

```
x=torch,xx(...,requires_grad=True)
# 默认为false，设为True之后为x.grad额外放置一个空间
y=f(x)

y.backward()

print(x.grad)
```

### CUDA

**gpu使用**

```
if torch.cuda/is_available():
	device=torch.device("cuda")
	y=torch.ones_lie(x,device=device)
	x=x.to(device)
	# x、y都在GPU上，计算结果也会在GPU上
```

![image-20220614171232536](https://raw.githubusercontent.com/Mhhhaster/for_picgo/main/202206141712733.png)
