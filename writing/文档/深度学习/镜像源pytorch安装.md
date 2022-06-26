# 镜像源解决pytorch安装速度慢的问题

python 安装库一般两种方法：pip 或者 conda

相较而言：pip安装通常版本更新快，依赖检查不严格

进入pytorch官网https://pytorch.org/，根据操作系统和cuda版本选择复制对应的命令

## （cuda版本查看方法）

点击nvidia设置打开nvidia控制面板——帮助——系统信息——组件——查看NVCUDA64.DLL对应产品名称

## conda安装：

conda安装需要设置镜像源

打开cmd命令行，或者anaconda prompt，输入命令`conda config --show channels`查看当前源

![image-20220107174328997](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20220107174328997.png)

我这里已经添加了三条源，添加镜像源方法如下：

```text
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

再删除默认源`conda config --show channels`

![image-20220107174801774](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20220107174801774.png)

到这里镜像源就配置好了，

![image-20220107173851977](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20220107173851977.png)

复制命令，打开cmd命令行，或者anaconda prompt，输入命令

**记得去掉末尾的-c pytorch**

![image-20220107174042509](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20220107174042509.png)

## pip安装

pip不需要配置镜像源，只需要在运行命令时加上`-i https://pypi.tuna.tsinghua.edu.cn/simple`

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果提示当前用户没有写入权限，用管理员权限打开cmd即可