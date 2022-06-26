# 一、Policy-Based——Policy Gradient

## 1.1 算法简介

mc，sarsa，q-learning是value-based方法：通过对价值函数（包括状态价值函数和行为价值函数）进行近似的参数化表达，然后使用*ϵ*-greedy可以直接从价值函数中产生一个策略。

Policy Gradient是Policy-Based方法：直接参数化策略本身，同时参数化的策略将不再是一个概率集合而是一个函数π<sub>θ</sub>(s,a) = P[a|s, θ]。Softmax Policy是离散策略，Gaussian Policy则针对连续空间

### 三种目标函数：

Start value：从该状态开始到 Episode 结束，个体将会得到怎样的期望reward

Average Value：连续环境不存在开始状态，对该时刻各状态的概率分布求和

Average reward per time-step：在一个时间步长里，查看个体处于所有状态的可能性，然后每一种状态下采取所有行为能够得到的即时奖励，所有奖励按概率求和得到

### 策略梯度定理：

![image-20211220213829703](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211220213829703.png)

## 1.2 核心代码注释

实现的函数介绍

①

```
def preprocess(image)   
```

输入：图片   

输出：6400*1的一维向量

作用：预处理

②

```
def calc_discounted_future_rewards(rewards, discount_factor):
    discounted_future_rewards = torch.empty(len(rewards))
    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        if rewards[t] != 0:    #说明此时游戏已决出胜负
            discounted_future_reward = 0
        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward
        #更新公式：discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]
    return discounted_future_rewards 
```

输入：折扣因子，reward

输出&作用：计算每一个时间步的未来折扣奖励

③

```
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        prob_up = torch.sigmoid(x)
        return prob_up
```

一个简单的两层神经网络，输入input_size和hidden_size，输出经过sigmoid得到一个概率

④

```
def run_episode(model, env, discount_factor, render=False):
    UP = 2
    DOWN = 3
    observation = env.reset()
    prev_x = preprocess(observation)
    action_chosen_log_probs = []
    rewards = []
    done = False
    timestep = 0
    while not done:
        if render:
        	.....
        	prob_up = model(x)
        	action = UP if random.random() < prob_up else DOWN
        	#对动作进行采样
	    	action_chosen_prob = prob_up if action == UP else (1 - prob_up)
        	action_chosen_log_probs.append(torch.log(action_chosen_prob))
        	#计算被选中的动作的对数概率，并加入列表
        	....
    loss = -(discounted_future_rewards * action_chosen_log_probs).sum()
    #目标函数Average reward per time-step：所有奖励按概率求和
    return loss, rewards.sum()
```

## 1.3 训练曲线

![image-20211220214210584](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211220214210584.png)

![image-20211220214139134](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211220214139134.png)

## 1.4 实现内容

# 二、A2C

## 2.1 算法简介

Actor-Critic型的算法需要同时训练价值网络以及策略网络，本质上还是基于策略的方法，因为其算法的核心还是在不断地优化策略。虽然我们要训练价值网络，但是其目标也只是“辅佐”策略网络更好地训练。A2C简而言之就是能让训练数据之间前后关联较弱、独立性较强的Actor-Critic。

## 2.2 核心代码注释

```
class TwoHeadNetwork(nn.Module):                    
    def __init__(self, input_dim):
        super(TwoHeadNetwork, self).__init__()
        self.feat = nn.Linear(input_dim, 200)       #两个网络可以部分共享
        self.policy = nn.Linear(200, 1)             #给出当前状态在不同动作发生的概率
        self.value  = nn.Linear(200, 1)             #当前状态的值函数
        
    def forward(self, x):
        x = F.relu(self.feat(x))
        
        logits = self.policy(x)
        logits = torch.sigmoid(logits)
        
        value = F.relu(self.value(x))

        return logits, value
```

train函数、calc_discounted_future_rewards函数和preprocess函数基本一样，细节不表

```
def run_episode(model, env, discount_factor, render=False):
    UP = 2
    DOWN = 3
    observation = env.reset()
    prev_x = preprocess(observation)
    action_chosen_log_probs = []
    value_esti=[]         #与pg算法中的不同之处
    rewards = []
    done = False
    timestep = 0
    while not done:
        if render:
        	.....
        	prob_up, values = model(x)  #model(x)有两个返回值：该状态下的估计的策略和值函数
        	action = UP if random.random() < prob_up else DOWN
        	#对动作进行采样
	    	action_chosen_prob = prob_up if action == UP else (1 - prob_up)
        	action_chosen_log_probs.append(torch.log(action_chosen_prob))
        	#计算被选中的动作的对数概率，并加入列表
        	value_esti.append(torch.Tensor([values]))   #需要额外维护value_esti
        	....
    value_esti = (value_esti - value_esti.mean()) / value_esti.std()  #同上
    loss = -(discounted_future_rewards * action_chosen_log_probs).sum()
    #目标函数Average reward per time-step：所有奖励按概率求和
    return loss, rewards.sum()
```

## 2.3 训练曲线

![image-20211220214028899](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211220214028899.png)

![image-20211220214037277](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211220214037277.png)

# 三、两个算法优缺点对比

![image-20211220214327435](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20211220214327435.png)

A2C训练慢，但收敛快