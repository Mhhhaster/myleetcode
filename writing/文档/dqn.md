```
VALID_ACTIONS = [0, 1, 2, 3]
```



```
class Estimator():#同时适用Q-Network和Target Network
	def __init__(self, scope="estimator", summaries_dir=None):
	*  *
	def _build_model( self ):# 三层CNN，输入图片state返回选中的动作的预测值
	*   *
	def predict(self, sess, s):  #基于观测返回动作价值[batchsize,num_action]
		return sess.run(self.predictions, { self.X_pl: s }) #在当前状态S下
	def update(self, sess, s, a, y):  #pred更新为target，返回loss
```



```
def make_epsilon_greedy_policy(estimator, num_action):  
	#estimator可以根据state返回value：根据图片返回选中动作的值
	本函数返回一个数组，每个动作的执行概率
```



```
def deep_q_learning(pramas):
	Returns：返回episode长度和汇报的两个数组
	Transition是sarsd元组
	total_t通过get_global_step()获取当前时间步
	policy传入q估计网络和动作空间数
	1. 填充回放池
	用monitor记录视频
	for i_episode in range(num_episodes):#对于每一个episode
		重置环境并process，重置loss
		for t in itertools.count():  #对于一局游戏中的每一步
			2.每1万个total_t就copy_model_parameters，q_est->target_est
			3.与环境交互一步step(action)
			如果回放池满了就弹出一个
			4. 传递transition到回放池中
			5. 从回放池中采样一个minibatch
			6. 使用minibatch计算q值和目标值
```



```
def populate_replay_buffer():#比较paddle和tensorflow
	
```