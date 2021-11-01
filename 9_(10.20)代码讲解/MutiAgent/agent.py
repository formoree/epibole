import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_memory import ReplayMemory
import os
import copy


V2V_power_dB_List = [23, 15, 5, -100]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """ 初始化q网络，为全连接网络
            input_dim: state的维度
            output_dim: 输出的action个数
        """
        super(MLP, self).__init__() #super:调用父类的方法
        # super().__init__()        #都可用
        # nn.Module.__init__(self)
#         self.fc1 = nn.Linear(input_dim, hidden_dim[0])  # 输入层
#         self.fc2 = nn.linear(hidden_dim[0], hidden_dim[1])
#         self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])  # 隐藏层
#         self.fc4 = nn.Linear(hidden_dim[2], output_dim)  # 输出层
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]), #[batch_size, size]全连接
            nn.BatchNorm1d(hidden_dim[0]), #加入可训练的参数做归一化
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(hidden_dim[2]),
            nn.ReLU())
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim[2], output_dim),
            nn.ReLU())  #

    def forward(self, x):
        # 各层对应的激活函数
        hidden_1_out = self.fc1(x)
        hidden_2_out = self.fc2(hidden_1_out)
        hidden_3_out = self.fc3(hidden_2_out)
        return self.fc4(hidden_3_out)

n_RB = 4

class Agent:
    def __init__(self):
        self.gamma = 0.95
        self.lr = 0.001
        self.input_dim = 33
        self.output_dim = n_RB * len(V2V_power_dB_List)
        self.hidden_dim = [500, 250, 120]
        self.ddqn = DDQN(self.input_dim, self.output_dim, self.hidden_dim)


class DDQN:
    # DoubleDQN其实就是Double-Qlearning在DQN上的拓展，上面Q和Q2两套Q值，
    # 分别对应DQN的policy network（更新的快）和target network（每隔一段时间与policy network同步）
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测gpu，没GPU用CPU
        self.lr = 0.001
        self.gamma = 1
        self.batch_size = 2000
        self.policy_net = MLP(input_dim, output_dim, hidden_dim).to(self.device)
        # self.policy_net.eval()
        self.target_net = MLP(input_dim, output_dim, hidden_dim).to(self.device)
        # self.target_net =copy.deepcopy(self.policy_net)
        self.target_net.eval()
        # self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr, momentum=0.95, eps=0.01 )
        self.memory = ReplayMemory()

    def choose_action(self, state, epsilon, test=False):  # 选动作区分智能体
        # n_power_levels = len(env.V2V_power_dB_List)
        n_power_levels = 4
        if np.random.rand() < epsilon and not test: #(产生随机数)
            action = np.random.randint(4 * n_power_levels)
        else:
            #放入policy_net中寻找动作 没有做过max 不会高估计
            with torch.no_grad():
                # 先 转为张量 便于丢给神经网络,state元素数据原本为float64
                # 注state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                state = torch.tensor([state], device=self.device, dtype=torch.float32)  # 升维度
                # state = torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1)).to(self.device))
                # print(state)
                # 如tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)

                #适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数
                # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min - batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
                self.policy_net.eval()
                q_value = self.policy_net(state)
                # q_value = self.policy_net(torch.from_numpy(np.array(state, dtype= np.float32).reshape(1,-1)).to(self.device))  # 输入状态从神经网络获取q值
                # tensor.max(1)返回每行的最大值以及对应的下标，
                # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                # 所以tensor.max(1)[1]返回最大值对应的下标，即action
                action = q_value.max(1)[1].item()  # .item()的用法是取出元素，但保持它以前的精度

        return action

    def choose_action_test(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            # self.target_net.eval()
            q_value_test = self.target_net(state)
            action = q_value_test.max(1)[1].item()

        return action


    def q_learning_mini_batch(self):  # 学习区 智能体
        # if len(self.memory) < self.batch_size:
        #     return
        state_batch, next_state_batch, reward_batch, action_batch = self.memory.sample()  # 选了一批的sars
        '''转为张量
        例如tensor([[-4.5543e-02, -2.3910e-01,  1.8344e-02,  2.3158e-01],...,[-1.8615e-02, -2.3921e-01, -1.1791e-02,  2.3400e-01]])'''

        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)


        # 计算当前(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch)
        next_q_values = self.policy_net(next_state_batch)
        # 代入当前选择的action，得到Q(s_t|a=a_t)
        q_value = q_values.gather(dim=1, index=action_batch) #然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        next_target_values = self.target_net(next_state_batch).detach() # 不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        # 选出Q(s_t‘, a)对应的action，代入到next_target_values获得target net对应的next_q_value，即Q’(s_t|a=argmax Q(s_t‘, a))
        #squeeze()的作用就是压缩维度，直接把维度为1的维给去掉。
        #unsqueeze()的作用是用来增加给定tensor的维度的，unsqueeze(dim)就是在维度序号为dim的地方给tensor增加一维
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_batch + self.gamma * next_target_q_value
        self.loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))  # 计算 均方误差loss
        output_loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))
        self.policy_net.train() #进行训练
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward() #反向传播
        self.optimizer.step()  # 更新评估网络的所有参数
        # 既然在BP过程中会产生梯度消失（就是偏导无限接近0，导致长时记忆无法更新）或梯度爆炸，那么最简单粗暴的方法就是，梯度截断Clip, 将梯度约束在某一个区间之内
        # 注意这个方法只在训练的时候使用，在测试的时候验证和测试的时候不用。
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 20
        )
        # 为了实现自动微分，PyTorch跟踪所有涉及张量的操作，可能需要为其计算梯度（即require_grad为True）。
        # 这些操作记录为有向图。 detach（）方法在张量上构造一个新视图，该张量声明为不需要梯度，即从进一步跟踪操作中将其排除在外，因此不记录涉及该视图的子图。

        # 深度学习模型用GPU训练数据时，需要将数据转换成tensor类型，输出也是tensor类型。
        return output_loss.detach().numpy()

    def update_target_network(self):
        self.target_net.parameters = copy.deepcopy(self.policy_net.parameters)
        # self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_models(self, model_path):
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_path, "model/" + model_path) #路径拼接
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.target_net.state_dict(), model_path + 'target_net_checkpoint.pth')
        torch.save(self.policy_net.state_dict(), model_path + 'policy_net_checkpoint.pt')

    def load_models(self, model_path):
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_path, "model/" + model_path)
        self.target_net.load_state_dict(torch.load(model_path + 'target_net_checkpoint.pth'))
        self.policy_net.load_state_dict(torch.load(model_path + 'policy_net_checkpoint.pt'))
        self.target_net.to(self.device)