import numpy as np
import random


class ReplayMemory:
    def __init__(self):  # 初始化，输入记忆池的容量
        self.memory_size = 200000
        self.batch_size = 2000
        self.buffer = []
        self.position = 0

    def add(self, prestate, poststate, reward, action):  # 添加当前状态、下一状态、决策行为和奖励
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (prestate, poststate, reward, action)
        self.position = (self.position + 1) % self.memory_size #训练集位置
        # print(self.buffer[0])

    def sample(self):  # 采样，通过添加存在多组添加的信息，训练时取出batch_size个进行训练

        if len(self.buffer) < self.batch_size:
            indexes = self.buffer  # 没存满也训练
            # print(len(self.buffer) < self.batch_size)
        else:
            indexes = random.sample(self.buffer, self.batch_size) #sample(list, k)返回一个长度为k新列表，新列表存放list所产生k个随机唯一的元素
        prestate, poststate, rewards, actions = zip(*indexes) #利用 * 号操作符，可以将元组解压为列表
        # print(actions)
        return prestate, poststate, rewards, actions

    def __len__(self):
        return len(self.buffer)