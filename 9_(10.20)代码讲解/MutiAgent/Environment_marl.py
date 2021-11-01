from __future__ import division # 导入精确除法
import numpy as np
import time
import random
import math


np.random.seed(1234)


class V2Vchannels:  #V2V信道模拟器
    # Simulator of the V2V Channels

    def __init__(self):  #初始化参数
        self.t = 0  #相干时间
        self.h_bs = 1.5  #基站天线高度
        self.h_ms = 1.5  #车辆天线高度
        self.fc = 2  #载波频率GHZ
        self.decorrelation_distance = 10  #最大想干距离
        self.shadow_std = 3  #V2V阴影衰落标准偏差

    def get_path_loss(self, position_A, position_B):  #获取路径损耗函数，采用曼哈顿网络模型分LOS、NLOS
        d1 = abs(position_A[0] - position_B[0])  #获取两辆车横纵坐标的距离
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001  #获取两辆车的距离，加个微量，避免d = 0的情况，小trick
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)  #有效BP距离

        def PL_Los(d):  #采用了曼哈顿网络布局LOS模型
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:  #超出最大有效距离
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5) 

        def PL_NLos(d_a, d_b):  #采用了曼哈顿网络布局NLOS模型
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:  #若信道为垂直信道时的路径损耗
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()，路损

    def get_shadowing(self, delta_distance, shadowing):  #获取车辆与车辆间的阴影衰落函数，算阴影衰落
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db


class V2Ichannels:  #V2I信道模拟器，

    # Simulator of the V2I channels

    def __init__(self):  #初始化参数函数
        self.h_bs = 25  #基站天线高度
        self.h_ms = 1.5  #车辆天线高度
        self.Decorrelation_distance = 50  #最大相干距离
        self.BS_position = [750 / 2, 1299 / 2]  #基站位置,基站安排在整个图两边的中央
        self.shadow_std = 8  #V2I阴影衰落标准偏差

    def get_path_loss(self, position_A): #获取路径衰落函数，不区分LOS与NLOS
        d1 = abs(position_A[0] - self.BS_position[0])  #获取到基站横纵坐标的绝对值
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)  #车辆到基站的距离
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):  #获取车辆到基站的阴影衰落
        nVeh = len(shadowing)  #获取阴影衰落的大小
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:  #车辆模拟器
    # Vehicle Simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity):  #参数获取函数
        self.position = start_position  #车辆起始的位置
        self.direction = start_direction  #车辆起始的方向
        self.velocity = velocity  #车辆的速度
        self.neighbors = []  #相邻车辆信息
        self.destinations = []  #车辆连接的终端信息


class Environ:  #环境模拟器

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_neighbor):
        self.down_lanes = down_lane  #下车道
        self.up_lanes = up_lane  #上车道
        self.left_lanes = left_lane  #左车道
        self.right_lanes = right_lane  #右车道
        self.width = width  #地图宽度
        self.height = height  #地图高度

        self.V2Vchannels = V2Vchannels()  #V2V信道信息
        self.V2Ichannels = V2Ichannels()  #V2I信道信息
        self.vehicles = [] #车辆信息

        self.demand = []  #负载
        self.V2V_Shadowing = []  #V2V信道衰落
        self.V2I_Shadowing = []  #V2I信道衰落
        self.delta_distance = []  #三角距离？
        self.V2V_channels_abs = []  #V2V路径损耗与阴影衰落的和
        self.V2I_channels_abs = []  #V2I路径损耗与阴影衰落的和

        self.V2I_power_dB = 23  # 单位dBm V2I信道传输功率
        self.V2V_power_dB_List = [23, 15, 5, -100]  # V2V信道的4个级别传输功率
        self.V2I_power = 10 ** (self.V2I_power_dB)  # V2V功率
        self.sig2_dB = -114  #噪声功率db
        self.bsAntGain = 8  #基站天线增益
        self.bsNoiseFigure = 5  #基站接收机噪声
        self.vehAntGain = 3  #车辆天线增益
        self.vehNoiseFigure = 9  #车辆接收机噪声
        self.sig2 = 10 ** (self.sig2_dB / 10)  #噪声功率w

        self.n_RB = n_veh  # 资源块
        self.n_Veh = n_veh  #车辆数
        self.n_neighbor = n_neighbor  #邻居的个数
        self.time_fast = 0.001  #快衰落更新时间为1ms
        self.time_slow = 0.1   #慢衰落更新时间为100 ms
        self.bandwidth = int(1e6)  #每个RB的带宽，1 MHz
        # self.bandwidth = 1500
        self.demand_size = int((4 * 190 + 300) * 8 )  #V2V有效负载：每100毫秒1060×2字节
        # self.demand_size = 20

        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2  #对于V2V来说所有的干扰功率

    def add_new_vehicles(self, start_position, start_direction, start_velocity):  #添加车辆信息函数,包括起始位置、方向、车速
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):  #获取N个车辆信息

        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))  #车道信息，有问题，为啥只有一个ind
            
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]  #位置
            start_direction = 'd'  #下车道 #方向
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))  #速度随机取10〜15 m / s

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            
            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        #初始化信道
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

    def renew_positions(self):  #更新车辆位置
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):  #遍历每辆车
            delta_distance = self.vehicles[i].velocity * self.time_slow  #根据速度改变位置
            change_direction = False  #False表示要改变方向，True表示不改变方向
            if self.vehicles[i].direction == 'u':  #上方向改变信息
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):  #是否左转

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  #来到左侧十字路口
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:  #是否右转
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:  #直行
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):  #下方向改变信息
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):  #是否左转
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:  #是否右转
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:  #直行
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):  #右方向改变方向信息
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):  #是否上转
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:  #是否下转
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:  #直行
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):  #左方向改变信息
                for j in range(len(self.up_lanes)):  #是否上转

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:  #是否下转
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:  #直行
                        self.vehicles[i].position[0] -= delta_distance

            # 到达地图边界，顺时针转向留在地图上
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):  #上方向改为右方向
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):  #下方向改为左方向
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):  #左方向改为上方向
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):  #右方向改为下方向
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1  #遍历

    def renew_neighbor(self):  #更新邻居信息（看不太懂）
        """ Determine the neighbors of each vehicles """

        #计算distance
        for i in range(len(self.vehicles)):  #遍历每辆车
            self.vehicles[i].neighbors = [] 
            self.vehicles[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]]) #complex转成复数 a + bj
        Distance = abs(z.T - z)  #z的转置-z

        # 将数组升序排列，但不改变数组，且返回对应的索引
        # import numpy as np
        # a = np.array([4, 2, 5, 7])
        # b = a.argsort()  # 将数组升序排列，但不改变数组，且返回对应的索引
        # print(a)  # [4 2 5 7]，其索引是[0,1,2,3]
        # print(b)  # 升序后的索引是[1 0 2 3]，对应元素[2,4,5,7]

        for i in range(len(self.vehicles)):  #遍历每辆车
            sort_idx = np.argsort(Distance[:, i])# 进行排序 左行右列 升序
            for j in range(self.n_neighbor):  # 更新邻居
                self.vehicles[i].neighbors.append(sort_idx[j + 1])  #添加邻居
            destination = self.vehicles[i].neighbors  #

            self.vehicles[i].destinations = destination  #

    def renew_channel(self):  #更新慢衰落信道
        """ Renew slow fading channel """

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))  #（k,k ）
        self.V2I_pathloss = np.zeros((len(self.vehicles)))  #（k，）

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j,i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing  #包含路径损耗和阴影衰落

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_channels_fastfading(self):  #更新快衰落信道
        """ Renew fast fading channel """

        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape))/ math.sqrt(2))

    def Compute_Performance_Reward_Train(self, actions_power):  #计算训练奖励，输入动作

        # 是个三维数组，以（层，行，列）进行说明，一层一个车，一行一个邻居，共有两列分别为RB选择（用RB的序号表示）和power选择（也用序号表示，作为power_db_list的索引）
        actions = actions_power[:, :, 0]  #资源块
        power_selection = actions_power[:, :, 1]  # 功率

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)  #初始化V2I信道容量为0
        V2I_Interference = np.zeros(self.n_RB)  #初始化干扰为0 
        for i in range(len(self.vehicles)):  #遍历每个车辆  
            for j in range(self.n_neighbor):  #遍历每个邻居
                if not self.active_links[i, j]:  #每行为车，每列为邻居，是否存在连接，存在则为干扰信道
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)  #V2V干扰
        self.V2I_Interference = V2I_Interference + self.sig2  #来自V2V链路干扰和噪声干扰的总功率
        V2I_Signals = 10 ** ((self.V2I_power_dB -  self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)  #信号功率
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))  #计算V2I的信道容量

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))  #初始化干扰为0
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))  #初始化信号功率为0
        actions[(np.logical_not(self.active_links))] = -1  #没有激活的链路不传输，设为-1
        for i in range(self.n_RB):  #遍历所有频段
            indexes = np.argwhere(actions == i)  #查找频谱共享的V2V
            for j in range(len(indexes)):  #遍历共享频谱的V2V
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                #V2V信号功率
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                #V2I链路对V2V链路的干扰
                V2V_Interference[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #V2V链路之间的干扰
                for k in range(j + 1, len(indexes)):  # 遍历频谱共享的V2V    不懂怎么有两个V2V干扰
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2  
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))  #计算V2V信道容量

        self.demand -= V2V_Rate * self.time_fast * self.bandwidth  #计算剩余负载
        self.demand[self.demand < 0] = 0  #剩余负载小于零，置为0

        self.individual_time_limit -= self.time_fast  #剩余时间

        reward_elements = V2V_Rate/10  #除以10归一化 计算reward
        reward_elements[self.demand <= 0] = 1  #当剩余负载为负，将每个时间步的V2V相关奖励设置为常数1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0  #剩余负载为负，传输完成，链路变为无效

        return V2I_Rate, V2V_Rate, reward_elements  #返回V2I、V2V信道容量、单个时间步奖励 最终的reward需要把这三个数值加权组合起来

    def Compute_Performance_Reward_Test_rand(self, actions_power):  #在随机基线下计算单个时间步长的奖励
        """ for random baseline computation """

        actions = actions_power[:, :, 0]  #决策
        power_selection = actions_power[:, :, 1]  #功率选择

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)  #初始化V2I信道容量
        V2I_Interference = np.zeros(self.n_RB)  #初始化V2I干扰
        for i in range(len(self.vehicles)):  #遍历每个车辆 
            for j in range(self.n_neighbor):  #遍历每个邻居
                if not self.active_links_rand[i, j]:  #没有激活的链路不传输，设为-1
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference_random = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference_random))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):  #遍历所有频段
            indexes = np.argwhere(actions == i)  #找到频谱共享的V2V
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                #V2I链路对V2V的干扰
                V2V_Interference[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #V2V链路之间的干扰
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_random = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference_random))

        self.demand_rand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand_rand[self.demand_rand < 0] = 0

        self.individual_time_limit_rand -= self.time_fast

        self.active_links_rand[np.multiply(self.active_links_rand, self.demand_rand <= 0)] = 0 # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate  #为啥不返回奖励呢

    def Compute_Interference(self, actions):  #计算V2V所有干扰
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2  #初始化V2V干扰为0

        channel_selection = actions.copy()[:, :, 0]  #信道选择
        power_selection = actions.copy()[:, :, 1]  #功率选择
        channel_selection[np.logical_not(self.active_links)] = -1  #无效链路置为-1

        # V2I对V2V的干扰
        for i in range(self.n_RB):  
            for k in range(len(self.vehicles)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # V2V之间的干扰
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:  #若为自己本链路或者链路无效无干扰
                            continue
                        V2V_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]]
                                                                                   - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)  #换算功率


    def act_for_training(self, actions):  #执行训练，返回单次的奖励

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)  #调用函数获取V2I、V2V信道容量和单个时间步的奖励

        lambdda = 0.
        reward = lambdda * np.sum(V2I_Rate) / (self.n_Veh * 10) + (1 - lambdda) * np.sum(reward_elements) / (self.n_Veh * self.n_neighbor)  #最终的奖励

        return reward
    def act_for_testing(self, actions):  #执行测试，返回V2I、V2V信道容量、V2V成功率

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)   #调用函数获取V2I、V2V信道容量和单个时间步的奖励
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  #V2V成功率

        return V2I_Rate, V2V_success, V2V_Rate

    def act_for_testing_rand(self, actions):  #在随机基线下执行测试

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)  #调用函数获取V2I、V2V信道容量和单个时间步的奖励
        V2V_success = 1 - np.sum(self.active_links_rand) / (self.n_Veh * self.n_neighbor)  #V2V成功率

        return V2I_Rate, V2V_success, V2V_Rate

    def new_random_game(self, n_Veh=0):
        # 初始化一些参数

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))  #将车辆分到4个车道
        self.renew_neighbor()  #更新邻居
        self.renew_channel()  #更新信道信息
        self.renew_channels_fastfading()  #更新快衰落

        self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')

        # 随机基线
        self.demand_rand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit_rand = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links_rand = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')


def get_state(env, idx=(0,0), ind_episode=1., epsilon=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])  #剩余负载
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])  #剩余时间

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    #np.concatenate((a1,a2...), axis=0/1(0按列拼接，1按行拼接))对数组进行拼接
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsilon])))

def get_sac_state(env, idx=(0,0), ind_episode=1.):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])  #剩余负载
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])  #剩余时间

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    #np.concatenate((a1,a2...), axis=0/1(0按列拼接，1按行拼接))对数组进行拼接
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode])))


