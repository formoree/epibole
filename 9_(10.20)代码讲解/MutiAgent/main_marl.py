from __future__ import division, print_function
import scipy
import scipy.io
import numpy as np
import Environment_marl
import os
from agent import Agent

# https://blog.csdn.net/m0_37495408/article/details/105587470 代码讲解链接
n_episode = 3000  #迭代次数
n_step_per_episode = 100  #每次迭代有100步
epsilon_final = 0.02  #探索率在前2400迭代从1退到0.02，2400之后保持0.02不变
epsilon_anneal_length = int(0.8*n_episode)  #退火长度2400
mini_batch_step = n_step_per_episode  #100
target_update_step = n_step_per_episode*4  #4个episode复制一次参数，400个迭代步数后更新target_network

n_episode_test = 100  # test episodes

######################################################
# def update_target_network(agent):
#     agent.ddqn.target_net.load_state_dict(agent.ddqn.policy_net.state_dict())

def train(env):
    record_reward = np.zeros([n_episode * n_step_per_episode, 1])  # 记录每一步的奖励
    record_loss = []  # 记录奖励、损失
    for i_episode in range(n_episode):
        print("-------------------------------")
        print('Episode:', i_episode)
        if i_episode < epsilon_anneal_length: #小于退火长度前 探索率改变
            epsilon = 1 - i_episode * (1 - epsilon_final) / (epsilon_anneal_length - 1)  # 前2400幕探索率的下降
        else:
            epsilon = epsilon_final  # 之后不变

        # 这里就是env.update
        if i_episode % 100 == 0:  # 每一100次迭代更新一下信道的位置、邻居、快衰落
            env.renew_positions()  # update vehicle position
            env.renew_neighbor()
            env.renew_channel()  # update channel slow fading
            env.renew_channels_fastfading()  # update channel fast fading

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        # 每次一步具体操作 智能体与环境进行交互产生信息
        for i_step in range(n_step_per_episode):
            time_step = i_episode * n_step_per_episode + i_step  # 总迭代步数
            state_old_all = [] #所有旧状态
            action_all = []  # 所有智能体的动作
            action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')  # 对动作做转化用的

            ###### 给定state（调用get state），从agent_tachi里获取act_all，并使用act_all更新env--------------------#
            ## 更新环境 并将更新信息存储在list中
            for i in range(n_veh):
                for j in range(n_neighbor):  # 根据邻居更新获取一些信息
                    state = Environment_marl.get_state(env, [i, j], i_episode / (n_episode - 1), epsilon)  # 获取state
                    state_old_all.append(state)  # 更新添加state
                    action = agents[i*n_neighbor+j].ddqn.choose_action(state, epsilon)  # 得到预测的action= n_RB*n_power_levels，包含选择的资源块和功率类型
                    action_all.append(action)  # 更新添加action

                    # 动作转化
                    action_all_training[i, j, 0] = action % n_RB  # chosen RB    动作[0,4,8,12]对应RB0，[1,5,9,13]对应RB1，[2,6,10,14]RB2，[3,7,11,15]对应RB3                              #
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level ，向下取整

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)  # 根据预测的action调用环境内的函数获取单步的奖励                                                 #
            record_reward[time_step] = train_reward  # 更新保存单步奖励 时间索引

            # 再次更新环境信息，环境调用到处都是
            env.renew_channels_fastfading()  # 更新快衰落，动作进入环境交互一次就改变一次快衰落（快衰落属于小尺度衰落）多普勒效应引起
            env.Compute_Interference(action_temp)  # 根据预测action计算干扰

            for i in range(n_veh):  # 重新计算每辆
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]  # 上一步的state
                    action = action_all[n_neighbor * i + j]  # 预测的action
                    state_new = Environment_marl.get_state(env, [i, j], i_episode / (n_episode - 1),epsilon)  # 虽然感觉跟上面状态一样，但是这是更新过快衰落和干扰后的新状态，归一化的episode
                    # 将信息放入各个agent的缓存里
                    agents[i * n_neighbor + j].ddqn.memory.add(state_old, state_new, train_reward, action)  # 将车辆更新的信息添加到记忆池

                    # training this agent
                    # 训练开始-> --> ---> ----> -----> ------>
                    if time_step % mini_batch_step == mini_batch_step - 1:  # 每100步训练一次policy_net
                        loss_val_batch = agents[i * n_neighbor + j].ddqn.q_learning_mini_batch()
                        # print(agents[i * n_neighbor + j].ddqn.memory.buffer[6][3])
                        record_loss.append(loss_val_batch)
                        # --------------------------------------------------------打印一些东西 ---------------------------------------------------------
                        if i == 0 and j == 0:
                            print('step:', time_step, 'loss', loss_val_batch,
                                  'reward', record_reward[max(time_step - 2000, 0):time_step].mean())

                    if time_step % target_update_step == target_update_step - 1:  # 400步更新一次target网络
                        # update_target_network(agents[i * n_neighbor + j])
                        agents[i * n_neighbor + j].ddqn.update_target_network()
                        if i == 0 and j == 0:
                            print('Update target Q network...')

    print('Training Done. Saving models...')
    for i in range(n_veh):  # 每次迭代根据车辆更新model_path、save_models
        for j in range(n_neighbor):
            model_path = 'marl_model' + '/agent_' + str(i * n_neighbor + j)
            agents[i * n_neighbor + j].ddqn.save_models(model_path)

    # ---------------------------------------------------------存储模块---------------------------------------------------------------

    current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + 'marl_model' + '/reward.mat')
    scipy.io.savemat(reward_path, {'reward': record_reward})

    record_loss = np.asarray(record_loss).reshape((-1, n_veh * n_neighbor))
    loss_path = os.path.join(current_dir, "model/" + 'marl_model' + '/train_loss.mat')
    scipy.io.savemat(loss_path, {'train_loss': record_loss})


if __name__ == "__main__":
    up_lanes = [i / 2.0 for i in[3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in[250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in[3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in[433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,1299 - 3.5 / 2]]
    # 比例缩小为原来的1/2
    width = 750 / 2  # 地图的长宽
    height = 1298 / 2
    n_veh = 4  # 车辆数
    n_neighbor = 1  # 邻居数
    n_RB = n_veh  # RB个数为4
    env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)  # 环境模拟器
    env.new_random_game()  # 初始化一些环境参数
    agents = []
    for ind_agent in range(n_veh * n_neighbor):  # initialize agents
        print("Initializing agent", ind_agent)
        agent = Agent()
        agents.append(agent)

    train(env)


