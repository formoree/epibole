from __future__ import division, print_function
import scipy
import scipy.io # import scipy.io
import numpy as np
import Environment_marl
import os
from agent import Agent


def is_test(env):
    print("\nLoading the model...")

    for i in range(n_veh):   #获取训练得到的模型
        for j in range(n_neighbor):
            model_path = 'marl_model' + '/agent_' + str(i * n_neighbor + j)
            agents[i * n_neighbor + j].ddqn.load_models(model_path)

    V2I_rate_list = []
    V2V_success_list = []
    # V2I_rate_list_rand = []
    # V2V_success_list_rand = []
    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    # rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    # demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    # power_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    for idx_episode in range(n_episode_test):  #单次迭代的具体操作
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()  #更新环境参数，位置、邻居、信道、快衰落
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        # env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        # env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        # env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        V2I_rate_per_episode = []
        # V2I_rate_per_episode_rand = []
        for test_step in range(n_step_per_episode):  #单步具体操作
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')  #初始化action
            for i in range(n_veh):  #按车辆更新old_state得到预测的acation、RB、power参数
                for j in range(n_neighbor):
                    state_old = Environment_marl.get_state(env, [i, j], 1, epsilon)
                    action = agents[i*n_neighbor+j].ddqn.choose_action_test(state_old)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()  #根据预测的action获取V2I、V2V的速率（信道容量）、V2V的成功率
            V2I_rate, V2V_success, V2V_rate = env.act_for_testing(action_temp)
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # V2I的总速率（bps），并添加更新
            rate_marl[idx_episode, test_step, :, :] = V2V_rate   #添加更新V2V速率
            demand_marl[idx_episode, test_step+1, :, :] = env.demand  #更新demand

            # # random baseline 随机基线下获取一些参数
            # action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            # action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor]) # band
            # action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor]) # power
            #
            # V2I_rate_rand, V2V_success_rand, V2V_rate_rand = env.act_for_testing_rand(action_rand)
            # V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps
            # rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand
            # demand_rand[idx_episode, test_step+1,:,:] = env.demand_rand
            # for i in range(n_veh):
            #     for j in range(n_neighbor):
            #         power_rand[idx_episode, test_step, i, j] = env.V2V_power_dB_List[int(action_rand[i, j, 1])]

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            if test_step == n_step_per_episode - 1:  #每获取100个（即100步后），添加更新V2V的传输成功率
                V2V_success_list.append(V2V_success)
                # V2V_success_list_rand.append(V2V_success_rand)

        V2I_rate_list.append(np.mean(V2I_rate_per_episode))  #每次迭代后添加更新V2I的速率
        # V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))

        #打印
        # print(round(np.average(V2I_rate_per_episode), 2), 'rand', round(np.average(V2I_rate_per_episode_rand), 2))
        # print(V2V_success_list[idx_episode], 'rand', V2V_success_list_rand[idx_episode])

    print('-------- marl -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    # print('-------- random -------------')
    # print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    # print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 2), 'Mbps')
    # print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
        # f.write('--------random ------------\n')
        # f.write('Rand Sum V2I rate: ' + str(round(np.average(V2I_rate_list_rand), 5)) + ' Mbps\n')
        # f.write('Rand Pr(V2V): ' + str(round(np.average(V2V_success_list_rand), 5)) + '\n')

    #存储
    current_dir = os.path.dirname(os.path.realpath(__file__))
    marl_path = os.path.join(current_dir, "model/" + label + '/rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    # rand_path = os.path.join(current_dir, "model/" + label + '/rate_rand.mat')
    # scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

    demand_marl_path = os.path.join(current_dir, "model/" + label + '/demand_marl.mat')
    scipy.io.savemat(demand_marl_path, {'demand_marl': demand_marl})
    # demand_rand_path = os.path.join(current_dir, "model/" + label + '/demand_rand.mat')
    # scipy.io.savemat(demand_rand_path, {'demand_rand': demand_rand})

    # power_rand_path = os.path.join(current_dir, "model/" + label + '/power_rand.mat')
    # scipy.io.savemat(power_rand_path, {'power_rand': power_rand})


if __name__ == "__main__":
    up_lanes = [i / 2.0 for i in [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]]
    # 比例缩小为原来的1/2
    width = 750 / 2  # 地图的长宽
    height = 1298 / 2
    n_episode_test = 50
    n_step_per_episode = 100
    n_veh = 4
    n_neighbor = 1
    n_RB = 4
    epsilon = 0.02
    label = 'marl_model'
    env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)  # 环境模拟器
    env.new_random_game()  # 初始化一些环境参数
    agents = []
    for ind_agent in range(n_veh * n_neighbor):  # initialize agents
        print("Initializing agent", ind_agent)
        agent = Agent()
        agents.append(agent)

    is_test(env)
