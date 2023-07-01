"""Parts of codes are brought from https://github.com/NetManAIOps/OmniAnomaly"""

import ast
import csv
import os
import sys
from pickle import dump
import json

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


"""
这段代码定义了一个名为load_as_np的函数，用于将数据集中的文件加载为NumPy数组。

函数接收多个参数，包括类别（category）、文件名（filename）、数据集（dataset）、数据集文件夹（dataset_folder）和输出文件夹（output_folder）。

首先，使用os.path.join函数将数据集文件夹、类别和文件名拼接成完整的文件路径。

然后，使用np.genfromtxt函数从文件中读取数据。函数指定了数据类型为np.float32，分隔符为逗号。

最后，将读取的数据存储在变量temp中，并返回该变量。
"""
def load_as_np(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    return temp

"""
这段代码定义了一个名为load_data的函数，用于加载数据集并保存为NumPy数组和JSON文件。

函数接收多个参数，包括数据集名称（dataset）、基本目录（base_dir）、输出文件夹（output_folder）和JSON文件夹（json_folder）。

首先，根据数据集名称判断数据集类型，若为'SMD'，则设置数据集文件夹为'OmniAnomaly/ServerMachineDataset'。

    接下来，列出数据集文件夹中'train'目录下的所有文件，并遍历文件列表。

    对于以'.txt'结尾的文件，分别调用load_as_np函数加载训练数据、测试数据和标签数据，并将它们存储在train_files、test_files和label_files列表中。

    然后，根据文件的索引，将训练数据、测试数据和标签数据保存为NumPy数组文件。

    接着，使用np.concatenate函数将train_files、test_files和label_files列表中的数组沿指定轴进行拼接，并将拼接后的数组保存为NumPy数组文件。

    之后，根据文件长度列表计算出通道划分，并将通道划分保存为JSON文件。
    
如果数据集是'SMAP'或'MSL'，则将数据集文件夹设置为'base_dir/telemanom/data'。

    然后，打开数据集文件夹中的'labeled_anomalies.csv'文件，并使用csv.reader读取文件内容。将读取的内容存储在res列表中，并去掉第一行的标题。

    对res列表按照日期进行排序。

    接下来，根据数据集信息筛选出与数据集名称相匹配的数据。然后，根据异常的起始和结束位置创建标签，并将标签存储在labels列表中。

    对于每个数据集信息，根据类别和长度信息更新类别划分和通道划分。并将通道划分保存在channel_divisions列表中。

    将labels列表转换为NumPy数组，并将其保存为NumPy数组文件。

    接着，将类别划分和通道划分保存为JSON文件。定义了一个名为concatenate_and_save的函数，用于拼接数据并保存。

    对于每个数据集信息，加载相应的数据文件，并将其拼接到data列表中。

    将data列表转换为NumPy数组，并使用MinMaxScaler对数据进行归一化。

    将拼接后的数据保存为NumPy数组文件。

对于数据集'SWaT'，加载正常数据和异常数据，并将它们分别保存为NumPy数组文件。

对于数据集'WADI'，加载正常数据和异常数据，并将它们分别保存为NumPy数组文件。
"""
def load_data(dataset, base_dir, output_folder, json_folder):
    if dataset == 'SMD':
        dataset_folder = os.path.join(base_dir, 'OmniAnomaly/ServerMachineDataset')
        file_list = os.listdir(os.path.join(dataset_folder, "train"))

        train_files = []
        test_files = []
        label_files = []
        file_length = [0]
        for filename in file_list:
            if filename.endswith('.txt'):
                #调用名为load_as_np的函数来加载训练数据，并将返回的结果添加到train_files列表中。
                train_files.append(load_as_np('train', filename, filename.strip('.txt'), dataset_folder, output_folder))
                test_files.append(load_as_np('test', filename, filename.strip('.txt'), dataset_folder, output_folder))
                label_files.append(load_as_np('test_label', filename, filename.strip('.txt'), dataset_folder, output_folder))
                file_length.append(len(label_files[-1]))

        for i, train, test, label in zip(range(len(test_files)), train_files, test_files, label_files):
            #将训练数据train保存为NumPy数组文件。保存的文件名根据变量i的值动态生成，使用format函数将i插入到字符串中。
            #保存的文件路径是由输出文件夹路径（output_folder）、数据集名称（dataset）和动态生成的文件名组成的。
            np.save(os.path.join(output_folder, dataset + "{}_train.npy".format(i)), train)
            np.save(os.path.join(output_folder, dataset + "{}_test.npy".format(i)), test)
            np.save(os.path.join(output_folder, dataset + "{}_test_label.npy".format(i)), label)

        #这行代码使用np.concatenate函数将train_files列表中的所有训练数据按照axis = 0（沿着垂直方向）进行拼接。
        #拼接后的结果存储在train_files变量中
        train_files = np.concatenate(train_files, axis=0)
        test_files = np.concatenate(test_files, axis=0)
        label_files = np.concatenate(label_files, axis=0)
        np.save(os.path.join(output_folder, dataset + "_train.npy"), train_files)
        np.save(os.path.join(output_folder, dataset + "_test.npy"), test_files)
        np.save(os.path.join(output_folder, dataset + "_test_label.npy"), label_files)

        """
        首先使用np.cumsum函数将file_length列表转换为累加和的形式，并将结果转换为Python列表。

        然后，通过遍历累加和列表，创建通道划分列表channel_divisions。对于每个索引i，将file_length[i]和file_length[i+1]作为通道划分的起始和结束位置，并将其添加到channel_divisions列表中。

        最后，使用open函数创建一个文件对象，并使用json.dump函数将channel_divisions列表以JSON格式写入文件中。文件名根据数据集名称和'test_channel.json'动态生成。
        """
        file_length = np.cumsum(np.array(file_length)).tolist()
        channel_divisions = []
        for i in range(len(file_length)-1):
            channel_divisions.append([file_length[i], file_length[i+1]])
        with open(os.path.join(json_folder, dataset + "_" + 'test_channel.json'), 'w') as file:
            json.dump(channel_divisions, file)
                
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = os.path.join(base_dir, 'telemanom/data')
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        #使用sorted函数对res列表进行排序。排序的依据是一个lambda函数，该函数根据每个元素的第一个元素进行排序。
        #lambda函数中的表达式k[0][0] + '-{:2d}'.format(int(k[0][2:]))用于构建排序键。它将每个元素的第一个元素的第一个字符和后两个字符作为整数进行格式化，以确保排序按照日期的正确顺序进行。
        #排序后的结果存储在res变量中。
        res = sorted(res, key=lambda k: k[0][0]+'-{:2d}'.format(int(k[0][2:])))        
#         label_folder = os.path.join(dataset_folder, 'test_label')
#         if not os.path.exists(label_folder):
#             os.mkdir(label_folder)
#         makedirs(label_folder, exist_ok=True)
        """
        这行代码使用列表推导式根据特定条件筛选出res列表中的元素，并将满足条件的元素存储在data_info列表中。条件筛选的条件是：
            元素的第二个元素（row[1]）等于给定的dataset。
            元素的第一个元素（row[0]）不等于'P-2'。
        """
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
#         data_info = [row for row in res if row[1] == dataset]
        labels = []
        class_divisions = {}
        channel_divisions = []
        current_index = 0

    """
    这段代码是一个循环，对data_info列表中的每个元素进行操作。

首先，将元素的第三个元素（row[2]）解析为Python对象，保存在anomalies变量中。

然后，将元素的最后一个元素（row[-1]）转换为整数，保存在length变量中。

接下来，创建一个长度为length、元素类型为布尔值的零数组label。

然后，对anomalies列表中的每个元素，将对应的label片段设置为True。

将label数组的元素追加到labels列表中。

接着，从元素的第一个元素（row[0]）中获取一个字符，并将其保存在_class变量中。

如果_class已经存在于class_divisions字典的键中，则更新对应值的第二个元素；否则，在class_divisions中添加一个新的键值对，其中键为_class，值为[current_index, current_index+length]。

将一个包含当前索引和当前索引+length的列表追加到channel_divisions列表中。

最后，将current_index增加length。
    """
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
            
            _class = row[0][0]
            if _class in class_divisions.keys():
                class_divisions[_class][1] += length
            else:
                class_divisions[_class] = [current_index, current_index+length]
            channel_divisions.append([current_index, current_index+length])
            current_index += length
            
        labels = np.asarray(labels)
#         print(dataset, 'test_label', labels.shape)
#         with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
#             dump(labels, file)
        np.save(os.path.join(output_folder, dataset + "_" + 'test_label' + ".npy"), labels)
        
        with open(os.path.join(json_folder, dataset + "_" + 'test_class.json'), 'w') as file:
            json.dump(class_divisions, file)
        with open(os.path.join(json_folder, dataset + "_" + 'test_channel.json'), 'w') as file:
            json.dump(channel_divisions, file)

        """
        在函数内部，首先创建一个空列表data，用于存储拼接后的数据。

        然后，通过遍历data_info列表中的每个元素，获取文件名并使用np.load函数加载对应文件的数据，将其追加到data列表中。

        将data列表转换为NumPy数组，并使用MinMaxScaler对数据进行归一化处理。

        最后，使用np.save函数将归一化后的数据保存为.npy文件，文件名根据给定的数据集名称和类别（'train'或'test'）动态生成。

        最外层的for循环用于依次调用concatenate_and_save函数，分别对'train'和'test'进行数据拼接和保存。
        """
        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
#             print(dataset, category, data.shape)
#             with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
#                 dump(data, file)
            data = MinMaxScaler().fit_transform(data)
            np.save(os.path.join(output_folder, dataset + "_" + category + ".npy"), data)

        for c in ['train', 'test']:
            concatenate_and_save(c)
            
    elif dataset == 'SWaT':
        """
        这段代码用于读取SWaT数据集中的正常数据和异常数据，并对其进行预处理和保存。

首先，通过pd.read_excel函数读取SWaT数据集中的正常数据文件（SWaT_Dataset_Normal_v1.xlsx），并将其转换为NumPy数组。

然后，使用MinMaxScaler对正常数据进行归一化处理，并将数据限制在0到1的范围内。

最后，使用np.save函数将归一化后的正常数据保存为.npy文件，文件名根据给定的数据集名称和类别（'train'）动态生成。

接着，通过pd.read_excel函数读取SWaT数据集中的异常数据文件（SWaT_Dataset_Attack_v0.xlsx），并将其转换为NumPy数组。

使用.iloc方法选择数据的特定行和列，将异常标签（最后一列）转换为布尔类型，并将其转换为整数类型。

同样，使用MinMaxScaler对异常数据进行归一化处理，并将数据限制在0到1的范围内。

接下来，使用np.save函数分别将归一化后的异常数据和异常标签保存为.npy文件，文件名根据给定的数据集名称和类别（'test'）动态生成。
        """
        dataset_folder = os.path.join(base_dir, 'SWaT/Physical')
        normal_data = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v1.xlsx'))
        normal_data = normal_data.iloc[1:, 1:-1].to_numpy()
        normal_data = MinMaxScaler().fit_transform(normal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_train.npy"), normal_data)
        
        abnormal_data = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.xlsx'))
        abnormal_label = abnormal_data.iloc[1:, -1] == 'Attack'
        abnormal_label = abnormal_label.to_numpy().astype(int)
        
        abnormal_data = abnormal_data.iloc[1:, 1:-1].to_numpy()
        abnormal_data = MinMaxScaler().fit_transform(abnormal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_test.npy"), abnormal_data)
        np.save(os.path.join(output_folder, dataset + "_test_label.npy"), abnormal_label)
        
    elif dataset == 'WADI':
        normal_data = pd.read_csv(os.path.join(base_dir, 'WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv'))
        normal_data = normal_data.dropna(axis='columns', how='all').dropna()
        normal_data = normal_data.iloc[:, 3:].to_numpy()
        normal_data = MinMaxScaler().fit_transform(normal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_train.npy"), normal_data)
        
        abnormal_data = pd.read_csv(os.path.join(base_dir, 'WADI/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv'), header=1)
        abnormal_data = abnormal_data.dropna(axis='columns', how='all').dropna()
        abnormal_label = abnormal_data.iloc[:, -1] == -1
        abnormal_label = abnormal_label.to_numpy().astype(int)
        
        abnormal_data = abnormal_data.iloc[:, 3:-1].to_numpy()
        abnormal_data = MinMaxScaler().fit_transform(abnormal_data).clip(0, 1)
        np.save(os.path.join(output_folder, dataset + "_test.npy"), abnormal_data)
        np.save(os.path.join(output_folder, dataset + "_test_label.npy"), abnormal_label)
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #使用argparse库添加一个名为“dataset”的命令行参数。--dataset参数是必需的，类型为字符串（str）。
    #help参数用于提供关于该命令行参数的描述信息，指定了可选的数据集名称选项（SMD/SMAP/MSL/SWaT/WADI）。
    parser.add_argument("--dataset", required=True, type=str,
                        help="Name of dataset; SMD/SMAP/MSL/SWaT/WADI")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Directory of raw data")
    parser.add_argument("--out_dir", default=None, type=str,
                        help="Directory of the processed data")
    parser.add_argument("--json_dir", default=None, type=str,
                        help="Directory of the json files for the processed data")
    options = parser.parse_args()
    
    datasets = ['SMD', 'SMAP', 'MSL', 'SWaT', 'WADI']
    """
    检查用户传入的数据集名称是否在datasets列表中，如果存在，则进行数据加载和处理。

首先，将options.data_dir赋值给base_dir。

然后，检查options.out_dir是否为None。如果是，则将output_folder设置为base_dir下的'processed'文件夹路径；否则，将output_folder设置为options.out_dir的值。如果output_folder路径不存在，则使用os.mkdir函数创建该文件夹。

接下来，检查options.json_dir是否为None。如果是，则将json_folder设置为base_dir下的'json'文件夹路径；否则，将json_folder设置为options.json_dir的值。如果json_folder路径不存在，则使用os.mkdir函数创建该文件夹。

最后，调用load_data函数，传入options.dataset、base_dir、output_folder和json_folder作为参数，进行数据加载和处理。
    """
    if options.dataset in datasets:
        base_dir = options.data_dir
        
        if options.out_dir == None:
            output_folder = os.path.join(base_dir, 'processed')
        else:
            output_folder = options.out_dir
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        if options.json_dir == None:
            json_folder = os.path.join(base_dir, 'json')
        else:
            json_folder = options.json_dir
        if not os.path.exists(json_folder):
            os.mkdir(json_folder)
            
        load_data(options.dataset, base_dir, output_folder, json_folder)
    
#     commands = sys.argv[1:]
#     load = []
#     if len(commands) > 0:
#         for d in commands:
#             if d in datasets:
#                 load_data(d)
#     else:
#         print("""
#         Usage: python data_preprocess.py <datasets>
#         where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
#         """)