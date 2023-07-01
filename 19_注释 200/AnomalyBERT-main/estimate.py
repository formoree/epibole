import os, json
import numpy as np
import torch
import argparse

import utils.config as config



# Estimate anomaly scores.
"""
定义了一个名为estimate的函数，用于估计模型在测试数据上的输出值。

函数接受多个参数，包括test_data表示测试数据，model表示模型，post_activation表示激活函数，out_dim表示输出维度，batch_size表示批大小，window_sliding表示窗口滑动大小，divisions表示数据划分，check_count表示检查计数，device表示设备。

首先，根据模型的最大序列长度和补丁大小计算窗口大小，确保其可以被窗口滑动大小整除。

然后，初始化输出值矩阵output_values为全零矩阵，大小为测试数据长度乘以输出维度。

接下来，通过循环遍历每个数据划分division来进行估计。

对于每个划分，首先计算数据长度和最后一个窗口的索引范围。

然后，从测试数据中获取当前划分的数据_test_data，并初始化输出值矩阵_output_values和重叠计数矩阵n_overlap为全零矩阵，大小分别为数据长度和数据长度。

接下来，使用torch.no_grad()上下文管理器，禁用梯度计算。

然后，通过循环遍历每个窗口的起始位置first来进行估计。

在内部循环中，通过索引i遍历窗口中的每个时间步，并根据起始位置和批大小获取当前批次的输入数据x。

接下来，对输入数据x进行模型的前向传播，并经过激活函数得到输出数据y。

将输出数据y添加到_output_values中对应的位置，并将重叠计数n_overlap相应位置加1。

计数变量count增加批大小n_batch。

如果count超过了checked_index，则打印当前计算的窗口数量。

接着，更新_first的值。

然后，通过将range(_first, last_window, _batch_sliding)和range(_first+_batch_sliding, last_window, _batch_sliding) + [last_window]进行zip操作，获取批次的起始位置和结束位置。

在内部循环中，通过索引i和j遍历批次中的每个时间步，并根据起始位置和窗口滑动大小获取当前窗口的输入数据x。

将输入数据x进行堆叠，得到形状为(n_batch, window_size, -1)的张量x，并将其转移到设备上。

接下来，对输入数据x进行模型的前向传播，并经过激活函数得到输出数据y。

将输出数据y添加到_output_values中对应的位置，并将重叠计数n_overlap相应位置加1。

计数变量count增加批大小n_batch。

如果count超过了checked_index，则打印当前计算的窗口数量。

然后，将_output_values除以n_overlap的每一行，并将结果赋值给_output_values，以计算平均值。

最后，将_output_values的值复制给output_values对应的划分位置。

最终，返回output_values作为模型在测试数据上的输出值。
"""
def estimate(test_data, model, post_activation, out_dim, batch_size, window_sliding, divisions,
             check_count=None, device='cpu'):
    # Estimation settings
    window_size = model.max_seq_len * model.patch_size
    assert window_size % window_sliding == 0
    
    n_column = out_dim
    n_batch = batch_size
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding

    output_values = torch.zeros(len(test_data), n_column, device=device)
    count = 0
    checked_index = np.inf if check_count == None else check_count
    
    # Record output values.
    for division in divisions:
        data_len = division[1] - division[0]
        last_window = data_len - window_size + 1
        _test_data = test_data[division[0]:division[1]]
        _output_values = torch.zeros(data_len, n_column, device=device)
        n_overlap = torch.zeros(data_len, device=device)
    
        with torch.no_grad():
            _first = -batch_sliding
            for first in range(0, last_window-batch_sliding+1, batch_sliding):
                for i in range(first, first+window_size, window_sliding):
                    # Call mini-batch data.
                    x = torch.Tensor(_test_data[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)
                    
                    # Evaludate and record errors.
                    y = post_activation(model(x))
                    _output_values[i:i+batch_sliding] += y.view(-1, n_column)
                    n_overlap[i:i+batch_sliding] += 1

                    count += n_batch

                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += check_count

                _first = first

            _first += batch_sliding

            for first, last in zip(range(_first, last_window, _batch_sliding),
                                   list(range(_first+_batch_sliding, last_window, _batch_sliding)) + [last_window]):
                # Call mini-batch data.
                x = []
                for i in list(range(first, last-1, window_sliding)) + [last-1]:
                    x.append(torch.Tensor(_test_data[i:i+window_size].copy()))

                # Reconstruct data.
                x = torch.stack(x).to(device)

                # Evaludate and record errors.
                y = post_activation(model(x))
                for i, j in enumerate(list(range(first, last-1, window_sliding)) + [last-1]):
                    _output_values[j:j+window_size] += y[i]
                    n_overlap[j:j+window_size] += 1

                count += n_batch

                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += check_count

            # Compute mean values.
            _output_values = _output_values / n_overlap.unsqueeze(-1)
            
            # Record values for the division.
            output_values[division[0]:division[1]] = _output_values
            
    return output_values

"""
主函数，用于加载测试数据、加载模型、进行数据划分，并调用estimate函数来估计模型在测试数据上的输出值，最后将结果保存到文件中。

首先，从config.TEST_DATASET中加载测试数据，并将其转换为浮点型。

然后，如果options.dataset在config.IGNORED_COLUMNS中有对应的忽略列，将测试数据中的这些列移除。

接下来，根据options.gpu_id设置设备。

然后，加载模型，并根据options.state_dict加载模型的状态字典（如果存在）。

接着，将模型设置为评估模式。

接下来，根据options.data_division确定数据划分方式。

如果data_division为'total'，则将整个测试数据作为一个划分。

否则，根据config.DATA_DIVISION中对应数据集和划分方式的文件路径加载划分信息，并将其转换为列表形式。

如果划分信息是一个字典，则取其所有值作为划分。

然后，根据options.reconstruction_output确定输出维度。

如果reconstruction_output为True，则输出维度为测试数据的列数。

否则，输出维度为1。

接下来，根据options.reconstruction_output确定激活函数。

如果reconstruction_output为True，则使用torch.nn.Identity作为激活函数。

否则，使用torch.nn.Sigmoid作为激活函数。

然后，调用estimate函数来估计模型在测试数据上的输出值，并将结果赋值给output_values。

接着，将output_values转移到CPU上，并将其保存到文件中。

最后，返回结果。
"""
def main(options):
    # Load test data.
    test_data = np.load(config.TEST_DATASET[options.dataset]).copy().astype(np.float32)
    
    # Ignore the specific columns.
    if options.dataset in config.IGNORED_COLUMNS.keys():
        ignored_column = np.array(config.IGNORED_COLUMNS[options.dataset])
        remaining_column = [col for col in range(len(test_data[0])) if col not in ignored_column]
        test_data = test_data[:, remaining_column]
    
    # Load model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = torch.load(options.model, map_location=device)
    if options.state_dict != None:
        model.load_state_dict(torch.load(options.state_dict, map_location='cpu'))
    model.eval()
    
    # Data division
    data_division = config.DEFAULT_DIVISION[options.dataset] if options.data_division == None else options.data_division 
    if data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
            
    n_column = len(test_data[0]) if options.reconstruction_output else 1
    post_activation = torch.nn.Identity().to(device) if options.reconstruction_output\
                      else torch.nn.Sigmoid().to(device)
            
    # Estimate scores.
    output_values = estimate(test_data, model, post_activation, n_column, options.batch_size,
                             options.window_sliding, divisions, options.check_count, device)
    
    # Save results.
    output_values = output_values.cpu().numpy()
    outfile = options.state_dict[:-3] + '_results.npy' if options.outfile == None else options.outfile
    np.save(outfile, output_values)
    
    
    
if __name__ == "__main__":
    #给定参数大小 运行main
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    
    parser.add_argument("--model", required=True, type=str, help='model file (.pt) to estimate')
    parser.add_argument("--state_dict", default=None, type=str, help='state dict file (.pt) to estimate')
    parser.add_argument("--outfile", default=None, type=str, help='output file name (.npy) to save anomaly scores')
    
    parser.add_argument("--data_division", default=None, type=str, help='data division; None(defualt)/channel/class/total')
    parser.add_argument("--check_count", default=5000, type=int, help='check count of window computing')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_sliding", default=16, type=int, help='sliding steps of windows; window size should be divisible by this value')
    parser.add_argument('--reconstruction_output', default=False, action='store_true', help='option for reconstruction model (deprecated)')
    
    options = parser.parse_args()
    main(options)
