import json, os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import utils.config as config



# Exponential weighted moving average
"""
定义了一个名为ewma的函数，用于计算指数加权移动平均值（Exponential Weighted Moving Average，EWMA）。

该函数接受一个名为series的参数，表示要计算EWMA的时间序列数据。

weighting_factor参数可选，用于指定加权因子的值，默认为0.9。加权因子的取值范围为0到1之间，越接近1表示对历史数据的权重越大。

函数内部首先计算当前因子current_factor，即1减去加权因子。

然后，将series复制给_ewma变量，以保持原始数据不变。

接下来，通过循环遍历_ewma序列，从第二个元素开始，根据EWMA公式更新每个元素的值。更新公式为前一个元素乘以加权因子，加上当前元素乘以当前因子。

最后，返回更新后的_ewma序列作为计算结果。
"""
def ewma(series, weighting_factor=0.9):
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i-1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


# Get anomaly sequences.
"""
这段代码定义了一个名为anomaly_sequence的函数，用于提取时间序列中的异常序列。

该函数接受一个名为label的参数，表示时间序列的异常标签。

首先，使用np.argwhere函数找到异常点的索引，并通过flatten方法将其展平为一维数组，赋值给anomaly_args。

然后，计算异常间隔的项，即相邻异常点之间的距离。使用逻辑运算符和比较运算符进行判断，生成布尔数组，并赋值给terms。

接下来，通过np.argwhere函数找到terms中为True的索引，并加1，生成异常序列的起始位置的索引。将其展平为一维数组，并赋值给sequence_args。

计算异常序列的长度，即相邻异常序列起始位置之间的距离，赋值给sequence_length。

将0插入到sequence_args的开头，确保异常序列的第一个起始位置索引为0。如果sequence_args的长度大于1，则将sequence_args的第二个元素插入到sequence_length的开头。

将异常序列的最后一个起始位置索引到异常序列的末尾的距离添加到sequence_length中。

获取异常序列的索引，即根据sequence_args从anomaly_args中获取对应的异常点的索引。

最后，将异常序列的起始位置索引和终止位置索引组合成一个二维数组，并转置为使其行对应于异常序列。返回异常序列的索引数组anomaly_label_seq和异常序列的长度列表sequence_length。
"""
def anomaly_sequence(label):
    anomaly_args = np.argwhere(label).flatten()  # Indices for abnormal points.
    
    # Terms between abnormal invervals
    terms = anomaly_args[1:] - anomaly_args[:-1]
    terms = terms > 1

    # Extract anomaly sequences.
    sequence_args = np.argwhere(terms).flatten() + 1
    sequence_length = list(sequence_args[1:] - sequence_args[:-1])
    sequence_args = list(sequence_args)

    sequence_args.insert(0, 0)
    if len(sequence_args) > 1:
        sequence_length.insert(0, sequence_args[1])
    sequence_length.append(len(anomaly_args) - sequence_args[-1])

    # Get anomaly sequence arguments.
    sequence_args = anomaly_args[sequence_args]
    anomaly_label_seq = np.transpose(np.array((sequence_args, sequence_args + np.array(sequence_length))))
    return anomaly_label_seq, sequence_length


# Interval-dependent point
def interval_dependent_point(sequences, lengths):
    n_intervals = len(sequences)
    n_steps = np.sum(lengths)
    return (n_steps / n_intervals) / lengths


"""
这段代码定义了一个名为f1_score的函数，用于计算F1分数。

该函数接受三个参数：gt表示真实标签，pr表示预测标签，anomaly_rate表示异常率。

首先，将gt序列进行扩展，加入两个零元素并转换为整数类型，赋值给gt_aug。

然后，通过计算gt_aug相邻元素的差值，获取异常的起始位置和结束位置。

将起始位置和结束位置组合成一个二维数组intervals。

将pr复制给pa，并计算pr的1-anomaly_rate分位数，赋值给q。

将pa中大于q的元素转换为1，小于等于q的元素转换为0，得到二值化后的预测标签pa。

如果modify参数为True，则进行修改后的F1计算。

首先，调用anomaly_sequence函数计算gt的异常序列的索引和长度，分别赋值给gt_seq_args和gt_seq_lens。

然后，调用interval_dependent_point函数计算与异常序列间隔相关的点，赋值给ind_p。

接下来，初始化TP和FN为0，并通过循环遍历gt_seq_args、gt_seq_lens和ind_p来计算TP和FN。

TP表示预测为正例且与异常序列重叠的样本数量，通过pa中异常序列的起始位置和结束位置来获取。FN表示预测为负例且与异常序列重叠的样本数量，通过异常序列的长度减去n_tp来获取。

计算TN和FP的方式与之前相同。

如果modify参数为False，则进行传统的F1计算。

如果adjust参数为True，则进行点调整操作。对于每个异常间隔，如果在pa中存在任何一个异常点，则将该间隔内的所有元素都设为1。

计算TP、TN、FP和FN的方式与之前相同。

最后，根据TP、FP和FN计算精确度precision、召回率recall和F1分数f1_score，并返回这三个值。
"""
def f1_score(gt, pr, anomaly_rate=0.05, adjust=True, modify=False):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(np.int32)
    
    # Modified F1
    if modify:
        gt_seq_args, gt_seq_lens = anomaly_sequence(gt)  # gt anomaly sequence args
        ind_p = interval_dependent_point(gt_seq_args, gt_seq_lens)  # interval-dependent point
        
        # Compute TP and FN.
        TP = 0
        FN = 0
        for _seq, _len, _p in zip(gt_seq_args, gt_seq_lens, ind_p):
            n_tp = pa[_seq[0]:_seq[1]].sum()
            n_fn = _len - n_tp
            TP += n_tp * _p
            FN += n_fn * _p
            
        # Compute TN and FP.
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()

    else:
        # point adjustment
        if adjust:
            for s, e in intervals:
                interval = slice(s, e)
                if pa[interval].sum() > 0:
                    pa[interval] = 1

        # confusion matrix
        TP = (gt * pa).sum()
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()
        FN = (gt * (1 - pa)).sum()

        assert (TP + TN + FP + FN) == len(gt)

    # Compute p, r, f1.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score



# Compute evaluation metrics.
def compute(options):
    # Load test data, estimation results, and label. 加载数据
    test_data = np.load(config.TEST_DATASET[options.dataset])
    test_label = np.load(config.TEST_LABEL[options.dataset]).copy().astype(np.int32)
    data_dim = len(test_data[0])

    if options.data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][options.data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
        
    output_values = np.load(options.result)
    if output_values.ndim == 2:
        output_values = output_values[:, 0]
    #调用函数计算ewma
    if options.smooth_scores:
        smoothed_values = ewma(output_values, options.smoothing_weight)
        
    # Result text file
    if options.outfile == None:
        prefix = options.result[:-4]
        result_file = prefix + '_evaluations.txt'
    else:
        prefix = options.outfile[:-4]
        result_file = options.outfile
    result_file = open(result_file, 'w')
        
    # Save test data and output results in figures.
    # 存储数据 画图
    if options.save_figures:
        save_folder = prefix + '_figures/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        for i, index in enumerate(divisions):
            label = test_label[index[0]:index[1]]
            
            fig, axs = plt.subplots(data_dim, 1, figsize=(20, data_dim))
            for j in range(data_dim):
                axs[j].plot(test_data[index[0]:index[1], j], alpha=0.6)
                axs[j].scatter(np.arange(index[1]-index[0])[label], test_data[index[0]:index[1]][label, j],
                                  c='r', s=1, alpha=0.8)
            fig.savefig(save_folder+'data_division_{}.jpg'.format(i), bbox_inches='tight')
            plt.close()
            
            fig, axs = plt.subplots(1, figsize=(20, 5))
            axs.plot(output_values[index[0]:index[1]], alpha=0.6)
            axs.scatter(np.arange(index[1]-index[0])[label], output_values[index[0]:index[1]][label],
                        c='r', s=1, alpha=0.8)
            fig.savefig(save_folder+'score_division_{}.jpg'.format(i), bbox_inches='tight')
            plt.close()
            
            if options.smooth_scores:
                fig, axs = plt.subplots(1, figsize=(20, 5))
                axs.plot(smoothed_values[index[0]:index[1]], alpha=0.6)
                axs.scatter(np.arange(index[1]-index[0])[label], smoothed_values[index[0]:index[1]][label],
                            c='r', s=1, alpha=0.8)
                fig.savefig(save_folder+'smoothed_score_division_{}.jpg'.format(i), bbox_inches='tight')
                plt.close()
        
    # Compute F1-scores.
    f1_str = 'Modified F1-score' if options.modified_f1 else 'F1-score'
    # F1 Without PA
    # 重复代码 意思就是计算然后存储
    result_file.write('<'+f1_str+' without point adjustment>\n\n')
    
    if options.data_division == 'total':
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = f1_score(test_label, output_values, rate, False, options.modified_f1)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest F1-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
        print('Best F1-score without point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
        
    else:
        average_eval = np.zeros(3)
        for division in divisions:
            _test_label = test_label[division[0]:division[1]]
            _output_values = output_values[division[0]:division[1]]
            best_eval = (0, 0, 0)
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = f1_score(_test_label, _output_values, rate, False, options.modified_f1)
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
            average_eval += np.array(best_eval)
        average_eval /= len(divisions)
        result_file.write('\nBest F1-score\n')
        result_file.write(f'precision: {average_eval[0]:.5f} | recall: {average_eval[1]:.5f} | F1-score: {average_eval[2]:.5f}\n\n\n')
        print('Best F1-score without point adjustment')
        print(f'precision: {average_eval[0]:.5f} | recall: {average_eval[1]:.5f} | F1-score: {average_eval[2]:.5f}\n')
    
    # F1 With PA
    if not options.modified_f1:
        result_file.write('<F1-score with point adjustment>\n\n')
        
        if options.data_division == 'total':
            best_eval = (0, 0, 0)
            best_rate = 0
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = f1_score(test_label, output_values, rate, True)
                result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
                    best_rate = rate
            result_file.write('\nBest F1-score\n')
            result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
            print('Best F1-score with point adjustment')
            print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
            
        else:
            average_eval = np.zeros(3)
            for division in divisions:
                _test_label = test_label[division[0]:division[1]]
                _output_values = output_values[division[0]:division[1]]
                best_eval = (0, 0, 0)
                for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                    evaluation = f1_score(_test_label, _output_values, rate, True)
                    if evaluation[2] > best_eval[2]:
                        best_eval = evaluation
                average_eval += np.array(best_eval)
            average_eval /= len(divisions)
            result_file.write('\nBest F1-score\n')
            result_file.write(f'precision: {average_eval[0]:.5f} | recall: {average_eval[1]:.5f} | F1-score: {average_eval[2]:.5f}\n\n\n')
            print('Best F1-score with point adjustment')
            print(f'precision: {average_eval[0]:.5f} | recall: {average_eval[1]:.5f} | F1-score: {average_eval[2]:.5f}\n')
    
    if options.smooth_scores:
        # F1 Without PA
        result_file.write('<'+f1_str+' of smoothed scores without point adjustment>\n\n')
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = f1_score(test_label, smoothed_values, rate, False, options.modified_f1)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest F1-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
        print('Best F1-score of smoothed scores without point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
        
        # F1 With PA
        if not options.modified_f1:
            result_file.write('<F1-score of smoothed scores with point adjustment>\n\n')
            best_eval = (0, 0, 0)
            best_rate = 0
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = f1_score(test_label, smoothed_values, rate, True)
                result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
                    best_rate = rate
            result_file.write('\nBest F1-score\n')
            result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
            print('Best F1-score of smoothed scores with point adjustment')
            print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
    
    # Close file.
    result_file.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    parser.add_argument("--result", required=True, type=str, help='result file (.npy) obtained from estimate.py')
    parser.add_argument("--outfile", default=None, type=str, help='output file name (.txt) to save computation logs')
    
    parser.add_argument('--smooth_scores', default=False, action='store_true', help='option for smoothing scores (ewma)')
    parser.add_argument("--smoothing_weight", default=0.9, type=float, help='ewma weight when smoothing socres')
    parser.add_argument('--modified_f1', default=False, action='store_true', help='modified f1 scores (not used now)')
    
    parser.add_argument('--save_figures', default=False, action='store_true', help='save figures of data and anomaly scores')
    parser.add_argument("--data_division", default='total', type=str, help='data division info when saving figures; channel/class/total')
    
    parser.add_argument("--min_anomaly_rate", default=0.001, type=float, help='minimum threshold rate')
    parser.add_argument("--max_anomaly_rate", default=0.3, type=float, help='maximum threshold rate')
    
    options = parser.parse_args()
    compute(options)