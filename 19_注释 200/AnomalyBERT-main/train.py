import os, time, json
import numpy as np
import torch
import torch.nn as nn
import argparse

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from models.anomaly_transformer import get_anomaly_transformer

from estimate import estimate
from compute_metrics import f1_score



def main(options):
    # Load data.
    """
    加载训练数据、替换数据、测试数据和测试标签，并获取数据的维度以及数值和分类列的数量。

首先，从config.TRAIN_DATASET中加载训练数据，并将其转换为浮点型。

然后，根据options.replacing_data确定是否使用替换数据。

如果options.replacing_data为None，则将train_data赋值给replacing_data。

否则，从config.TRAIN_DATASET中加载替换数据，并将其转换为浮点型。

接着，从config.TEST_DATASET中加载测试数据，并将其转换为浮点型。

然后，从config.TEST_LABEL中加载测试标签，并将其转换为整型。

接下来，获取训练数据的维度d_data。

然后，从config.NUMERICAL_COLUMNS中获取数值列的索引，并将其转换为数组。

接着，获取数值列的数量num_numerical。

接下来，从config.CATEGORICAL_COLUMNS中获取分类列的索引，并将其转换为数组。

最后，获取分类列的数量num_categorical。
    """
    train_data = np.load(config.TRAIN_DATASET[options.dataset]).copy().astype(np.float32)
    replacing_data = train_data if options.replacing_data == None\
                     else np.load(config.TRAIN_DATASET[options.replacing_data]).copy().astype(np.float32)
    test_data = np.load(config.TEST_DATASET[options.dataset]).copy().astype(np.float32)
    test_label = np.load(config.TEST_LABEL[options.dataset]).copy().astype(np.int32)
    
    d_data = len(train_data[0])
    numerical_column = np.array(config.NUMERICAL_COLUMNS[options.dataset])
    num_numerical = len(numerical_column)
    categorical_column = np.array(config.CATEGORICAL_COLUMNS[options.dataset])
    num_categorical = len(categorical_column)
    
    # Ignore the specific columns.
    """
    在存在被忽略列的情况下，将训练数据、替换数据和测试数据中的相应列移除，并更新数值和分类列的索引。

首先，判断options.dataset是否在config.IGNORED_COLUMNS的键中。

如果是，则获取对应数据集的被忽略列，并将其转换为数组。

然后，根据被忽略列的索引，生成剩余列的索引。

接着，将训练数据中的剩余列提取出来。

同时，如果options.replacing_data不为空，则将替换数据中的剩余列提取出来。

然后，将测试数据中的剩余列提取出来。

接下来，更新d_data为剩余列的数量。

然后，根据被忽略列的索引，更新数值列的索引。

同时，将数值列索引中大于被忽略列索引的值减去1。

接着，根据被忽略列的索引，更新分类列的索引。

同时，将分类列索引中大于被忽略列索引的值减去1。
    """
    if options.dataset in config.IGNORED_COLUMNS.keys():
        ignored_column = np.array(config.IGNORED_COLUMNS[options.dataset])
        remaining_column = [col for col in range(d_data) if col not in ignored_column]
        train_data = train_data[:, remaining_column]
        replacing_data = train_data if options.replacing_data == None else replacing_data[:, remaining_column]
        test_data = test_data[:, remaining_column]
        
        d_data = len(remaining_column)
        numerical_column -= (numerical_column[:, None] - ignored_column[None, :] > 0).astype(int).sum(axis=1)
        categorical_column -= (categorical_column[:, None] - ignored_column[None, :] > 0).astype(int).sum(axis=1)
        
    # Data division
    # 进行数据分割
    data_division = config.DEFAULT_DIVISION[options.dataset] if options.data_division == None else options.data_division 
    if data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
    
    n_features = options.n_features
    data_seq_len = n_features * options.patch_size

    # Define model.
    # 定义模型（包括现有参数） 使用gpu
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = get_anomaly_transformer(input_d_data=d_data,
                                    output_d_data=1 if options.loss=='bce' else d_data,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=n_features,
                                    positional_encoding=None,
                                    relative_position_embedding=True,
                                    transformer_n_layer=options.n_layer,
                                    transformer_n_head=8,
                                    dropout=options.dropout).to(device)
    
    # Load a checkpoint if exists.
    if options.checkpoint != None:
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'))

    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    log_dir = os.path.join(config.LOG_DIR, time.strftime('%y%m%d%H%M%S_'+options.dataset, time.localtime(time.time())))
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))
    
    # hyperparameters save
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)
    
    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    # Train model.
    max_iters = options.max_steps + 1
    n_batch = options.batch_size
    valid_index_list = np.arange(len(train_data) - data_seq_len)
#     anomaly_weight = options.partial_loss / options.total_loss

    # Train loss 训练模型 选择loss函数
    lr = options.lr
    if options.loss == 'l1':
        train_loss = nn.L1Loss().to(device)
        rec_loss = nn.MSELoss().to(device)
    elif options.loss == 'mse':
        train_loss = nn.MSELoss().to(device)
        rec_loss = nn.MSELoss().to(device)
    elif options.loss == 'bce':
        train_loss = nn.BCELoss().to(device)
#         train_loss = lambda pred, gt: -(gt * torch.log(pred + 1e-8) + (1 - gt.bool().float()) * torch.log(1 - pred + 1e-8)).mean()
        sigmoid = nn.Sigmoid().to(device)
    
    
    # Similarity map and constrastive loss
    """
    计算相似度矩阵的函数。给定特征矩阵，它将计算特征之间的相似度，并返回相似度矩阵。

首先，使用torch.matmul函数计算特征矩阵与其转置的乘积，得到相似度矩阵。

然后，使用torch.norm函数计算特征矩阵中每个向量的范数。

接着，使用torch.matmul函数计算范数的乘积，使用unsqueeze函数在适当的位置添加维度，然后加上一个很小的常数（1e-8）以避免除以零。

最后，将相似度矩阵除以范数的乘积，得到归一化的相似度矩阵，并将其返回
    """
    def similarity_map(features):
        similarity = torch.matmul(features, features.transpose(-1, -2))
        norms = torch.norm(features, dim=-1)
        denom = torch.matmul(norms.unsqueeze(-1), norms.unsqueeze(-2)) + 1e-8
        return similarity / denom
    #创建一个对角线掩码（diagonal mask）。对角线掩码是一个布尔型张量，用于在计算相似度矩阵时排除特征与自身的相似度
    diag_mask = torch.eye(n_features, device=device).bool().unsqueeze(0)

    """
    对比损失函数（contrastive loss）的实现。给定特征矩阵和异常标签，它将计算对比损失并返回。

首先，调用之前定义的similarity_map函数计算特征矩阵的相似度矩阵。

然后，使用torch.log函数计算相似度矩阵的指数，并使用masked_fill方法将对角线掩码应用到相似度矩阵上，将对角线上的值设为0，以排除特征与自身的相似度。然后，对相似度矩阵沿着最后一个维度求和。

接着，将相似度矩阵应用对角线掩码，将对角线上的值设为0。

然后，根据异常标签创建异常掩码（anomaly）和正常掩码（normal）。

接下来，使用异常标签的sum方法计算每个样本中异常标签的总数，并使用expand_as方法将其扩展为与异常标签形状相同的张量。

然后，计算正样本项（positive_term）。

首先，将相似度矩阵应用异常掩码，将异常样本的相似度值设为0。

然后，对相似度矩阵进行转置，并选择正常样本的相似度。

接着，沿着最后一个维度求平均值，并减去之前计算的相似度和。

最后，将正样本项除以（异常样本数 - 特征数）的值。

接下来，计算负样本项（negative_term），即异常样本的相似度和。

最后，将正样本项和负样本项的均值相加，并返回
    """
    def contrastive_loss(features, anomaly_label):
        similarity = similarity_map(features)
        similarity_sum = torch.log(torch.exp(similarity).masked_fill(diag_mask, 0).sum(dim=-1))
        similarity.masked_fill_(diag_mask, 0)

        anomaly = anomaly_label.bool()
        normal = anomaly == False
        n_anomaly = anomaly_label.sum(dim=-1, keepdim=True).expand_as(anomaly_label)

        positive_term = similarity
        positive_term[anomaly] = 0
        positive_term = positive_term.transpose(-1, -2)[normal].mean(dim=-1) - similarity_sum[normal]
        positive_term /= (n_anomaly - n_features)[normal]

        negative_term = similarity_sum[anomaly]

        return positive_term.mean() + negative_term.mean()

    """
    生成替换权重数组（replacing weights）的函数。给定一个区间长度（interval_len），它将返回一个与此区间长度相匹配的替换权重数组。

首先，根据区间长度的十分之一计算热身长度（warmup_len）。

然后，使用np.linspace函数在0和options.replacing_weight之间生成一个长度为热身长度的线性空间（包括0和options.replacing_weight）。

接着，使用np.full函数生成一个长度为(interval_len-2*warmup_len)的数组，每个元素都是options.replacing_weight。

最后，使用np.linspace函数在options.replacing_weight和0之间生成一个长度为热身长度的线性空间（包括options.replacing_weight和0）。

将这三个数组沿着None轴（即扁平化）进行连接，并返回结果。

这个函数用于生成一个替换权重数组，可用于在某个区间内逐渐增加然后逐渐减少替换的权重。
    """
    def replacing_weights(interval_len):
        warmup_len = interval_len // 10
        return np.concatenate((np.linspace(0, options.replacing_weight, num=warmup_len),
                               np.full(interval_len-2*warmup_len, options.replacing_weight),
                               np.linspace(options.replacing_weight, 0, num=warmup_len)), axis=None)
    
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    #余弦退火调度器（CosineLRScheduler）来调整学习率
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_iters,
                                  lr_min=lr*0.01,
                                  warmup_lr_init=lr*0.001,
                                  warmup_t=max_iters // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False,
                                 )

    
    # Replaced data length table
    replacing_rate = (options.replacing_rate_max/10, options.replacing_rate_max)
    replacing_len_max = int(options.replacing_rate_max * data_seq_len)
    replacing_len_half_max = replacing_len_max // 2
    
    replacing_table = list(np.random.randint(int(data_seq_len*replacing_rate[0]), int(data_seq_len*replacing_rate[1]), size=10000))
    replacing_table_index = 0
    replacing_table_length = 10000
    
    # Synthesis probability
    """
    这计算了一些概率值，用于一些数据处理的操作。

首先，计算了软替换的概率（soft_replacing_prob），它等于1减去options.soft_replacing。

接着，计算了均匀替换的概率（uniform_replacing_prob），它等于软替换的概率（soft_replacing_prob）减去options.uniform_replacing。

然后，计算了峰值噪声的概率（peak_noising_prob），它等于均匀替换的概率（uniform_replacing_prob）减去options.peak_noising。

接下来，根据options.loss的值来计算了长度调整的概率（length_adjusting_prob）。如果options.loss等于'bce'，则长度调整的概率等于峰值噪声的概率（peak_noising_prob）减去options.length_adjusting；否则，长度调整的概率等于峰值噪声的概率（peak_noising_prob）。

最后，白噪声的概率（white_noising_prob）等于options.white_noising。

这些概率值可以用于控制一些数据处理的操作，具体的操作可以根据实际需求进行进一步的处理
    """
    soft_replacing_prob = 1 - options.soft_replacing
    uniform_replacing_prob = soft_replacing_prob - options.uniform_replacing
    peak_noising_prob = uniform_replacing_prob - options.peak_noising
    length_adjusting_prob = peak_noising_prob - options.length_adjusting if options.loss == 'bce' else peak_noising_prob
    white_noising_prob = options.white_noising
    
    # Soft replacing flip options
    if options.flip_replacing_interval == 'all':
        vertical_flip = True
        horizontal_flip = True
    elif options.flip_replacing_interval == 'vertical':
        vertical_flip = True
        horizontal_flip = False
    elif options.flip_replacing_interval == 'horizontal':
        vertical_flip = False
        horizontal_flip = True
    elif options.flip_replacing_interval == 'none':
        vertical_flip = False
        horizontal_flip = False

    
    # Start training.
    for i in range(options.initial_iter, max_iters):
        """
        首先，使用np.random.choice函数从valid_index_list中随机选择n_batch个索引，生成一个大小为n_batch的一维数组，保存在变量first_index中。

接着，创建一个空列表x，用于存储训练数据。

然后，使用循环遍历first_index中的每个索引，并将对应的训练数据（从索引j开始，长度为data_seq_len）复制为一个新的张量，并将其转移到设备（device）上，然后将该张量添加到列表x中。

最后，使用torch.stack函数将列表x中的张量沿着新的维度进行堆叠，并将结果保存在变量x_true中。这样，x_true就是一个形状为(n_batch, data_seq_len, ...)的张量，其中...表示训练数据的其他维度。

这段代码用于生成一个训练数据批次，其中每个样本的长度为data_seq_len，并将其转移到设备上进行后续的训练操作。
        """
        first_index = np.random.choice(valid_index_list, size=n_batch)
        x = []
        for j in first_index:
            x.append(torch.Tensor(train_data[j:j+data_seq_len].copy()).to(device))
        x_true = torch.stack(x).to(device)

        # Replace data.
        """
        首先，将replacing_table_index的值赋给current_index。然后，将replacing_table_index增加n_batch。

接下来，创建一个空列表replacing_lengths，用于存储替换数据的长度。

然后，检查replacing_table_index是否超过replacing_table_length。如果超过了，说明需要从replacing_table的末尾取一部分长度，再从replacing_table的开头取剩余的长度，以保证总长度为n_batch。将这些长度添加到replacing_lengths中，并将replacing_table_index减去replacing_table_length。

如果replacing_table_index没有超过replacing_table_length，说明可以直接从replacing_table中取出一段长度为n_batch的数据，将这些长度添加到replacing_lengths中。

如果replacing_table_index等于replacing_table_length，将replacing_table_index重置为0。

将replacing_lengths转换为一个NumPy数组。

接下来，使用np.random.randint函数生成一个形状为(n_batch, d_data)的数组replacing_index，其中每个元素的值在0和(len(replacing_data)-replacing_lengths+1)之间。

最后，使用np.random.randint函数生成一个数值在0和data_seq_len-replacing_lengths+1之间的整数target_index。

这段代码用于生成替换数据和目标数据的索引，可用于进一步处理数据和进行训练
        """
        current_index = replacing_table_index
        replacing_table_index += n_batch

        replacing_lengths = []
        if replacing_table_index > replacing_table_length:
            replacing_lengths = replacing_table[current_index:]
            replacing_table_index -= replacing_table_length
            replacing_lengths = replacing_lengths + replacing_table[:replacing_table_index]
        else:
            replacing_lengths = replacing_table[current_index:replacing_table_index]
            if replacing_table_index == replacing_table_length:
                replacing_table_index = 0

        replacing_lengths = np.array(replacing_lengths)
        replacing_index = np.random.randint(0, (len(replacing_data)-replacing_lengths+1)[:, np.newaxis],
                                            size=(n_batch, d_data))
        target_index = np.random.randint(0, data_seq_len-replacing_lengths+1)

        # Replacing types and dimensions
        """
        这段代码用于生成替换数据的类型和异常点。

首先，使用np.random.uniform函数生成一个形状为(n_batch,)的数组replacing_type，其中每个元素的值在0和1之间。

接着，使用np.random.uniform函数分别生成形状为(n_batch, num_numerical)和(n_batch, num_categorical)的数组replacing_dim_numerical和replacing_dim_categorical，其中每个元素的值在0和1之间。

然后，将replacing_dim_numerical减去最大值（在每行上取最小值，并保持维度），并与0.3比较，得到一个布尔值数组。将这个操作的结果赋值给replacing_dim_numerical。

如果num_categorical大于0，将replacing_dim_categorical减去最大值（在每行上取最小值，并保持维度），并与0.3比较，得到一个布尔值数组。将这个操作的结果赋值给replacing_dim_categorical。

接下来，创建一个空列表x_rep，用于存储替换的区间。

然后，使用torch.zeros函数创建一个形状为(n_batch, data_seq_len)的张量x_anomaly，所有元素的值都为0，并将其转移到设备（device）上。

这段代码生成了替换数据的类型和异常点，可用于进一步处理数据和进行训练。

注释掉的代码部分应该是用于生成替换维度的布尔值数组replacing_dim，但是由于代码被注释掉了，所以不会执行。
        """
        replacing_type = np.random.uniform(0., 1., size=(n_batch,))
        replacing_dim_numerical = np.random.uniform(0., 1., size=(n_batch, num_numerical))
        replacing_dim_categorical = np.random.uniform(0., 1., size=(n_batch, num_categorical))
        
        replacing_dim_numerical = replacing_dim_numerical\
                                  - np.maximum(replacing_dim_numerical.min(axis=1, keepdims=True), 0.3) <= 0.001
        if num_categorical > 0:
            replacing_dim_categorical = replacing_dim_categorical\
                                        - np.maximum(replacing_dim_categorical.min(axis=1, keepdims=True), 0.3) <= 0.001
        
#         replacing_dim = np.empty(n_batch, d_data, dtype=bool)
#         replacing_dim[numerical_column] = replacing_dim_numerical
#         replacing_dim[categorical_column] = replacing_dim_categorical

        x_rep = []  # list of replaced intervals
        x_anomaly = torch.zeros(n_batch, data_seq_len, device=device)  # list of anomaly points
        
        # Create anomaly intervals.
        """
        循环，每次循环都会对输入数据进行一些替换和噪声操作。以下是代码中不同部分的解释：
针对长度大于0的情况：
    将 x[j][tar:tar+leng] 拷贝到 x_rep 列表中；
    对 x_rep[-1] 进行转置，保存在 _x 变量中；
    计算 rep_len_num 和 rep_len_cat，分别表示替换的数值列和分类列的数量；
    根据 dim_num 和 dim_cat 中的索引获取数值和分类列的名称；
    如果 rep_len_cat 大于0，则获取分类列的名称。
对于 typ 大于 soft_replacing_prob 的情况，进行外部区间替换：
    对数值列进行替换，替换的列数由 rep_len_num 决定；
    对于每一列，从 numerical_column 中随机选择一个列，根据 rep 中的索引获取替换的起始位置和长度，再根据 filp 来决定是否进行翻转操作，将随机替换的区间赋值给 _x_temp；
    将 _x_temp 乘以权重 weights，并加上原始数据乘以 (1 - weights)，最后将结果赋值给 _x[target_column_numerical]；
    对于分类列，进行类似的处理。
对于 typ 大于 uniform_replacing_prob 的情况，进行均匀替换：
    生成一个 rep_len_num × 1 的随机数矩阵，赋值给 _x[target_column_numerical]；
    对于分类列，可以类似地生成一个随机的 0/1 矩阵进行替换。
对于 typ 大于 peak_noising_prob 的情况，进行峰值噪声：
    随机选择一个峰值索引 peak_index，并根据 _x[target_column_numerical, peak_index] 的值来生成噪声值 peak_value，然后将其赋值给 _x[target_column_numerical, peak_index]；
    对于分类列，可以类似地生成一个随机的 0/1 矩阵进行替换；
    将受影响的区间标记为异常，并将 _x 赋值给 x[j]。
对于 typ 大于 length_adjusting_prob 的情况，进行长度调整（仅适用于二元交叉熵损失）：
    如果长度大于 replacing_len_half_max，则进行长度扩展，将后面的数据向后移动并复制；
    如果长度较小，则进行长度缩短，将数据从前面的原始位置和原始索引中复制。
对于 typ 小于 white_noising_prob 的情况：
    使用 torch.normal 函数生成一个均值为 0，标准差为 0.003，大小为 (rep_len_num, leng) 的随机数矩阵；
    将生成的随机数矩阵与 _x[target_column_numerical] 相加，并使用 clamp 函数将值限制在 0 到 1 之间；
    将受影响的区间标记为异常，并将 _x 赋值给 x[j]。
否则，将 x_rep 列表中的最后一个元素设置为 None
        """
        for j, rep, tar, leng, typ, dim_num, dim_cat in zip(range(n_batch), replacing_index, target_index, replacing_lengths,
                                                            replacing_type, replacing_dim_numerical, replacing_dim_categorical):
            if leng > 0:
                x_rep.append(x[j][tar:tar+leng].clone())
                _x = x_rep[-1].clone().transpose(0, 1)
                rep_len_num = len(dim_num[dim_num])
                rep_len_cat = len(dim_cat[dim_cat]) if len(dim_cat) > 0 else 0
                target_column_numerical = numerical_column[dim_num]
                if rep_len_cat > 0:
                    target_column_categorical = categorical_column[dim_cat]
                
                # External interval replacing
                if typ > soft_replacing_prob:
                    # Replacing for numerical columns
                    _x_temp = []
                    col_num = np.random.choice(numerical_column, size=rep_len_num)
                    filp = np.random.randint(0, 2, size=(rep_len_num,2)) > 0.5
                    for _col, _rep, _flip in zip(col_num, rep[:rep_len_num], filp):
                        random_interval = replacing_data[_rep:_rep+leng, _col].copy()
                        # fliping options
                        if horizontal_flip and _flip[0]:
                            random_interval = random_interval[::-1].copy()
                        if vertical_flip and _flip[1]:
                            random_interval = 1 - random_interval
                        _x_temp.append(torch.from_numpy(random_interval))
                    _x_temp = torch.stack(_x_temp).to(device)
                    weights = torch.from_numpy(replacing_weights(leng)).float().unsqueeze(0).to(device)
                    _x[target_column_numerical] = _x_temp * weights + _x[target_column_numerical] * (1 - weights)

                    # Replacing for categorical columns
                    if rep_len_cat > 0:
                        _x_temp = []
                        col_cat = np.random.choice(categorical_column, size=rep_len_cat)
                        for _col, _rep in zip(col_cat, rep[-rep_len_cat:]):
                            _x_temp.append(torch.from_numpy(replacing_data[_rep:_rep+leng, _col].copy()))
                        _x_temp = torch.stack(_x_temp).to(device)
                        _x[target_column_categorical] = _x_temp

                        x_anomaly[j, tar:tar+leng] = 1
                        x[j][tar:tar+leng] = _x.transpose(0, 1)

                # Uniform replacing
                elif typ > uniform_replacing_prob:
                    _x[target_column_numerical] = torch.rand(rep_len_num, 1, device=device)
#                     _x[target_column_categorical] = torch.randint(0, 2, size=(rep_len_cat, 1), device=device).float()
                    x_anomaly[j, tar:tar+leng] = 1
                    x[j][tar:tar+leng] = _x.transpose(0, 1)

                # Peak noising
                elif typ > peak_noising_prob:
                    peak_index = np.random.randint(0, leng)
                    peak_value = (_x[target_column_numerical, peak_index] < 0.5).float().to(device)
                    peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * torch.rand(rep_len_num, device=device)
                    _x[target_column_numerical, peak_index] = peak_value

#                     peak_value = (_x[target_column_categorical, peak_index] < 0.5).float().to(device)
#                     _x[target_column_categorical, peak_index] = peak_value
                    
                    peak_index = tar + peak_index
                    tar_first = np.maximum(0, peak_index - options.patch_size)
                    tar_last = peak_index + options.patch_size + 1
                    
                    x_anomaly[j, tar_first:tar_last] = 1
                    x[j][tar:tar+leng] = _x.transpose(0, 1)
                    
                # Length adjusting (only for bce loss)
                elif typ > length_adjusting_prob:
                    # Lengthening
                    if leng > replacing_len_half_max:
                        scale = np.random.randint(2, 5)
                        _leng = leng - leng % scale
                        scaled_leng = _leng // scale
                        x[j][tar+_leng:] = x[j][tar+scaled_leng:-_leng+scaled_leng].clone()
                        x[j][tar:tar+_leng] = torch.repeat_interleave(x[j][tar:tar+scaled_leng], scale, axis=0)
                        x_anomaly[j, tar:tar+_leng] = 1
                    # Shortening
                    else:
                        origin_index = first_index[j]
                        if origin_index > replacing_len_max * 1.5:
                            scale = np.random.randint(2, 5)
                            _leng = leng * (scale - 1)
                            x[j][:tar] = torch.Tensor(train_data[origin_index-_leng:origin_index+tar-_leng].copy()).to(device)
                            x[j][tar:tar+leng] = torch.Tensor(train_data[origin_index+tar-_leng:origin_index+tar+leng:scale].copy()).to(device)
                            x_anomaly[j, tar:tar+leng] = 1
                    
                # White noising (deprecated)
                elif typ < white_noising_prob:
                    _x[target_column_numerical] = (_x[target_column_numerical]\
                                                   + torch.normal(mean=0., std=0.003, size=(rep_len_num, leng), device=device))\
                                                  .clamp(min=0., max=1.)
                    x_anomaly[j, tar:tar+leng] = 1
                    x[j][tar:tar+leng] = _x.transpose(0, 1)
                
                else:
                    x_rep[-1] = None
            
            else:
                x_rep.append(None)
            
        # Process data.
        z = torch.stack(x)
        y = model(z)

        # Compute losses.
        if options.loss == 'bce':
            y = y.squeeze(-1)
            loss = train_loss(sigmoid(y), x_anomaly)
#             partial_loss = 0

#             for pred, gt, ano_label, tar, leng in zip(y, x_rep, x_anomaly, target_index, replacing_lengths):
#                 if leng > 0 and gt != None:
#                     partial_loss += train_loss(sigmoid(pred[tar:tar+leng]), ano_label[tar:tar+leng])
#             loss += options.partial_loss * partial_loss
        
        else:
            loss = options.total_loss * train_loss(x_true, y)
            partial_loss = 0

            for pred, gt, tar, leng in zip(y, x_rep, target_index, replacing_lengths):
                if leng > 0 and gt != None:
                    partial_loss += train_loss(pred[tar:tar+leng], gt.to(device))
            if not torch.isnan(partial_loss):
                loss += options.partial_loss * partial_loss

#         if options.contrastive_loss > 0:
#             con_loss = contrastive_loss(features, x_anomaly)
#             loss += options.contrastive_loss * con_loss

        # Print training summary.
        """
        训练循环，每隔一定步数进行一次训练和评估。
如果当前迭代次数 i 是 options.summary_steps 的倍数：
    使用 torch.no_grad() 上下文管理器，表示在此范围内不需要计算梯度；
    如果 options.loss 是 'bce'，则进行二分类预测，将 sigmoid(y) 大于 0.5 的值转换为整数，并将 x_anomaly 转换为布尔型整数；
    计算准确率 acc，即预测值和 x_anomaly 相等的元素个数除以总数据量 total_data_num；
    使用 summary_writer 添加训练损失和准确率的标量值；
    将 model 设为评估模式，使用 estimate 函数对测试数据进行评估，得到异常评估结果 estimation；
    将 estimation 的第一列转为 numpy 数组；
    将 model 设为训练模式；
    初始化 best_eval 和 best_rate 为 0；
    对于 rate 在 0.001 到 0.301 之间以 0.001 为步长的范围内，计算 F1-score，并更新 best_eval 和 best_rate；
    使用 summary_writer 添加最佳异常率、精确率、召回率和 F1-score 的标量值；
    打印当前迭代次数、损失值、训练准确率、最佳异常率、精确率、召回率和 F1-score；
    保存模型参数。
更新梯度：
    清零优化器的梯度；
    计算损失的梯度；
    使用 nn.utils.clip_grad_norm_ 对模型参数的梯度进行裁剪，以防止梯度爆炸；
    使用优化器更新模型参数；
    调用 scheduler.step_update(i) 进行学习率更新。
最后，保存模型参数到文件。
        """
        if i % options.summary_steps == 0:
            with torch.no_grad():
                if options.loss == 'bce':
                    pred = (sigmoid(y) > 0.5).int()
                    x_anomaly = x_anomaly.bool().int()
                    total_data_num = n_batch * data_seq_len
                    
                    acc = (pred == x_anomaly).int().sum() / total_data_num
                    summary_writer.add_scalar('Train/Loss', loss.item(), i)
                    summary_writer.add_scalar('Train/Accuracy', acc, i)
                    
                    model.eval()
                    estimation = estimate(test_data, model,
                                          sigmoid if options.loss == 'bce' else nn.Identity().to(device),
                                          1 if options.loss == 'bce' else d_data,
                                          n_batch, options.window_sliding, divisions, None, device)
                    estimation = estimation[:, 0].cpu().numpy()
                    model.train()
                    
                    best_eval = (0, 0, 0)
                    best_rate = 0
                    for rate in np.arange(0.001, 0.301, 0.001):
                        evaluation = f1_score(test_label, estimation, rate, False, False)
                        if evaluation[2] > best_eval[2]:
                            best_eval = evaluation
                            best_rate = rate
                    summary_writer.add_scalar('Valid/Best Anomaly Rate', best_rate, i)
                    summary_writer.add_scalar('Valid/Precision', best_eval[0], i)
                    summary_writer.add_scalar('Valid/Recall', best_eval[1], i)
                    summary_writer.add_scalar('Valid/F1', best_eval[2], i)
                    
                    print(f'iteration: {i} | loss: {loss.item():.10f} | train accuracy: {acc:.10f}')
                    print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
                    
                else:
                    origin = rec_loss(z[:, :, numerical_column], x_true).item()
                    rec = rec_loss(x_true, y).item()

                    summary_writer.add_scalar('Train Loss', loss.item(), i)
                    summary_writer.add_scalar('Original Error', origin, i)
                    summary_writer.add_scalar('Reconstruction', rec, i)
                    summary_writer.add_scalar('Error rate', rec/origin, i)

                    print('iter ', i, ',\tloss : {:.10f}'.format(loss.item()), ',\torigin err : {:.10f}'.format(origin), ',\trec : {:.10f}'.format(rec), sep='')
                    print('\t\terr rate : {:.10f}'.format(rec/origin), sep='')
                    print()
            torch.save(model.state_dict(), os.path.join(log_dir, 'state/state_dict_step_{}.pt'.format(i)))

        # Update gradients.
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

        optimizer.step()
        scheduler.step_update(i)

    torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_steps", default=150000, type=int, help='maximum_training_steps')
    parser.add_argument("--summary_steps", default=500, type=int, help='steps for summarizing and saving of training log')
    parser.add_argument("--checkpoint", default=None, type=str, help='load checkpoint file')
    parser.add_argument("--initial_iter", default=0, type=int, help='initial iteration for training')
    
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    parser.add_argument("--replacing_data", default=None, type=str, help='external data for soft replacement; None(default)/SMAP/MSL/SMD/SWaT/WADI')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_features", default=512, type=int, help='number of features for a window')
    parser.add_argument("--patch_size", default=4, type=int, help='number of data points in a patch')
    parser.add_argument("--d_embed", default=512, type=int, help='embedding dimension of feature')
    parser.add_argument("--n_layer", default=6, type=int, help='number of transformer layers')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--replacing_rate_max", default=0.15, type=float, help='maximum ratio of replacing interval length to window size')
    
    parser.add_argument("--soft_replacing", default=0.5, type=float, help='probability for soft replacement')
    parser.add_argument("--uniform_replacing", default=0.15, type=float, help='probability for uniform replacement')
    parser.add_argument("--peak_noising", default=0.15, type=float, help='probability for peak noise')
    parser.add_argument("--length_adjusting", default=0.0, type=float, help='probability for length adjustment')
    parser.add_argument("--white_noising", default=0.0, type=float, help='probability for white noise (deprecated)')
    
    parser.add_argument("--flip_replacing_interval", default='all', type=str,
                        help='allowance for random flipping in soft replacement; vertical/horizontal/all/none')
    parser.add_argument("--replacing_weight", default=0.7, type=float, help='weight for external interval in soft replacement')
    
    parser.add_argument("--window_sliding", default=16, type=int, help='sliding steps of windows for validation')
    parser.add_argument("--data_division", default=None, type=str, help='data division for validation; None(default)/channel/class/total')
    
    parser.add_argument("--loss", default='bce', type=str, help='loss type')
    parser.add_argument("--total_loss", default=0.2, type=float, help='total loss weight (deprecated)')
    parser.add_argument("--partial_loss", default=1., type=float, help='partial loss weight (deprecated)')
    parser.add_argument("--contrastive_loss", default=0., type=float, help='contrastive loss weight (deprecated)')
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)
    
    parser.add_argument("--default_options", default=None, type=str, help='default options for datasets; None(default)/SMAP/MSL/SMD/SWaT/WADI')
    """
    根据命令行参数解析并加载默认选项，然后调用主函数进行处理。以下是代码中不同部分的解释：

使用 parser.parse_args() 解析命令行参数，并将结果赋值给 options 变量。
如果 options.default_options 不为 None：
    如果 options.default_options 以 'SMD' 开头，将其赋值给 default_options 变量；
    加载名为 'data/default_options_SMD.pt' 的文件，并将结果赋值给 options；
    将 options.dataset 设置为 default_options；
    否则，加载名为 'data/default_options_' + options.default_options + '.pt' 的文件，并将结果赋值给 options。
调用 main(options) 函数，将 options 作为参数传递给主函数进行处理。
    """
    options = parser.parse_args()
    if options.default_options != None:
        if options.default_options.startswith('SMD'):
            default_options = options.default_options
            options = torch.load('data/default_options_SMD.pt')
            options.dataset = default_options
        else:
            options = torch.load('data/default_options_'+options.default_options+'.pt')
    
    main(options)