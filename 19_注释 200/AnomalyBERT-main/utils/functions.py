import numpy as np
import copy

import torch
import torch.nn as nn



# Make clones of a layer.
def clone_layer(module, N):
    """
    这段代码定义了一个名为clone_layer的函数，用于克隆一个模块。

函数接收两个参数，一个是要克隆的模块（module），另一个是克隆的次数（N）。

通过使用列表推导式和copy.deepcopy函数，将模块克隆N次，并将克隆的模块存储在nn.ModuleList中。

最后，返回存储克隆模块的nn.ModuleList。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Make masking matrix.
"""
这段代码定义了一个名为masking_matrix的函数，用于生成一个掩码矩阵。

函数接收多个参数，包括批次大小（n_batch）、最大序列长度（max_seq_len）、掩码标记长度（mask_lengths）、掩码标记的第一个索引（first_indices）和设备类型（device）。

首先，创建一个形状为（n_batch，max_seq_len，1）的全零张量mask，并将其转换为布尔类型。

然后，根据掩码标记的长度和第一个索引，计算出掩码标记的最后一个索引last_indices。

接下来，使用循环遍历每个批次的序列，并在掩码矩阵中将对应的位置标记为True。

最后，返回生成的掩码矩阵。
"""
def masking_matrix(n_batch, max_seq_len, mask_lengths, first_indices, device='cpu'):
    """
    <input info>
    n_batch : batch size (number of sequences)
    max_seq_len : maximum sequence length
    mask_lengths : (n_batch,), mask token length
    first_indices : (n_batch,), first indices of mask tokens
    device : device of masking matrix
    """
    mask = torch.zeros(n_batch, max_seq_len, 1, device=device).bool()
    last_indices = first_indices + mask_lengths
    for i, first, last in zip(range(n_batch), first_indices, last_indices):
        mask[i, first:last] = True
    return mask