import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from utils.functions import clone_layer



# Main transformer encoder
class TransformerEncoder(nn.Module):
    """
    这段代码定义了一个TransformerEncoder类，它是一组Transformer编码器层的堆叠。构造函数接收一个位置编码层、一个编码器层和编码器的层数作为参数。

    在forward方法中，输入张量x首先通过位置编码层进行编码（如果位置编码层存在）。然后，输入张量通过一系列的编码器层进行编码。最后，返回编码后的张量。

    这个类的作用是将输入序列通过多个Transformer编码器层进行编码，以获取表示输入的高级特征。
    """
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = clone_layer(encoder_layer, n_layer)
            
        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer
        
    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        position_vector = None
        if self.positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for layer in self.encoder_layers:
            out = layer(out)

        return out
    
    
# Encoder layer
class EncoderLayer(nn.Module):
    """
    这段代码定义了一个EncoderLayer类，它包含了一个注意力层（attention_layer）、一个前馈层（feed_forward_layer）、一个归一化层（norm_layer）和一个Dropout层（dropout_layer）。

    在构造函数中，注意力层、前馈层和归一化层被初始化，并且权重参数被初始化为Xavier均匀分布或零。dropout参数用于初始化Dropout层。

    在forward方法中，输入张量x首先通过第一个归一化层进行归一化。然后，输入张量通过注意力层进行处理，并通过Dropout层进行随机失活。注意力层的输出与输入张量进行残差连接（加法操作），得到out1。

    out1再通过第二个归一化层进行归一化。然后，out1通过前馈层进行处理，并再次通过Dropout层进行随机失活。前馈层的输出与out1进行残差连接，得到最终输出out2。

    这个类的作用是定义Transformer编码器层的结构和操作。
    """
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        
    def forward(self, x):
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.attention_layer(out1)
        out1 = self.dropout_layer(out1) + x
        
        out2 = self.norm_layers[1](out1)
        out2 = self.feed_forward_layer(out2)
        return self.dropout_layer(out2) + out1
    
    
    
class MultiHeadAttentionLayer(nn.Module):
    """
    定义了一个MultiHeadAttentionLayer类，它实现了多头注意力机制。构造函数接收一个输入的维度（d_embed）、头数（n_head）、最大序列长度（max_seq_len）和一个布尔值（relative_position_embedding）作为参数。

    在构造函数中，首先检查d_embed是否能被n_head整除。然后计算每个头的维度（d_k）以及缩放因子（scale）。

    接下来，使用克隆函数（clone_layer）复制三个线性层（word_fc_layers）来对输入进行线性变换，得到查询（query）、键（key）和值（value）。

    然后，将查询、键和值进行分割，并进行相应的维度变换，以便进行多头注意力计算。

    接下来，计算注意力得分（scores），并根据需要添加相对位置嵌入（relative_position_embedding）。如果relative_position_embedding为True，将获取预先计算好的相对位置嵌入表（relative_position_embedding_table），并根据相对位置索引（relative_position_index）将其添加到注意力得分中。

    然后使用softmax函数对得分进行归一化，并利用注意力权重对值进行加权求和，得到注意力输出。

    最后，将注意力输出通过线性层（output_fc_layer）进行变换，得到最终输出。

    这个类的作用是实现多头注意力机制，用于对输入进行加权求和得到注意力表示。
    """
    def __init__(self, d_embed, n_head, max_seq_len=512, relative_position_embedding=True):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_embed % n_head == 0  # Ckeck if d_model is divisible by n_head.
        
        self.d_embed = d_embed
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)
        
        self.word_fc_layers = clone_layer(nn.Linear(d_embed, d_embed), 3)
        self.output_fc_layer = nn.Linear(d_embed, d_embed)
        
        self.max_seq_len = max_seq_len
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            # Table of 1D relative position embedding
            self.relative_position_embedding_table = nn.Parameter(torch.zeros(2*max_seq_len-1, n_head))
            trunc_normal_(self.relative_position_embedding_table, std=.02)
            
            # Set 1D relative position embedding index.
            coords_h = np.arange(max_seq_len)
            coords_w = np.arange(max_seq_len-1, -1, -1)
            coords = coords_h[:, None] + coords_w[None, :]
            self.relative_position_index = coords.flatten()

    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        n_batch = x.shape[0]
        device = x.device
        
        # Apply linear layers.
        query = self.word_fc_layers[0](x)
        key = self.word_fc_layers[1](x)
        value = self.word_fc_layers[2](x)
        
        # Split heads.
        query_out = query.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        key_out = key.view(n_batch, -1, self.n_head, self.d_k).contiguous().permute(0, 2, 3, 1)
        value_out = value.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute attention and concatenate matrices.
        scores = torch.matmul(query_out * self.scale, key_out)
        
        # Add relative position embedding
        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[self.relative_position_index].view(
                self.max_seq_len, self.max_seq_len, -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous().unsqueeze(0)
            scores = scores + position_embedding
        
#         if masking_matrix != None:
#             scores = scores + masking_matrix * (-1e9) # Add very small negative number to padding columns.
        probs = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(probs, value_out)
        
        # Convert 4d tensor to proper 3d output tensor.
        attention_out = attention_out.transpose(1, 2).contiguous().view(n_batch, -1, self.d_embed)
            
        return self.output_fc_layer(attention_out)

    
    
class PositionWiseFeedForwardLayer(nn.Module):
    """
    这段代码定义了一个PositionWiseFeedForwardLayer类，它实现了位置前馈层。构造函数接收输入维度（d_embed）、前馈层维度（d_ff）和dropout概率（dropout）作为参数。

    在构造函数中，定义了两个线性层（first_fc_layer和second_fc_layer），一个激活函数层（activation_layer）和一个Dropout层（dropout_layer）。

    在forward方法中，输入张量x首先通过第一个线性层进行线性变换。然后，将变换后的张量通过激活函数层进行非线性变换，并通过Dropout层进行随机失活。最后，通过第二个线性层再次进行线性变换，得到最终输出。

    这个类的作用是实现位置前馈层，用于对输入进行非线性变换。
    """
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.dropout_layer(self.activation_layer(out))
        return self.second_fc_layer(out)
    
    
    
# Sinusoidal positional encoding
"""
这段代码定义了一个SinusoidalPositionalEncoding类，它实现了正弦位置编码。构造函数接收输入维度（d_embed）、最大序列长度（max_seq_len）和dropout概率（dropout）作为参数。

在构造函数中，定义了一个Dropout层（dropout_layer）。

然后，通过torch.arange函数生成一个形状为（max_seq_len，1）的位置张量positions，用于表示序列中每个位置的索引。同时，通过torch.arange函数生成一个形状为（d_embed//2，）的分母张量denominators，用于计算正弦和余弦函数的分母。

接下来，通过torch.matmul函数将位置张量positions和分母张量denominators相乘得到编码矩阵encoding_matrix。

然后，创建一个形状为（1，max_seq_len，d_embed）的空编码张量encoding。通过切片操作，将正弦编码和余弦编码分别填充到编码张量的奇数列和偶数列。

最后，通过self.register_buffer函数将编码张量encoding注册为模型的缓冲区。

这个类的作用是实现正弦位置编码，用于为输入序列的每个位置添加位置信息。
"""
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(torch.arange(0, d_embed, 2) * (np.log(0.0001) / d_embed)).unsqueeze(0)
        encoding_matrix = torch.matmul(positions, denominators)
        
        encoding = torch.empty(1, max_seq_len, d_embed)
        encoding[0, :, 0::2] = torch.sin(encoding_matrix)
        encoding[0, :, 1::2] = torch.cos(encoding_matrix[:, :(d_embed//2)])

        self.register_buffer('encoding', encoding)
        
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.encoding)
    
    
# Absolute position embedding
"""
这段代码定义了一个AbsolutePositionEmbedding类，它实现了绝对位置嵌入。构造函数接收输入维度（d_embed）、最大序列长度（max_seq_len）和dropout概率（dropout）作为参数。

在构造函数中，定义了一个Dropout层（dropout_layer）和一个可学习的嵌入层（embedding）。嵌入层的形状为（1，max_seq_len，d_embed），用于存储绝对位置嵌入。

在forward方法中，输入张量x首先通过dropout层进行随机失活。然后，将dropout后的张量与嵌入层相加，得到绝对位置嵌入后的张量。

这个类的作用是实现绝对位置嵌入，用于为输入序列的每个位置添加位置信息。
"""
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_embed))
        trunc_normal_(self.embedding, std=.02)
        
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.embedding)
    
    

# Get a transformer encoder with its parameters.
"""
这段代码定义了一个名为get_transformer_encoder的函数，用于获取Transformer编码器。

函数接收多个参数，包括输入维度（d_embed）、位置编码类型（positional_encoding）、是否使用相对位置嵌入（relative_position_embedding）、编码器层数（n_layer）、注意力头数（n_head）、前馈层维度（d_ff）、最大序列长度（max_seq_len）和dropout概率（dropout）。

根据位置编码类型的不同，选择不同的位置编码层。如果位置编码类型为"Sinusoidal"、"sinusoidal"或"sin"，则使用SinusoidalPositionalEncoding类进行位置编码。如果位置编码类型为"Absolute"、"absolute"或"abs"，则使用AbsolutePositionEmbedding类进行位置编码。如果位置编码类型为None或"None"，则位置编码层为空。

接下来，创建注意力层、前馈层、归一化层和编码器层。注意力层使用MultiHeadAttentionLayer类，前馈层使用PositionWiseFeedForwardLayer类，归一化层使用nn.LayerNorm类，编码器层使用EncoderLayer类。

最后，返回一个TransformerEncoder对象，它包含位置编码层、编码器层和编码器层数。
"""
def get_transformer_encoder(d_embed=512,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            n_layer=6,
                            n_head=8,
                            d_ff=2048,
                            max_seq_len=512,
                            dropout=0.1):
    
    if positional_encoding == 'Sinusoidal' or positional_encoding == 'sinusoidal' or positional_encoding == 'sin':
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding == 'Absolute' or positional_encoding =='absolute' or positional_encoding == 'abs':
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None
    
    attention_layer = MultiHeadAttentionLayer(d_embed, n_head, max_seq_len, relative_position_embedding)
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)
    
    return TransformerEncoder(positional_encoding_layer, encoder_layer, n_layer)
