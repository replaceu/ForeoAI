#======================0.导入依赖库=============================#
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

#训练集（句子输入部分）#
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
#测试集（构建词表）#
#编码端的词表
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
#src_vocab_size：实际情况下，它的长度应该是所有德语单词的个数
src_vocab_size = len(src_vocab)
# 解码端的词表
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
# 实际情况下，它应该是所有英语单词个数
tgt_vocab_size = len(tgt_vocab)
def make_batch(sentences):
    # 输入数据集
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    # 输出数据集
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    # 目标数据集
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch),torch.LongTensor(output_batch),torch.LongTensor(target_batch)

#自定义DataLoader
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
         return self.enc_inputs[idx],self.dec_inputs[idx],self.dec_outputs[idx]

    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

src_len = 5  # length of source 编码端的输入长度
tgt_len = 5  # length of target 解码端的输入长度

#Transformer参数#
d_model = 512  # Embedding Size 每一个字符转化成Embedding的大小
d_ff = 2048  # FeedForward dimension 前馈神经网络映射到多少维度
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer  encoder和decoder的个数，这个设置的是6个encoder和decoder堆叠在一起
n_heads = 8  # number of heads in Multi-Head Attention  多头注意力机制时，把头分为几个，这里说的是分为8个

"""
主要流程:
1.输入文本进行词嵌入和位置编码，作为最终的文本嵌入
2.文本嵌入经过Encoder编码，得到注意力加权后输出的编码向量以及自注意力权重矩阵
3.将编码向量和样本共同输入解码器，经过注意力加权等操作后输出最中的上下文向量
4.映射到词表大小的线性层上进行解码生成文本
5.最终返回代表预测结果的逻辑矩阵
"""

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        #编码层
        self.encoder = Encoder().to(device)
        #解码层
        self.decoder = Decoder().to(device)
        #输出层，d_model是解码层每一个token输出维度的大小，之后会做一个tgt_vocab_size大小的softmax
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    # 实现函数
    def forward(self, enc_inputs, dec_inputs):
        """
         Transformers的输入：两个序列（编码端的输入，解码端的输入）
         enc_inputs(编码端输入): [batch_size, src_len] 形状：batch_size乘src_len
         dec_inputs(解码端输入): [batch_size, tgt_len] 形状：batch_size乘tgt_len
         """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs:[batch_size, src_len, d_model]
        # enc_self_attns:[n_layers, batch_size, n_heads, src_len, src_len]
        # enc_outputs:编码端输出
        # enc_self_attns:Q、K转置相乘后softmax的矩阵值，代表每个单词和其他单词的相关性

        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]

        # dec_outputs:解码端输出，用于后续的linear映射
        # dec_self_attns:类比于enc_self_attns是查看每个单词对解码端中输入的其余单词的相关性
        # dec_enc_attns:解码端中每个单词对encoder中每个单词的相关性
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

# --------#
# 模型训练
# --------#

#调用Transformer模型
model = Transformer().to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss(ignore_index=0)
# 优化器,用Adam的话效果不好
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
        """
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

        # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------------------------------#
# EncoderLayer：包含两个部分，多头注意力机制和前馈神经网络
# ---------------------------------------------------#
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        #多头注意力机制
        self.enc_self_attn = MultiHeadAttention()
        #前馈神经网络
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model]，
        需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数.
        """
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# -----------------------------------------------------------------------------#
# Encoder部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#


"""
主要流程：
1.输入文本的索引tensor,经过词嵌入层（Word enbedding),然后和位置编码线性相加作为输入层的最终输出
2.随后每一层的输出作为下一层编码块的输入，在每个编码块里进行注意力计算、前馈神经网络、残差连接、层归一化等操作
3.最终返回编码器最后一层的输出和每一层的注意力权重矩阵
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 这行其实就是生成一个矩阵，大小是: src_vocab_size * d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.pos_emb = PositionalEncoding(d_model)
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        这里我们的enc_inputs形状是： [batch_size x source_len]
        """
        # 下面这行代码通过src_emb进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        # 这行是位置编码，把两者相加放到了pos_emb函数里面
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # 去看EncoderLayer层函数
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

#Padding Mask：形成一个符号矩阵
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # 最终得到的应该是一个最后n列为1的矩阵，即K的最后n个token为PAD。
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# -----------------------------------------------------------------------------#
# Decoder Layer包含了三个部分：解码器自注意力、“编码器-解码器”注意力、基于位置的前馈网络
# -----------------------------------------------------------------------------#
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# -----------------------------------------------------------------------------#
# Decoder 部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #词向量embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        #位置编码
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]

        #get_attn_pad_mask 自注意力层的时候的pad部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        #get_attn_subsequent_mask这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        #两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        #这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 生成一个形状为[max_len,d_model]的全为0的tensor
        pe = torch.zeros(max_len, d_model)
        # position:[max_len,1]，即[5000,1]，这里插入一个维度是为了后面能够进行广播机制然后和div_term直接相乘
        # 注意，要理解一下这里position的维度。每个pos都需要512个编码。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 共有项，利用指数函数e和对数函数log取下来，方便计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 这里position * div_term有广播机制，因为div_term的形状为[d_model/2],即[256],符合广播条件，广播后两个tensor经过复制，形状都会变成[5000,256]，*表示两个tensor对应位置处的两个元素相乘
        # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置赋值给pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # 同理，这里是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        # 上面代码获取之后得到的pe:[max_len*d_model]

        # 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 定一个缓冲区，其实简单理解为这个参数不更新就可以，但是参数仍然作为模型的参数保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # 这里的self.pe是从缓冲区里拿的
        # 切片操作，把pe第一维的前seq_len个tensor和x相加，其他维度不变
        # 这里其实也有广播机制，pe:[max_len,1,d_model]，第二维大小为1，会自动扩张到batch_size大小。
        # 实现词嵌入和位置编码的线性相加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 输入进来的维度分别是
        # Q:[batch_size x n_heads x len_q x d_k]
        # K:[batch_size x n_heads x len_k x d_k]
        # V:[batch_size x n_heads x len_k x d_v]
        # matmul操作即矩阵相乘
        # [batch_size x n_heads x len_q x d_k] matmul [batch_size x n_heads x d_k x len_k] -> [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # masked_fill_(mask,value)这个函数，用value填充源向量中与mask中值为1位置相对应的元素，
        # 要求mask和要填充的源向量形状需一致
        # 把被mask的地方置为无穷小，softmax之后会趋近于0，Q会忽视这部分的权重
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context:[batch_size,n_heads,len_q,d_k]
        # attn:[batch_size,n_heads,len_q,len_k]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Wq,Wk,Wv其实就是一个线性层，用来将输入映射为Q、K、V
        # 这里输出是d_k*n_heads，因为是先映射，后分头

        #d_model=512 d_k=d_v=64 n_heads=8
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        #LayerNorm进行归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # attn_mask:[batch_size,len_q,len_k]
        # 输入的数据形状：
        # Q: [batch_size x len_q x d_model],
        # K: [batch_size x len_k x d_model],
        # V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # 分头；一定要注意的是q和k分头之后维度是一致的，所以一看这里都是d_k
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # v_s: [batch_size x n_heads x len_k x d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask:[batch_size x len_q x len_k] ---> [batch_size x n_heads x len_q x len_k]
        # 就是把pad信息复制n份，重复到n个头上以便计算多头注意力机制
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算ScaledDotProductAttention
        # 得到的结果有两个：context:[batch_size x n_heads x len_q x d_v],
        # attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # 这里实际上在拼接n个头，把n个头的加权注意力输出拼接，然后过一个线性层，context变成
        # [batch_size,len_q,n_heads*d_v]。这里context需要进行contiguous，因为transpose后源tensor变成不连续的
        # 了，view操作需要连续的tensor。
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        # 过残差、LN，输出output: [batch_size x len_q x d_model]和这一层的加权注意力表征向量
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]