import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

#设置输入流水线
#该数据集包含来约 50000 条训练样本，1100 条验证样本，以及 2000 条测试样本
examples,metadata = tfds.load('ted_hrlr_translate/pt_to_en',with_info=True,as_supervised=True)
train_examples,val_examples = examples['train'],examples['validation']
#从训练数据集创建自定义子词分词器
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy for pt,en in train_examples),target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy for pt,en in train_examples),target_vocab_size=2**13)

sample_string = 'transformer is awesome'
tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string

#如果单词不在词典中，则分词器（tokenizer）通过将单词分解为子词来对字符串进行编码
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

BUFFER_SIZE=20000
BATCH_SIZE=64
#将开始和结束标记（token）添加到输入和目标
def encode(lang1,lang2):
    lang1 = [tokenizer_pt.vocab_size]+tokenizer_pt.encode(lang1.numpy())+[tokenizer_pt.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1,lang2
#为了使本示例较小且相对较快，删除长度大于40个标记的样本
MAX_LENGTH = 40
def filter_max_length(x,y,max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x)<=max_length,tf.size(y)<=max_length)

def tf_encode(pt,en):
    result_pt,result_en = tf.py_function(encode,[pt,en],[tf.int64,tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt,result_en

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
#将数据集缓存到内存中以加快读取速度
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch,en_batch = next(iter(val_dataset))

#位置编码
def get_angles(pos,i,d_model):
    angle_rates = 1/np.power(10000,(2*(i//2))/np.float32(d_model))
    return pos*angle_rates

def positional_encoding(position,d_model):
    angle_rads = get_angles(np.arange(position)[:,np.newaxis],np.arange(d_model)[np.newaxis,:],d_model)
    #将sin应用于数组中的偶数索引-2i
    angle_rads[:,0::2] = np.sin(angle_rads[:, 0::2])
    # 将cos应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, 512)
print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()

#遮挡一批序列中所有的填充标记（pad tokens）,确保模型不会将填充作为输入
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0),tf.float32)

    #添加额外的维度
    return seq[:,tf.nexaxis,tf.newaxis,:]# (batch_size, 1, 1, seq_len)

#前瞻遮挡（look-ahead mask）用于遮挡一个序列中的后续标记（future tokens）。换句话说，该 mask 表明了不应该使用的条目。
#这意味着要预测第三个词，将仅使用第一个和第二个词。与此类似，预测第四个词，仅使用第一个，第二个和第三个词，依此类推
def create_look_ahead_mask(size):
    mask = 1-tf.linalg.band_part(tf.ones((size,size)),-1,0)
    return mask


def scaled_product_attention(q,k,v,mask):
    """
    计算注意力权重
    q,k,v必须具有匹配的前置维度
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和
    :param q:请求的形状==(...,seq_len_q,depth)
    :param k:主键的形状==(...,seq_len_k,depth)
    :param v:数值的形状==(...,seq_len_v,depth_v)
    :param mask:张量，其形状能转化为(...,seq_len_q,seq_len_k),默认为None
    :return:注意力权重
    """
    tf.matmul(q,k,transpose_b=True)#(...,seq_len_q,seq_len_k)
    #缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)
    #将mask加入到缩放的张量上
    if mask is not None:
        scaled_attention_logits+=(mask*-1e9)
    #softmax在最后一个轴（seq_len_k）上归一化，因此分数相加等于1
    attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)
    output = tf.matmul(attention_weights,v)
    return output,attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

#前馈网络由两层全连接层组成，两层之间有一个ReLU激活函数
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = model.point_wise_feed_forward_network(d_model,dff)

        self.layernorma1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorma2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self,x,training,mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,self.d_model)
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # 将嵌入和位置编码相加。
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)

#解码器层
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

#解码器（Decoder）
class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,maximum_position_encoding,rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights={}
        x = self.embedding(x)#(batch_size,target_seq_len,d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x +=self.pos_encoding[:,:seq_len,:]
        x = self.dropout(x,training=training)

        for i in range(self.num_layers):
            x,block1,block2 = self.dec_layers[i](x,enc_output,training,look_ahead_mask,padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        #x.shape==(batch_size,target_seq-len,d_model)
        return x,attention_weights


#多头注意力机制
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model%self.num_heads==0

        self.depth = d_model//self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self,x,batch_size):
        """
        分拆最后一个维度到(num_heads, depth)转置结果使得形状为(batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3])

    def call(self,v,k,q,mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)# (batch_size, seq_len, d_model)
        k = self.wk(k)# (batch_size, seq_len, d_model)
        v = self.wv(v)# (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention,attention_weights = model.scaled_product_attention(q,k,v,mask)
        scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])# (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model))#(batch_size,seq_len_q,d_model)

        output = self.dense(concat_attention)#(batch_size,seq_len_q,d_model)
        return output,attention_weights

#Transformer包括编码器，解码器和最后的线性层，解码器的输出是线性层的输入，返回线性层的输出
class Transformer(tf.keras.Model):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,pe_input,pe_target,rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers,d_model,num_heads,dff,input_vocab_size,pe_input,rate)
        self.decoder = Decoder(num_layers,d_model,num_heads,dff,target_vocab_size,pe_target,rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self,inp,tar,training,enc_padding_mask,look_ahead_mask,dec_padding_mask):
        enc_output = self.encoder(inp,training,enc_padding_mask)#(batch_size,inp_seq_len,d_model)
        dec_output,attention_weights  = self.decoder(tar,enc_output,training,look_ahead_mask,dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output,attention_weights
