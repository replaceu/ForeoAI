import math
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
from labml_helpers.module import Module


class Unet(Module):
    """
    DDPM Unet去噪模型主体架构
    """

    def __init__(self,image_channels:int=3,n_channels:int=64,
                 ch_mults:Union[Tuple[int,...],List[int]]=(1,2,2,4),
                 is_attn:Union[Tuple[bool,...],List[int]]=(False,False,True,True),n_blocks:int=2):
        """
        :param image_channels: 原始输入图片的channel数，对RGB图像来说等于3
        :param n_channels: 在进Unet之前，会对原始图片做一次初步卷积，该初步卷积对应的out_channel数
        :param ch_mults: 在Encoder下采样的每一层的out_channel倍数，例如ch_mults[i]=2,
                         表示第i层特征图的out_channel数是第i-1层的2倍。Decoder上采样时
                         也是同理，用的是反转后的ch_mults

        :param is_attn:在Encoder下采样/Decoder上采样的每一层，是否要在CNN做特征提取后再引入attention
        :param n_blocks:在Encoder下采样/Decoder下采样的每一层，需要用多少个DownBlock/UpBlock（见图），
                        Deocder层最终使用的UpBlock数=n_blocks+1
        """
        super().__init__()
        #在Encoder下采样/Decoder上采样的过程中，图像依次缩小/放大，每次变动都会产生一个新的图像分辨率
        #这里指的就是不同图像分辨率的个数，也可以理解成是Encoder/Decoder的层数
        n_resolutions=len(ch_mults)
        # 对原始图片做预处理，例如图中，将32*32*3 -> 32*32*64
        self.image_proj=nn.Conv2d(image_channels,n_channels,kernel_size=(3,3))
        # time_embedding，TimeEmbedding是nn.Module子类
        self.time_emb = TimeEmbedding(n_channels*4)
        # --------------------------
        # 定义Encoder部分
        # --------------------------
        # down列表中的每个元素表示Encoder的每一层
        down = []
        #初始化out_channel和in_channel
        out_channels = in_channels=n_channels
        #遍历每一层
        for i in range(n_resolutions):
            #根据设定好的规则，得到盖层的out_channel
            out_channels = in_channels*ch_mults[i]
            #根据设定好的规则，每一层有n_blocks个DownBlock
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels,out_channels,n_channels*4,is_attn[i]))
                in_channels = out_channels
            #对Encoder来说，每一层结束后，我们都做一次下采样，但Encoder的最后一层不做下采样
            if i<n_resolutions-1:
                down.append(Downsample(in_channels))

        # self.down即是完整的Encoder部分
        self.down = nn.ModuleList(down)

        # --------------------------
        # 定义Middle部分
        # --------------------------
        self.middle = MiddleBlock(out_channels,n_channels*4)

        # --------------------------
        # 定义Decoder部分
        # --------------------------
        # 和Encoder部分基本一致
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))

        # self.up即是完整的Decoder部分
        self.up = nn.ModuleList(up)
        #定义group_norm，激活函数，和最后一层的CNN（用于将Decoder最上一层的特征图还原成原始尺寸）
        self.norm = nn.GroupNorm(8,n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels,image_channels,kernel_size=(3,3),padding=(1,1))

    def forward(self,x:torch.Tensor,t:torch.Tensor):
        """
        :param x:输入数据xt,尺寸大小为（batch_size，in_channels,height,width）
        :param t: 输入数据t，尺寸大小为batch_size
        """
        #取得time_embedding
        t = self.time_emb(t)
        #对原始图片做初步CNN处理
        x = self.image_proj(x)
        # --------
        # Encoder
        # --------
        h = [x]
        for m in self.down:
            x = m(x,t)
            h.append(x)
        # -------
        # Middle
        # -------
        x = self.middle(x,t)
        # -------
        # Decoder
        # -------
        for m in self.up:
            if isinstance(m,Upsample):
                x = m(x,t)
            else:
                s = h.pop()
                x = torch.cat((x,s),dim=1)
                x = m(x,t)

        return self.final(self.act(self.norm(x)))


class ResidualBlock(Module):
    """
    每一个Residual block都有两层CNN做特征提取
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        :param in_channels: 输入图片的channel数量
        :param out_channels: 经过residual_block后输出特征图的channel数量
        :param time_channels: time_embedding的向量维度，例如t原来是个整型，值为1，表示时刻1，现在要将其变成维度为(1, time_channels)的向量
        :param n_groups: Group_Norm中的超惨
        :param dropout: dropout_rate
        """
        super().__init__()

        # 第一层卷积=GroupNorm+CNN
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # 第二层卷积 = Group Norm+CNN
        self.norm2 = nn.GroupNorm(n_groups, in_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # 当in_c = out_c时，残差连接直接将输入输出相加；
        # 当in_c != out_c时，对输入数据做一次卷积，将其通道数变成和out_c一致，再和输出相加
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # t向量的维度time_channels可能不等于out_c，所以我们要对起做一次线性转换
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: 输入数据xt,尺寸大小为（batch_size，in_channels,height,width）
        :param t: 输入数据t,尺寸大小为（batch_size,time_c）
        """

        # 1.输入数据先过一层卷积
        h = self.conv1(self.act1(self.norm1(x)))
        # 2. 对time_embedding向量，通过线性层使time_c变为out_c，再和输入数据的特征图相加
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # 3.过第二层卷积
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        # 4.返回残差连接后的结果
        return h + self.shortcut(x)


class AttentionBlock(Moudle):
    """
    Attention模块:和Transformer中的multi-head attention原理及实现方式一致
    """
    def __init__(self,n_channels:int,n_heads:int=1,d_k:int=None,n_groups:int=32):
        """
        :param n_channels: 等待做attention操作的特征图的channel数
        :param n_heads: attention头数
        :param d_k: 每一个attention头处理的向量维度
        :param n_groups: Group Norm超参数
        """
        super().__init__()
        #一般而言，d_k = n_channels//n_heads，需保证n_channels能被n_heads整除
        if d_k is None:
            d_k = n_channels
        # 定义Group Norm
        self.norm = nn.GroupNorm(n_groups, n_channels)
        #Multi-head attention层: 定义输入token分别和q,k,v矩阵相乘后的结果
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        #MLP层
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Params:
            x: 输入数据xt，尺寸大小为（batch_size, in_channels, height, width）
            t: 输入数据t，尺寸大小为（batch_size, time_c）
        """
        #t并没有用到，但是为了和ResidualBlock定义方式一致，这里也引入了t
        _ = t
        #获取shape
        batch_size, n_channels, height, width = x.shape
        #将输入数据的shape改为(batch_size, height*weight, n_channels)
        #这三个维度分别等同于transformer输入中的(batch_size, seq_length, token_embedding)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        #计算输入过矩阵q,k,v的结果，self.projection通过矩阵计算，一次性把这三个结果出出来
        #也就是qkv矩阵是三个结果的拼接
        #其shape为：(batch_size, height*weight, n_heads, 3 * d_k)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        #将拼接结果切开，每一个结果的shape为(batch_size, height*weight, n_heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        #以下是正常计算attention score的过程
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        #将结果reshape成(batch_size, height*weight, n_heads * d_k)
        #n_heads * d_k = n_channels
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # MLP层，输出结果shape为(batch_size, height*weight, n_channels)
        res = self.output(res)
        # 残差连接
        res += x

        # 将输出结果从序列形式还原成图像形式，
        # shape为(batch_size, n_channels, height, width)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(Module):
    """
    Down block，即Encoder中每一层的核心处理逻辑
    DownBlock = ResidualBlock + AttentionBlock
    在我们的例子中，Encoder的每一层都有2个DownBlock
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(Module):
    def __init__(self,in_channels:int,out_channels:int,time_channels:int,has_attn:bool):
        super().__init__()
        self.res = ResidualBlock(in_channels+out_channels,out_channels,time_channels)
        if has_attn:
            self.attn  =AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity
    def foward(self,x:torch.Tensor,t:torch.Tensor):
        x = self.res(x,t)
        x = self.attn(x)
        return x

class TimeEmbedding(nn.Module):
    """
    TimeEmbedding模块将把整型t，以Transformer函数式位置编码的方式，映射成向量，
    其shape为(batch_size, time_channel)
    """

    def __init__(self, n_channels: int):
        """
        Params:
            n_channels：即time_channel
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        Params:
            t: 维度（batch_size），整型时刻t
        """
        # 以下转换方法和Transformer的位置编码一致
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        # 输出维度(batch_size, time_channels)
        return emb

class Upsample(nn.Module):
    """
    上采样
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    下采样
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)

class MiddleBlock(Module):
    """
    MiddleBlock
    这是UNet结构中，连接Encoder和Decoder的最下层部分，
    MiddleBlock = ResidualBlock + AttentionBlock + ResidualBlock
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x

class Swish(Module):
    def forward(self,x):
        return x*torch.sigmoid(x)