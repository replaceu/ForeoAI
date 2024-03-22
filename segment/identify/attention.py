import numpy as np
import torch
from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

class MultiHeadedAttention(nn.Module):
    def __init__(self,num_heads:int,d_model:int,dropout:float=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % num_heads==0

        self.k_dim = d_model//num_heads
        self.num_heads = num_heads
        #W^Q,W^K,W^V,W^O
        self.proj_weights = clones(nn.Linear(d_model,d_model),4)
        self.attention_score = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query:Tensor,key:Tensor,value:Tensor,mask:Optional[Tensor] = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        #1.Apply W^Q, W^K, W^V to generate new query, key, value
        query,key,value = [proj_weight(x).view(batch_size,-1,self.num_heads,self.k_dim).transpose(1,2)
        for proj_weight,x in zip(self.proj_weights,[query,key,value])]

        # 2 Calculate attention score and the out
        out,self.attention_score = attention(query,key,value,mask=mask,dropout=self.dropout)

        #3.concat output
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.k_dim)

        #4.apply w^o to get final output
        out = self.proj_weights[-1](out)

        return out

def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])