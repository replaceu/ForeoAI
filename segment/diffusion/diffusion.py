from typing import Tuple, Optional
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import gather

class DenoiseDiffusion:
    """
    Denoise Diffusion
    """
    def __init__(self,eps_model:nn.Module,n_steps:int,device:torch.device):
        """
        :param eps_model: UNet去噪模型
        :param n_steps: 训练总步数T
        :param device: GPU/CPU
        """
        super().__init__()
        #定义UNet架构模型
        self.eps_model = eps_model
        #人为设置超参数beta,满足beta随着t的增大而增大
        self.beta = torch.linspace(0.0001,0.02,n_steps).to(device)
        #根据beta计算alpha
        self.alpha = 1-self.beta
        #根据alpha计算alpha_bar
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        #定义训练总步长
        self.n_steps = n_steps
        #sampling中的sigma_t
        self.sigma_t = self.beta

    def q_xt_x0(self,x0:torch.Tensor,t:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        """
        Diffusion Process的中间步骤，根据x0和t，推导出xt所服从的高斯分布的mean和var
        :param x0:来自训练数据干净的图片
        :param t:某一步time_step
        :return:mean:xt所服从的高斯分布的均值
                var：xt所服从的高斯分布的方差
        """

        # ----------------------------------------------------------------
        # gather：人为定义的函数，从一连串超参中取出当前t对应的超参alpha_bar
        # 由于xt = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
        # 其中epsilon~N(0, I)
        # 因此根据高斯分布性质，xt~N(sqrt(alpha_bar_t) * x0, 1-alpha_bar_t)
        # 即为本步中我们要求的mean和var
        # ----------------------------------------------------------------

        mean = gather(self.alpha_bar,t)**0.5*x0
        var = 1-gather(self.alpha_bar,t)

        return mean,var

    def q_sample(self,x0:torch.Tensor,t:torch.Tensor,eps:Optional[torch.Tensor]=None):
        """
        Diffusion Process:根据xt所服从的高斯分布的mean和var，求出xt
        :param x0: 来自训练数据的干净的图片
        :param t: 某一步time_step
        :return: xt:第t时刻加完噪声的图片
        """

        # ----------------------------------------------------------------
        # xt = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
        #    = mean + sqrt(var) * epsilon
        # 其中，epsilon~N(0, I)
        # ----------------------------------------------------------------

        if eps is None:
            eps = torch.randn_like(x0)
        mean,var = self.q_xt_x0(x0,t)
        return mean+(var**0.5)*eps

    def p_sample(self,xt:torch.Tensor,t:torch.Tensor):
        """
        smpling:当模型训练好之后，根据x_t和t,推出x_{t-1}
        :param xt: t时刻的图片
        :param t: 某一步time_step
        :return: x_{t-1}:第t-1时刻的图片
        """

        #eps_mode:训练好的UNet去噪模型
        #eps_theta：用训练好的UNet去噪模型，预测第t步的噪声
        eps_theta = self.eps_model(xt,t)

        # 根据Sampling提供的公式，推导出x_{t-1}
        alpha_bar = gather(self.alpha_bar,t)
        alpha =gather(self.alpha,t)
        eps_coef = (1-alpha)/(1-alpha_bar)**0.5
        mean = 1/(alpha**0.5)*(x-eps_coef*eps_theta)
        var = gather(self.sigma_t,t)
        eps = torch.randn(xt.shape,device=xt.device)

        return mean+(var**0.5)*eps


    def loss(self,x0:torch.Tensor,noise:Optional[torch.Tensor]=None):
        """
        1. 随机抽取一个time_step t
        2. 执行diffusion process(q_sample)，随机生成噪声epsilon~N(0, I)，然后根据x0, t和epsilon计算xt
        3. 使用UNet去噪模型（p_sample），根据xt和t得到预测噪声epsilon_theta
        4. 计算mse_loss(epsilon, epsilon_theta)

        【MSE只是众多可选loss设计中的一种，大家也可以自行设计loss函数】

        Params:
         x0：来自训练数据的干净的图片
         noise: diffusion process中随机抽样的噪声epsilon~N(0, I)
        Return:
         loss: 真实噪声和预测噪声之间的loss
        """
        batch_size = x0.shape[0]
        #随机抽样t
        t = torch.randint(0,self.n_steps,(batch_size,),device=x0.device,dtype=torch.long)
        #如果传入为噪声，则从N(0,1)中抽样噪声
        if noise is None:
            noise = torch.randn_like(x0)
        #执行Diffusion process计算xt
        xt = self.q_sample(x0,t,eps=noise)
        #执行Denoise Process,得到预测的epsilon_theta
        eps_theta = self.eps_model(xt,t)

        #返回真实噪声和预测噪声之间的mse_loss
        return F.mse_loss(noise,eps_theta)






