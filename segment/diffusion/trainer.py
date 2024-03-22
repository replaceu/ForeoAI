
import torchvision
from PIL import Image
from typing import List

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from diffusion import DenoiseDiffusion
from diffusion.unet import Unet

class configs(BaseConfigs):
    def train(self):
        """
        单epoch训练DDPM
        """
        # 遍历每一个batch
        for data in monit.iterate('train', self.data_loader):
            # step数加1（tracker是自定义类）
            tracker.add_global_step()
            # 将这个batch的数据移动到GPU/CPU上
            data = data.to(self.device)

            # 每个batch开始时，梯度清0
            self.optimizer.zero_grad()
            # self.diffusion即为DenoiseModel实例，执行forward，计算loss
            loss = self.diffusion.loss(data)
            # 计算梯度
            loss.backward()
            # 更新
            self.optimizer.step()
            # 保存loss,用于后续可视化之类的操作
            tracker.save('loss', loss)

    def sample(self):
        """
        利用当前模型，将一张随机高斯噪声（xt）逐步还原回x0,x0将用于评估模型效果（例如ID分数）
        :return:
        """


