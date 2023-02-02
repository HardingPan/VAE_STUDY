import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
# 导入vae
import vae

# 设备配置
torch.cuda.set_device(0)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is ok')
else:
    device = torch.device('cpu')
    print('GPU is not ok, CPU is ok')

# 文件夹设置
sample_dir = 'vae_img'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# 参数设置
epochs = 100
batch_size = 128
learning_rate = 1e-3
h_dim = 400
z_dim = 20
image_size = 784

# 数据设置，使用的是MINIST数据集
dataset = torchvision.datasets.MNIST(root='../../../data/minist',
                                     train=True,
                                     transform=tfs.ToTensor(),
                                     download=True)

# 数据加载器
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)


# 实例化模型
model = vae.VAE().to(device)
print("VAE loading completed")

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (x, _) in enumerate(data_loader):
        # 获取样本，并前向传播
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)
        
        loss, BCE, KLD = vae.loss_function(x_reconst, x, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, epochs, i+1, len(data_loader), BCE.item(), KLD.item()))
    
    # 利用训练的模型进行测试
    with torch.no_grad():
        # 随机生成的图像
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # 重构的图像
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
