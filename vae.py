import torch
from torch import nn

# 输入层维度
input_dim = 784
# 过渡层维度
inter_dim = 256
# 隐变量维度
latent_dim = 2


class VAE(nn.Module):
    def __int__(self, D_input=input_dim, D_inter=inter_dim, D_latent=latent_dim):
        super(VAE, self).__init__()

        # 自定义encoder网络层
        self.encoder = nn.Sequential
        (
            nn.Linear(D_input, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2)
        )

        # 自定义decoder网络层
        self.encoder = nn.Sequential
        (
            nn.Linear(D_input, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2),
            nn.Sigmoid
        )

        '''
        重构函数
        输入：均值(μ)，log(σ^2)
        输出：重参数化后隐空间上的z
        备注：使用的log_var是方差的对数，即2*lg(σ)。我们取的z是符合标准高斯分布的，为得到sample z，
        我们通过以下变换得到sample z：
            (N(μ,σ)-μ)/σ=N(0,I) ==> N(μ,σ)=σ*N(0,I)+μ
        '''
        def reparameterise(mu, log_var):
            # exp(log(σ^2) * 0.5) = σ
            std = log_var.mul(0.5).exp()
            # 返回一个与输入相同大小的张量，该张量由均值为0、方差为1的正态分布的随机数填充
            esp = torch.randn_like(*mu.size())
            # N(μ, σ) = σ * N(0, I) + μ
            z = mu + std * esp

            return z
        
        '''
        向前传递函数
        输入：图片
        输出：重构生成的图像，输入张量经encoder编码以后生成的均值和方差
        '''
        def forward(self, x):
            org_size = x.size()
            batch = org_size[0]
            x = x.view(batch, -1)

            h = self.encoder(x)
            # 将均值和方差分开
            mu, log_var = h.chunk(2, dim=1)
            z = self.reparameterise(mu, log_var)
            recon_x = self.decoder(z).view(size=org_size)

            return recon_x, mu, log_var


'''
返回图像函数
输入：隐空间取得的张量（1x784）
输出：图像张量（28*28)
'''
def z_to_img(x):
    # 数据标准化
    x = x.clamp(0, 1)
    x = x.view(x.size[0], 1, 28, 28)
    return x

# 设置重构函数
rec_fnction = nn.MSELoss(size_average=False)
'''
损失函数
输入：生成图像张量，原始图像张量，均值，方差
输出：loss
'''
def loss_function(recon_x, x, mu, logvar):
    # BCE的loss，BCE是重构误差
    BCE = rec_fnction(recon_x, x)
    # 计算KL_Divergency
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # loss是BCE和KLD之和
    loss = BCE + KLD

    return loss, BCE, KLD