import torch
import numpy as np
from operator import mul
from functools import reduce
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from scipy.special import softmax

seed = random.randint(1,1000)
torch.manual_seed(seed)

def position_encoding_init(n_position, d_pos_vec):
    ''' 
    Init the sinusoid position encoding table 
    '''
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).to(torch.float32)

class Denoising(torch.nn.Module):

    def __init__(self, x_dim, num_diffusion_timesteps):
        super(Denoising, self).__init__()

        self.linear1 = torch.nn.Linear(x_dim, 256)
        self.emb = position_encoding_init(num_diffusion_timesteps,x_dim)
        self.linear2 = torch.nn.Linear(256, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, x_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x_input, t):
        emb_t = self.emb[t]
        # x_input shape: (Batch, x_dim)
        # emb_t shape: (x_dim,) -> broadcast to (Batch, x_dim)
        x = self.linear1(x_input+emb_t)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x


class Diffusion:
    def __init__(self, 
    batch_size = 1, 
    epoch = 10000, 
    data_size = 128 , 
    training_step_per_spoch = 100, 
    num_diffusion_step = 100
    ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.data_size = data_size
        self.training_step_per_spoch = training_step_per_spoch
        self.num_diffusion_step = num_diffusion_step

    def setup(self):
        self.beta_start = .0004
        self.beta_end = .02
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.betas = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_diffusion_step) ** 2
        self.alphas = 1 - self.betas

        # send parameters to device
        self.betas = torch.tensor(self.betas).to(torch.float32).to(self.device)
        self.alphas = torch.tensor(self.alphas).to(torch.float32).to(self.device)

        # alpha_bar_t is the product of all alpha_ts from 0 to t
        self.list_bar_alphas = [self.alphas[0]]
        for t in range(1,self.num_diffusion_step):
            self.list_bar_alphas.append(reduce(mul,self.alphas[:t]))
            
        self.list_bar_alphas = torch.cumprod(self.alphas, axis=0).to(torch.float32).to(self.device)

        self.criterion = nn.MSELoss()
        self.denoising_model = Denoising(self.data_size, self.num_diffusion_step).to(self.device)
        self.denoising_model.emb = self.denoising_model.emb.to(self.device)
        self.optimizer = optim.AdamW(self.denoising_model.parameters(), lr=3e-4)

    def train(self,training_data, metrics, method):

        indices = [i for i in range(len(training_data))]
        for epoch in range(self.epoch):
            running_loss = 0.0
            Ts = np.random.randint(1,self.num_diffusion_step, size=self.training_step_per_spoch)
            for _, t in enumerate(Ts):
                index = np.random.choice(indices)
                x_init = training_data[index]
                x_init = torch.from_numpy(x_init).to(torch.float32).to(self.device)
                q_t = self.q_sample(x_init, t, self.list_bar_alphas, self.device)
                        
                mu_t, cov_t = self.posterior_q(x_init, q_t, t, self.alphas, self.list_bar_alphas, self.device)
                sigma_t = cov_t[0][0]
                self.optimizer.zero_grad()
        
                mu_theta = self.denoising_model(q_t , t)
                loss1 = self.criterion(mu_t, mu_theta)
                loss1.backward()
                running_loss += loss1.detach()

                # add other guidances to loss
                if method != 'generative':
                    guidances = metrics[index]
                    guidances = torch.from_numpy(guidances).to(torch.float32).to(self.device)
                    loss2 = self.criterion(guidances, guidances * 0)
                    loss2.requires_grad = True
                    loss2.backward()
                    running_loss += loss2.detach()

                self.optimizer.step()

    def q_sample(self, x_start, t, list_bar_alphas, device):
        alpha_bar_t = list_bar_alphas[t]
        mean = alpha_bar_t*x_start
        cov = torch.eye(x_start.shape[0]).to(device)
        cov = cov*(1-alpha_bar_t)
        return torch.distributions.MultivariateNormal(loc=mean,covariance_matrix=cov).sample().to(device)

    def denoise_with_mu(self, denoise_model, x_t, t, list_alpha, list_alpha_bar, DATA_SIZE, device):
        alpha_t = list_alpha[t]
        beta_t = 1 - alpha_t
        alpha_bar_t = list_alpha_bar[t]
        
        mu_theta = denoise_model(x_t,t)
        
        # Sample from posterior mean
        x_t_before = torch.distributions.MultivariateNormal(loc=mu_theta,covariance_matrix=torch.diag(beta_t.repeat(self.data_size))).sample().to(device)
            
        return x_t_before


    def posterior_q(self, x_start, x_t, t, list_alpha, list_alpha_bar, device):
        beta_t = 1 - list_alpha[t]
        alpha_t = list_alpha[t]
        alpha_bar_t = list_alpha_bar[t]
        alpha_bar_t_before = list_alpha_bar[t-1]
        
        first_term = x_start * torch.sqrt(alpha_bar_t_before) * beta_t / (1 - alpha_bar_t)
        second_term = x_t * torch.sqrt(alpha_t)*(1- alpha_bar_t_before)/ (1 - alpha_bar_t)
        mu_tilde = first_term + second_term
        
        cov = torch.eye(x_start.shape[0]).to(device)*(1-alpha_bar_t)
        
        return mu_tilde, cov

    def clip_samples(self, sample):
        sample = sample.cpu().numpy()
        
        # --- 修复关键点：确保 sample 至少是 2D (Batch, Dim) ---
        # 如果生成的是单个样本 (Dim,)，则扩展为 (1, Dim)
        if sample.ndim == 1:
            sample = sample[None, :]
            
        # MountainCar State: [position, velocity]
        # Position: [-1.2, 0.6], Velocity: [-0.07, 0.07]
        if sample.shape[1] == 2:
            sample[:, 0] = np.clip(sample[:, 0], -0.6, -0.4)
            sample[:, 1] = np.clip(sample[:, 1],0, 0)
        else:
            # Fallback if dimensions change
            sample = np.clip(sample, -1, 1)

        return sample

    def generate(self):
        # 初始化噪声
        data = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.data_size),
            covariance_matrix=torch.eye(self.data_size)
        ).sample().to(self.device)
        
        # --- 修复关键点：确保初始噪声有 Batch 维度 ---
        if data.dim() == 1:
            data = data.unsqueeze(0) # (1, data_size)

        for t in range(0,self.num_diffusion_step):
            data = self.denoise_with_mu(self.denoising_model,data,self.num_diffusion_step-t-1, self.alphas, self.list_bar_alphas, self.data_size, self.device)

        return self.clip_samples(data)

    def load_model(self, model_path):
        self.denoise_model = torch.load(model_path)
        self.denoising_model.eval()