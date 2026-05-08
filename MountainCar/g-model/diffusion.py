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

        self.betas = torch.tensor(self.betas).to(torch.float32).to(self.device)
        self.alphas = torch.tensor(self.alphas).to(torch.float32).to(self.device)

        self.list_bar_alphas = [self.alphas[0]]
        for t in range(1,self.num_diffusion_step):
            self.list_bar_alphas.append(reduce(mul,self.alphas[:t]))
            
        self.list_bar_alphas = torch.cumprod(self.alphas, axis=0).to(torch.float32).to(self.device)
        self.criterion = nn.MSELoss(reduction='none')
        self.denoising_model = Denoising(self.data_size, self.num_diffusion_step).to(self.device)
        self.denoising_model.emb = self.denoising_model.emb.to(self.device)
        self.optimizer = optim.AdamW(self.denoising_model.parameters(), lr=3e-4)

    def train(self, training_data, metrics, method):
        """
        Modified train function to implement weighted loss based on guidance metrics.
        """
        indices = [i for i in range(len(training_data))]
        
        lambda_guide = 0.1

        for epoch in range(self.epoch):
            running_loss = 0.0
            Ts = np.random.randint(1,self.num_diffusion_step, size=self.training_step_per_spoch)
            for _, t in enumerate(Ts):
                # Sample data
                index = np.random.choice(indices)
                x_init = training_data[index]
                x_init = torch.from_numpy(x_init).to(torch.float32).to(self.device)
                
                # Diffusion process
                q_t = self.q_sample(x_init, t, self.list_bar_alphas, self.device)
                        
                # Posterior mean (Target)
                mu_t, cov_t = self.posterior_q(x_init, q_t, t, self.alphas, self.list_bar_alphas, self.device)
                
                self.optimizer.zero_grad()
        
                mu_theta = self.denoising_model(q_t , t)
                
                # Calculate raw MSE per sample
                loss_mse = self.criterion(mu_t, mu_theta).mean() 

                if method == 'generative':
                    loss = loss_mse
                else:
                    # Retrieve guidance metric for this sample
                    guidance_val = metrics[index]
                    guidance_tensor = torch.tensor(guidance_val, dtype=torch.float32, device=self.device)
                    
                    # Weighting Logic:
                    # Weight = 1 + lambda * metric (Promote high novelty/metric)
                    # When guidance is high (novel), the loss weight increases, forcing the model to learn it.
                    weight = 1.0 + lambda_guide * guidance_tensor
                    
                    loss = loss_mse * weight

                loss.backward()
                self.optimizer.step()
                running_loss += loss.detach()

    def q_sample(self, x_start, t, list_bar_alphas, device):
        alpha_bar_t = list_bar_alphas[t]
        mean = alpha_bar_t*x_start
        cov = torch.eye(x_start.shape[0]).to(device)
        cov = cov*(1-alpha_bar_t)
        return torch.distributions.MultivariateNormal(loc=mean,covariance_matrix=cov).sample().to(device)

    def denoise_with_mu(self, denoise_model, x_t, t, list_alpha, list_alpha_bar, DATA_SIZE, device):
        alpha_t = list_alpha[t]
        beta_t = 1 - alpha_t
        
        mu_theta = denoise_model(x_t,t)
        
        if t > 0:
            noise = torch.randn_like(x_t).to(device)
            sigma_t = torch.sqrt(beta_t)
            x_t_before = mu_theta + sigma_t * noise
        else:
            x_t_before = mu_theta
            
        return x_t_before


    def posterior_q(self, x_start, x_t, t, list_alpha, list_alpha_bar, device):
        beta_t = 1 - list_alpha[t]
        alpha_t = list_alpha[t]
        alpha_bar_t = list_alpha_bar[t]
        
        if t == 0:
            alpha_bar_t_before = torch.tensor(1.0).to(device)
        else:
            alpha_bar_t_before = list_alpha_bar[t-1]
        
        first_term = x_start * torch.sqrt(alpha_bar_t_before) * beta_t / (1 - alpha_bar_t)
        second_term = x_t * torch.sqrt(alpha_t)*(1- alpha_bar_t_before)/ (1 - alpha_bar_t)
        mu_tilde = first_term + second_term
        
        cov = torch.eye(x_start.shape[0]).to(device)*(1-alpha_bar_t)
        
        return mu_tilde, cov

    def clip_samples(self, sample):
        sample = sample.detach().cpu().numpy()
        
        # Ensure correct shape (Batch, Dim)
        if sample.ndim == 1:
            sample = sample[None, :]
            
        if sample.shape[1] == 2:
            sample[:, 0] = np.clip(sample[:, 0], -0.6, -0.4)
            sample[:, 1] = np.clip(sample[:, 1], 0, 0)
        else:
            sample = np.clip(sample, -1, 1)

        return sample

    def generate(self):
        with torch.no_grad():
            data = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.data_size).to(self.device),
                covariance_matrix=torch.eye(self.data_size).to(self.device)
            ).sample().to(self.device)
            
            if data.dim() == 1:
                data = data.unsqueeze(0)

            for t in range(0,self.num_diffusion_step):
                data = self.denoise_with_mu(self.denoising_model,data,self.num_diffusion_step-t-1, self.alphas, self.list_bar_alphas, self.data_size, self.device)

        return self.clip_samples(data)

    def load_model(self, model_path):
        self.denoise_model = torch.load(model_path)
        self.denoising_model.eval()