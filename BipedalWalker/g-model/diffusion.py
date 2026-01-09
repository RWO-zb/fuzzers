import torch
import numpy as np
from operator import mul
from functools import reduce
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

seed = random.randint(1,1000)
torch.manual_seed(seed)

def position_encoding_init(n_position, d_pos_vec):
    ''' 
    Init the sinusoid position encoding table 
    '''
    # keep dim 0 for padding token position encoding zero vector
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

        # send parameters to device
        self.betas = torch.tensor(self.betas).to(torch.float32).to(self.device)
        self.alphas = torch.tensor(self.alphas).to(torch.float32).to(self.device)

        # alpha_bar_t is the product of all alpha_ts from 0 to t
        self.list_bar_alphas = [self.alphas[0]]
        for t in range(1,self.num_diffusion_step):
            self.list_bar_alphas.append(reduce(mul,self.alphas[:t]))
            
        self.list_bar_alphas = torch.cumprod(self.alphas, axis=0).to(torch.float32).to(self.device)

        # Modified: Use reduction='none' for weighted loss
        self.criterion = nn.MSELoss(reduction='none')
        self.denoising_model = Denoising(self.data_size, self.num_diffusion_step).to(self.device)
        # disgusting hack to put embedding layer on 'device' as well, as it is not a pytorch module!
        self.denoising_model.emb = self.denoising_model.emb.to(self.device)
        self.optimizer = optim.AdamW(self.denoising_model.parameters(), lr=3e-4)


    def train(self, training_data, metrics, method):
        """
        Modified train function to implement weighted loss based on guidance metrics.
        """
        indices = [i for i in range(len(training_data))]
        
        # Hyperparameter lambda from the paper
        lambda_guide = 0.1

        for epoch in range(self.epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            # sample a bunch of timesteps
            Ts = np.random.randint(1,self.num_diffusion_step, size=self.training_step_per_spoch)
            for _, t in enumerate(Ts):
                # produce corrupted sample
                index = np.random.choice(indices)
                x_init = training_data[index]
                x_init = torch.from_numpy(x_init).to(torch.float32).to(self.device)
                
                # Fix: reshape x_init if necessary to ensure it works with batch processing logic if extended
                # In current code x_init is likely 1D array from numpy, converting to [dim] tensor.
                # If training_data is list of arrays, this is fine. 

                q_t = self.q_sample(x_init, t, self.list_bar_alphas, self.device)
                        
                # calculate the mean and variance of the posterior forward distribution q(x_t-1 | x_t,x_0)
                mu_t, cov_t = self.posterior_q(x_init, q_t, t, self.alphas, self.list_bar_alphas, self.device)
                # get just first element from diagonal of covariance since they are all equal
                sigma_t = cov_t[0][0]
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                mu_theta = self.denoising_model(q_t , t)
                
                # --- Weighted Loss Implementation ---
                # Calculate raw MSE per element, then mean over dimensions [batch_size, dim] -> [batch_size]
                # Here batch_size is effectively 1, so x_init shape is [dim] or [1, dim]
                
                # If x_init is 1D [dim], loss_mse will be scalar after mean. 
                # If we want to support batching later, keep dimensions in mind. 
                # Current code structure implies processing one sample at a time effectively in the inner loop (index chosen singly).
                
                loss_mse = self.criterion(mu_t, mu_theta) # Shape: same as mu_t
                loss_mse = loss_mse.mean() # Scalar loss for this sample
                
                if method == 'generative':
                    loss = loss_mse
                else:
                    # Retrieve guidance metric for this sample
                    guidance_val = metrics[index] 
                    # guidance_val should be a scalar or single-element array here
                    guidance_tensor = torch.tensor(guidance_val, dtype=torch.float32, device=self.device)
                    
                    # Weight = 1 + lambda * metric (Promote high novelty/metric)
                    weight = 1.0 + lambda_guide * guidance_tensor
                    
                    loss = loss_mse * weight

                loss.backward()
                self.optimizer.step()
                running_loss += loss.detach()

            # Optional: Print loss
            # print('running_loss:\t',running_loss)


    def q_sample(self, x_start, t, list_bar_alphas, device):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        alpha_bar_t = list_bar_alphas[t]
        
        mean = alpha_bar_t*x_start
        # Fix: ensure identity matrix size matches input dimension correctly
        dim = x_start.shape[0] if len(x_start.shape) == 1 else x_start.shape[1]
        cov = torch.eye(dim).to(device)
        cov = cov*(1-alpha_bar_t)
        return torch.distributions.MultivariateNormal(loc=mean,covariance_matrix=cov).sample().to(device)


    def denoise_with_mu(self, denoise_model, x_t, t, list_alpha, list_alpha_bar, DATA_SIZE, device):
        """
        Denoising function considering the denoising models tries to model the posterior mean
        """
        alpha_t = list_alpha[t]
        beta_t = 1 - alpha_t
        
        mu_theta = denoise_model(x_t,t)
        
        # Add noise if t > 0
        if t > 0:
            noise = torch.randn_like(x_t).to(device)
            sigma_t = torch.sqrt(beta_t)
            x_t_before = mu_theta + sigma_t * noise
        else:
            x_t_before = mu_theta
            
        return x_t_before


    def posterior_q(self, x_start, x_t, t, list_alpha, list_alpha_bar, device):
        """
        calculate the parameters of the posterior distribution of q
        """
        beta_t = 1 - list_alpha[t]
        alpha_t = list_alpha[t]
        alpha_bar_t = list_alpha_bar[t]
        
        # alpha_bar_{t-1}
        if t == 0:
            alpha_bar_t_before = torch.tensor(1.0).to(device)
        else:
            alpha_bar_t_before = list_alpha_bar[t-1]
        
        # calculate mu_tilde
        first_term = x_start * torch.sqrt(alpha_bar_t_before) * beta_t / (1 - alpha_bar_t)
        second_term = x_t * torch.sqrt(alpha_t)*(1- alpha_bar_t_before)/ (1 - alpha_bar_t)
        mu_tilde = first_term + second_term
        
        dim = x_start.shape[0] if len(x_start.shape) == 1 else x_start.shape[1]
        cov = torch.eye(dim).to(device)*(1-alpha_bar_t)
        
        return mu_tilde, cov

    def clip_samples(self, sample):
        # FIX: Added .detach() to break gradient graph before converting to numpy
        sample = sample.detach().cpu().numpy()
        sample = np.clip(sample, 1, 3)
        sample = sample.astype(int)
        return sample

    def generate(self):
        # Optimization: Use torch.no_grad() for inference to save memory and avoid tracking gradients
        with torch.no_grad():
            data = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.data_size).to(self.device),
                covariance_matrix=torch.eye(self.data_size).to(self.device)
            ).sample().to(self.device)
            
            for t in range(0,self.num_diffusion_step):
                data = self.denoise_with_mu(self.denoising_model,data,self.num_diffusion_step-t-1, self.alphas, self.list_bar_alphas, self.data_size, self.device)

        return self.clip_samples(data) # Return 1D array is handled by clip_samples return shape if input is 1D

    def load_model(self, model_path):
        self.denoising_model = torch.load(model_path)
        self.denoising_model.eval()