import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

from config import Config


class DiffusionUtils:
    def __init__(self, config: Config):
        self.config = config
        self.betas = self.geometric_beta_schedule(timesteps=self.config.T)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    @staticmethod
    def geometric_beta_schedule(timesteps, start=0.0001, end=0.02):
        decay_rate = (end / start) ** (1.0 / (timesteps - 1))
        return torch.tensor([start * decay_rate ** i for i in range(timesteps)])

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        vals = vals.to(self.config.DEVICE)
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t.to(self.config.DEVICE) * x_0.to(self.config.DEVICE) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.config.DEVICE) * noise.to(self.config.DEVICE), noise.to(self.config.DEVICE)

    @torch.no_grad()
    def sample_timestep(self, x, t, model):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def show_forward_diffusion(self, adj: Tensor):
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        num_disp = 10
        stepsize = int(self.config.T / num_disp)

        for idx in range(0, self.config.T, stepsize):
            t = torch.tensor([idx]).type(torch.int64)
            plt.subplot(1, num_disp + 1, int(idx / stepsize) + 1)
            adj, noise = self.forward_diffusion_sample(adj, t)
            show_tensor_image(adj[0])

        plt.show()


def show_tensor_image(adj: Tensor):
    adj = adj.squeeze().numpy()
    plt.imshow(adj, cmap='gray')
