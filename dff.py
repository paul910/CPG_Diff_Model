import torch
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

from dataloader import get_adj_dataloader
from unet import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
MODEL_DEPTH = 4
LEARNING_RATE = 1e-4
T = 1000
EPOCHS = 1000
DATA_PATH = 'data/reveal'

LOAD_MODEL = False
MODEL_PATH = 'model/model.pth'


dataloader = get_adj_dataloader(DATA_PATH, BATCH_SIZE, MODEL_DEPTH)
model = Unet(MODEL_DEPTH).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

if LOAD_MODEL:
    model.load_state_dict(torch.load(MODEL_PATH))


def geometric_beta_schedule(timesteps, start=0.0001, end=0.02):
    decay_rate = (end / start) ** (1.0 / (timesteps - 1))
    return torch.tensor([start * decay_rate ** i for i in range(timesteps)])


betas = geometric_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def show_forward_diffusion():
    adj = next(iter(dataloader))
    plt.figure(figsize=(30,30))
    plt.axis('off')
    num_show = 15
    stepsize = int(T / num_show)

    for idx in range(0, T, stepsize):
        t = torch.tensor([idx]).type(torch.int64)
        plt.subplot(1, num_show + 1, int(idx / stepsize) + 1)
        adj, noise = forward_diffusion_sample(adj, t)
        show_tensor_image(adj[0])

    plt.show()


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


def show_tensor_image(adj: Tensor):
    adj = adj.squeeze().numpy()
    plt.imshow(adj, cmap='gray')


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(num_nodes):
    # Sample noise
    num_nodes = num_nodes
    adj = torch.randn((1, 1, num_nodes, num_nodes), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_show = 10
    stepsize = int(T / num_show)

    for i in reversed(range(T)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        adj = sample_timestep(adj, t)
        # Edit: This is to maintain the natural range of the distribution
        adj = torch.clamp(adj, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_show, int(i / stepsize) + 1)
            show_tensor_image(adj.detach().cpu())
    plt.show()


for epoch in range(EPOCHS):
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch.shape[0] != BATCH_SIZE:
            continue

        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch, t)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            torch.save(model.state_dict(), MODEL_PATH)
            sample_plot_image(32)
