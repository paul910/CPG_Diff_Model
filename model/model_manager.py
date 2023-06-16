import matplotlib.pyplot as plt
import torch
import gc
import csv
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from config import Config
from model.model import Unet
from utils.utils import DiffusionUtils, show_tensor_image


class ModelManager:
    def __init__(self, config: Config, diffusion_utils: DiffusionUtils, data_loader):
        self.config = config
        self.diffusion_utils = diffusion_utils
        self.model = Unet(self.config.MODEL_DEPTH, self.config.MODEL_START_CHANNELS, self.config.TIME_EMB_DIM).to(self.config.DEVICE)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.data_loader = data_loader

        # Show forward diffusion
        if config.VISUALIZE:
            _, adj = next(enumerate(self.data_loader))
            diffusion_utils.show_forward_diffusion(adj[0])

        if self.config.LOAD_MODEL:
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE))

    def get_loss(self, x_0, t):
        x_noisy, noise = self.diffusion_utils.forward_diffusion_sample(x_0, t)
        noise_pred = self.model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def train_model(self):
        with open('loss_values.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'loss'])

        for epoch in range(self.config.EPOCHS):
            for step, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                if batch.shape[0] != self.config.BATCH_SIZE:
                    continue

                torch.cuda.empty_cache()
                gc.collect()

                self.optimizer.zero_grad()
                t = torch.randint(0, self.config.T, (self.config.BATCH_SIZE,), device=self.config.DEVICE).long()
                loss = self.get_loss(batch, t)
                loss.backward()
                self.optimizer.step()

                with open('loss_values.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, step, loss.item()])

                if step % 100 == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                    if self.config.VISUALIZE:
                        self.sample_plot_image(64)

    @torch.no_grad()
    def sample_plot_image(self, num_nodes):
        # Sample noise
        num_nodes = num_nodes
        adj = torch.randn((1, 1, num_nodes, num_nodes), device=self.config.DEVICE)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        num_show = 10
        stepsize = int(self.config.T / num_show)

        for i in reversed(range(self.config.T)):
            t = torch.full((1,), i, device=self.config.DEVICE, dtype=torch.long)
            adj = self.diffusion_utils.sample_timestep(adj, t, self.model)
            adj = torch.clamp(adj, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_show, int(i / stepsize) + 1)
                show_tensor_image(adj.detach().cpu())

        plt.show()
