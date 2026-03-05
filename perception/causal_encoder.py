from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pipeline.config import ModelConfig
@dataclass
class CausalEncoderOutput:
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    reconstruction: torch.Tensor
class CausalEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder_conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Flatten())
        self.fc_mu = nn.Linear(512 * 8 * 8, self.config.latent_dim)
        self.fc_log_var = nn.Linear(512 * 8 * 8, self.config.latent_dim)
        self.decoder_input = nn.Linear(self.config.latent_dim, 512 * 8 * 8)
        self.decoder_conv = nn.Sequential(nn.Unflatten(1, (512, 8, 8)), nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid())
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_conv(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return (mu, log_var)
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z)
        reconstruction = self.decoder_conv(hidden)
        return reconstruction
    def intervene(self, z: torch.Tensor, dim: int, value: float) -> torch.Tensor:
        z_intervened = z.clone()
        z_intervened[:, dim] = value
        return z_intervened
    def forward(self, x: torch.Tensor) -> CausalEncoderOutput:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return CausalEncoderOutput(z=z, mu=mu, log_var=log_var, reconstruction=reconstruction)
    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(x)
        recon_loss = F.mse_loss(output.reconstruction, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp()) / x.size(0)
        total_loss = recon_loss + self.config.beta * kl_div
        return (total_loss, recon_loss, kl_div)