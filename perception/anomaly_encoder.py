from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from pipeline.config import ModelConfig
@dataclass
class AnomalyOutput:
    score: float
    features: torch.Tensor
    reconstruction: torch.Tensor
    is_anomaly: bool
class AnomalyEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_extractor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(2048, self.config.latent_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(self.config.latent_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(self.config.latent_dim, 256 * 8 * 8), nn.ReLU(inplace=True), nn.Unflatten(1, (256, 8, 8)), nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid())
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features_2d = self.backbone(x)
        features_1d = self.feature_extractor(features_2d)
        return features_1d
    def reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)
    def anomaly_score(self, x: torch.Tensor, reconstruction: torch.Tensor, class_prob: torch.Tensor) -> float:
        mse_error = F.mse_loss(reconstruction, x, reduction='mean').item()
        prob = class_prob.mean().item()
        score = 0.5 * prob + 0.5 * min(mse_error * 10.0, 1.0)
        return float(min(max(score, 0.0), 1.0))
    def forward(self, x: torch.Tensor) -> AnomalyOutput:
        features = self.encode(x)
        class_prob = self.classifier(features)
        reconstruction = self.reconstruct(features)
        score_val = self.anomaly_score(x, reconstruction, class_prob)
        is_anomaly = score_val > self.config.anomaly_threshold
        return AnomalyOutput(score=score_val, features=features, reconstruction=reconstruction, is_anomaly=is_anomaly)
    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        class_prob = self.classifier(features)
        reconstruction = self.reconstruct(features)
        bce_loss = F.binary_cross_entropy(class_prob, labels.float())
        mse_loss = F.mse_loss(reconstruction, x)
        return bce_loss + mse_loss