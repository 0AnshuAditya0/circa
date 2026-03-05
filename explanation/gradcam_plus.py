import cv2
import numpy as np
import torch
from captum.attr import LayerGradCam
from perception.anomaly_encoder import AnomalyEncoder
class GradCAMPlusPlus:
    def __init__(self, model: AnomalyEncoder, target_layer: str):
        self.model = model
        resolved_layer = dict([*self.model.named_modules()]).get(target_layer)
        if resolved_layer is None:
            raise ValueError(f"Target layer '{target_layer}' not found within the AnomalyEncoder.")
        def forward_wrapper(inputs):
            features = self.model.encode(inputs)
            class_prob = self.model.classifier(features)
            return class_prob
        self.explainer = LayerGradCam(forward_wrapper, resolved_layer)
    def generate(self, x: torch.Tensor, target_class: int=0) -> np.ndarray:
        attributions = self.explainer.attribute(x, target=target_class)
        heatmap = attributions.squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        max_intensity = np.max(heatmap)
        if max_intensity > 0:
            heatmap /= max_intensity
        orig_h, orig_w = (x.shape[2], x.shape[3])
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized
    def overlay(self, heatmap: np.ndarray, original: np.ndarray) -> np.ndarray:
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        if original.dtype != np.uint8:
            if np.max(original) <= 1.0:
                original = np.uint8(255 * original)
            else:
                original = np.uint8(original)
        alpha = 0.5
        overlaid_visual = cv2.addWeighted(colored_heatmap, alpha, original, 1 - alpha, 0)
        return overlaid_visual