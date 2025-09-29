"""Minimal stub of Glow-TTS duration predictor for compatibility."""

import torch
from torch import nn


class DurationPredictor(nn.Module):
    """Stub duration predictor - Glow-TTS removed, keeping for compatibility."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Warning: Glow-TTS DurationPredictor is a stub - functionality removed")
        
    def forward(self, x, *args, **kwargs):
        # Return dummy duration prediction
        return torch.ones(x.size(0), x.size(2), device=x.device)