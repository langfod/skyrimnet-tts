"""Minimal stub of VITS discriminator for VC compatibility."""

import torch
from torch import nn


class DiscriminatorS(nn.Module):
    """Stub discriminator - VITS removed, keeping for VC compatibility."""
    
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        print("Warning: VITS discriminator is a stub - functionality removed")
        
    def forward(self, x):
        # Return dummy output with correct shape
        return [torch.zeros_like(x)]