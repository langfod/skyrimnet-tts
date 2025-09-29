"""Minimal stub of VITS networks for VC compatibility."""

import torch
from torch import nn


class PosteriorEncoder(nn.Module):
    """Stub posterior encoder - VITS removed, keeping for VC compatibility."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Warning: VITS PosteriorEncoder is a stub - functionality removed")
        
    def forward(self, x, *args, **kwargs):
        # Return dummy output with correct shape
        return x, torch.zeros(x.size(0), device=x.device)


class ResidualCouplingBlocks(nn.Module):
    """Stub residual coupling blocks - VITS removed, keeping for VC compatibility."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Warning: VITS ResidualCouplingBlocks is a stub - functionality removed")
        
    def forward(self, x, *args, **kwargs):
        return x


class TextEncoder(nn.Module):
    """Stub text encoder - VITS removed, keeping for VC compatibility."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Warning: VITS TextEncoder is a stub - functionality removed")
        
    def forward(self, x, *args, **kwargs):
        return x