"""Compact segmentation U-Net used for APD-Net task feedback."""
from __future__ import annotations
import torch
from torch import nn
class Segmentor(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,base,3,padding=1),nn.GroupNorm(8,base),nn.GELU(),nn.Conv2d(base,base*2,3,padding=1),nn.GroupNorm(8,base*2),nn.GELU(),nn.Conv2d(base*2,base*2,3,padding=1),nn.GELU(),nn.Conv2d(base*2,1,1))
    def forward(self,x): return self.net(x)
