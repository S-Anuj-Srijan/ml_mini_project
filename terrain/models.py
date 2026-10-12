"""Compact time-domain CNN, with optional feature concatenation."""
import torch
from torch import nn

class TerrainCNN(nn.Module):
    def __init__(self, extra=0):
        super().__init__()
        layers = []
        for cin in [10,24,24,24]:
            layers.extend([nn.Conv1d(cin,24,5,padding=2),nn.ReLU(),
                           nn.MaxPool1d(2),nn.Dropout(.3)])
        self.encoder=nn.Sequential(*layers)
        # 170 -> 85 -> 42 -> 21 -> 10, so flattening yields 240 learned features.
        self.head=nn.Sequential(nn.Linear(240+extra,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,5))
    def forward(self,x,features):
        return self.head(torch.cat([self.encoder(x).flatten(1),features],dim=1))
