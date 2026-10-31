"""Spiking network: [Linear -> LIF -> BatchNorm -> Dropout] x 2 -> Linear -> LIF."""

from __future__ import annotations

import snntorch as snn
import torch
from torch import nn


class SNNModel(nn.Module):
    def __init__(self, input_size: int = 64 * 64, hidden_size: int = 200, n_classes: int = 4, beta: float = 0.69,
                 dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.lif2 = snn.Leaky(beta=beta)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size // 2, n_classes)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, time, H, W). One time step per frame; returns the output
        spikes and membrane potential at the final time step."""
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        spk3 = mem3
        for t in range(x.shape[1]):
            flat = x[:, t].flatten(1)
            spk1, mem1 = self.lif1(self.fc1(flat), mem1)
            spk1 = self.dropout1(self.bn1(spk1))
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2 = self.dropout2(self.bn2(spk2))
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
        return spk3, mem3
