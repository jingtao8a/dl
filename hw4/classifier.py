import torch
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        self.parent = nn.Linear(40, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=2)

        self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, n_spks),
		)

    def forward(self, mels):
        """
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
        out = self.parent(mels) # (batch size, length, d_model)
        out = out.permute(1, 0, 2) # (length, batch size, d_model)
        out = self.encoder_layer(out)# (length, batch size, d_model)
        out = out.transpose(0, 1)# (batch size, length, d_model)

        stats = out.mean(dim = 1)  # (batch size, d_model)

        out = self.pred_layer(stats) # (batch size, n_spks)

        return out