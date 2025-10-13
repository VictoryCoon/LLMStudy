import torch
import torch.nn as nn
from .EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, vocabulary_size, d_model, num_layers, num_heads, d_ff, max_length=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_length, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def _generate_positional_encoding(self, max_length, d_model):
        position = torch.arange(max_length).unsqueeze(1).float()
        i = torch.arange(d_model).unsqueeze(0).float()
        angle = position / torch.pow(10000,(2*(i//2))/d_model)
        PE = torch.zeros(max_length, d_model)
        PE[:,0::2] = torch.sin(angle[:,0::2])
        PE[:,1::2] = torch.cos(angle[:,1::2])
        return PE.unsqueeze(0)

    def forward(self, source_tokens, source_mask=None):
        B,T = source_tokens.size()
        x = self.embedding(source_tokens)*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.positional_encoding[:,:T,:].to(x.device)
        for layer in self.layers:
            x = layer(x, source_mask)
        return x