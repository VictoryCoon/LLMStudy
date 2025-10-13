import torch
import torch.nn as nn
from .DecoderLayer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, vocabulary_size, d_model, num_layers, num_heads, d_ff, max_length=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_length, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocabulary_size)
        self.d_model = d_model

    def _generate_positional_encoding(self, max_length, d_model):
        position = torch.arange(max_length).unsqueeze(1).float()
        i = torch.arange(d_model).unsqueeze(0).float()
        angle = position / torch.pow(10000,(2*(i//2))/d_model)
        PE = torch.zeros(max_length, d_model)
        PE[:,0::2] = torch.sin(angle[:,0::2])
        PE[:,1::2] = torch.cos(angle[:,1::2])
        return PE.unsqueeze(0)

    def forward(self, target_tokens, enc_output, target_mask=None, enc_mask=None):
        B,T = target_tokens.size()
        x = self.embedding(target_tokens)*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.positional_encoding[:,:T,:].to(x.device)
        for layer in self.layers:
            x = layer(x, enc_output, target_mask=target_mask, enc_mask=enc_mask)
        logits = self.linear(x)
        return logits