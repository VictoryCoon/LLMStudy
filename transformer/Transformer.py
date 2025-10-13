import torch
import torch.nn as nn
from .decoder.Decoder import Decoder
from .encoder.Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_length=128, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(source_vocabulary_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)
        self.decoder = Decoder(target_vocabulary_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)

    def make_source_mask(self, source_tokens):
        mask = (source_tokens != 0).unsqueeze(1).unsqueeze(1) # (B,1,1,T_src)
        return mask

    def make_target_mask(self, target_tokens):
        B,T = target_tokens.size()
        causal = torch.tril(torch.ones(T,T).bool()).to(target_tokens.device)
        target_padding = (target_tokens != 0).unsqueeze(1).unsqueeze(2) # (B,1,1,T)
        mask = causal.unsqueeze(0).expand(B,-1,-1) & target_padding.squeeze(1)
        return mask

    def forward(self, source_tokens, target_tokens):
        source_mask = self.make_source_mask(source_tokens)
        target_mask = self.make_target_mask(target_tokens)
        enc_output = self.encoder(source_tokens, source_mask=None)
        logits = self.decoder(target_tokens, enc_output, target_mask=target_mask, enc_mask=None)
        return logits