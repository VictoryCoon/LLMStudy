import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Lq = Q.size(1)
        Lk = K.size(1)
        Lv = V.size(1)

        Q = self.W_q(Q).view(B,Lq,self.num_heads,self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B,Lk,self.num_heads,self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B,Lv,self.num_heads,self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf")) # -1e9

        attention = F.softmax(scores, dim=-1)
        output = attention @ V
        output = output.transpose(1,2).contiguous().view(B,Lq,self.d_model)
        return self.W_o(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        residual = x
        x = self.self_attention(x, x, x, mask=src_mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)

        return x

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

    def forward(self, src_tokens, src_mask=None):
        B,T = src_tokens.size()
        x = self.embedding(src_tokens)*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.positional_encoding[:,:T,:].to(x.device)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, target_mask=None, enc_mask=None):
        residual = x
        x = self.self_attention(x, x, x, mask=target_mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        residual = x
        x = self.cross_attention(x, enc_output, enc_output, mask=enc_mask)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(residual + x)
        return x

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

class Transformer(nn.Module):
    def __init__(self, src_vocabulary_size, target_vocabulary_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_length=128, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocabulary_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)
        self.decoder = Decoder(target_vocabulary_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)

    def make_src_mask(self, src_tokens):
        mask = (src_tokens != 0).unsqueeze(1).unsqueeze(1) # (B,1,1,T_src)
        return mask

    def make_target_mask(self, target_tokens):
        B,T = target_tokens.size()
        causal = torch.tril(torch.ones(T,T).bool()).to(target_tokens.device)
        target_padding = (target_tokens != 0).unsqueeze(1).unsqueeze(2) # (B,1,1,T)
        mask = causal.unsqueeze(0).expand(B,-1,-1) & target_padding.squeeze(1)
        return mask

    def forward(self, src_tokens, target_tokens):
        src_mask = self.make_src_mask(src_tokens)
        target_mask = self.make_target_mask(target_tokens)
        enc_output = self.encoder(src_tokens, src_mask=None)
        logits = self.decoder(target_tokens, enc_output, target_mask=target_mask, enc_mask=None)
        return logits