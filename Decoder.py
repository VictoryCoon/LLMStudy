import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.size() # B는 Query의 무엇인가?
        _, Lk, _ = k.size()
        _, Lv, _ = v.size()

        Q = self.W_Q(q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_K(k).view(B, Lv, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2,-1))/torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        #Mask : Expected shape boradcastable to (B, h, Lq, Lk)
#        if mask is not None:
#            scores = scores.masked_fill(mask.unsqueeze(1),float('-inf'))

        weigths = F.softmax(scores, dim=-1)
        out = weigths @ V

        out = out.transpose(1,2).contiguous().view(B, Lq, self.d_model)
        return self.W_O(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn          = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt : Decoder input Embeddings
        # memory : Encoder output
        # tgt_mask : Mask for masked self-attention
        # memory_mask : Optional mask for encoder-decoder attention

        # Masked self-attention
        _tgt = tgt
        self_attn_out = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(_tgt + self.dropout(self_attn_out))

        # Encoder-Decoder attention
        _tgt = tgt
        enc_dec_out = self.enc_dec_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(_tgt + self.dropout(enc_dec_out))

        # Feed-Forward
        _tgt = tgt
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))

        return tgt

batch_size = 2
src_len = 7 # Encoder length (source)
tgt_len = 5 # Decoder length (target)
d_model = 16
num_heads = 4
d_ff = 64

encoder_output = torch.randn(batch_size, src_len, d_model)
decoder_input = torch.randn(batch_size, tgt_len, d_model)

tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1,-1)

decoder_layer = TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
out = decoder_layer(decoder_input, encoder_output, tgt_mask=tgt_mask, memory_mask=None)

print(f"encoder_outputs.shape      : ${encoder_output.shape}")   # (B, src_len, d_model)
print(f"decoder_inputs.shape       : ${decoder_input.shape}")     # (B, tgt_len, d_model)
print(f"tgt_mask.shape             : ${tgt_mask.shape}")                # (B, tgt_len, tgt_len)
print(f"decoder_layer output shape : ${out.shape}")         # (B, tgt_len, d_model)

mha = decoder_layer.self_attn

with torch.no_grad():
    Q = mha.W_Q(decoder_input).view(batch_size, tgt_len, num_heads, d_model // num_heads).transpose(1, 2)
    K = mha.W_K(decoder_input).view(batch_size, tgt_len, num_heads, d_model // num_heads).transpose(1, 2)
    scores = (Q @ K.transpose(-2,-1)/torch.sqrt(torch.tensor(d_model/num_heads, dtype=torch.float32))).squeeze()

    print(f"\nSample scores (before masking) shape : ${scores.shape}")
    print(f"Scores for sample 0, head 0 (before masking)\n${scores[0,0]}")
    scores_masked = scores.masked_fill(tgt_mask.unsqueeze(1), float('-inf'))
    print(f"\nScores for sample 0, head 0 (after masking)\n${scores_masked[0,0]}")
