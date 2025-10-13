import torch
from transformer.Transformer import Transformer

model = Transformer(
    source_vocabulary_size=10000,
    target_vocabulary_size=10000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=512,
    max_length=128,
    dropout=0.1
)

source = torch.randint(0,10000,(2,5))
target = torch.randint(0,10000,(2,5))

logits = model(source,target)
print(f"Input Shape : ${source.shape} / ${target.shape}")
print(f"Logit Shape : ${logits.shape}")
print(logits)