import torch
import torch.nn as nn
import torch.optim as optim
from transformer.Transformer import Transformer
from tokenizer.BPETokenizer import BPETokenizer
from transformer.decoder.GreedyDecoder import GreedyDecoder

corpus = [
	"나는 밥을 먹었다",
	"나는 학교에 갔다",
	"나는 오늘 출근했다",
	"나는 매일 출근한다",
	"밥은 맛있었다",
	"학교는 재밌었다",
	"출근길은 매우 힘들다",
	"회사는 출근해야 도착한다",
	"회사는 돈을 버는 곳이다",
	"나는 회사원이다",
	"나는 학생이 아니다",
	"나는 학생이 아니기 때문에 등교를 안한다",
	"나는 회사원이기 때문에 출근을 한다"
]

# Tokenizer 정의
tokenizer = BPETokenizer()
tokenizer.train(' '.join(corpus), num_merges=30)
source_batch = tokenizer.batch_to_tensor(corpus)
target_batch = tokenizer.batch_to_tensor(corpus)

# Transformer - Hyperparameter 지정
device = "mps"
model = Transformer(
    source_vocabulary_size=len(tokenizer.token_to_id),
    target_vocabulary_size=len(tokenizer.token_to_id),
    d_model=512,
    num_layers=2,
    num_heads=8,
    d_ff=512,
    max_length=128,
    dropout=0.1
).to(device)

# Learning Loop
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
optimizer = optim.Adam(model.parameters(),lr=1e-4)  #0.0004

EPOCHS = 50

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(source_batch, target_batch[:, :-1])  # <eos> 제외
    loss = criterion(output.reshape(-1,output.size(-1)), target_batch[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    #print(f"Epoch [{epoch+1}/{EPOCHS}], Loss : {loss.item():.4f}")

# Result - 1
# with torch.no_grad():
#     test = "나는 밥을 먹었습니다."
#     encoded = tokenzier.batch_to_tensor([test])
#     output = model(encoded,encoded)
#     print(output.shape)

# Result - 2
decoder = GreedyDecoder(model, tokenizer, device=device, max_len=20)
test_sentence = "에휴"
output_sentence = decoder.decode(test_sentence)

print(f"🧩 입력문장: {test_sentence}")
print(f"💬 출력문장: {output_sentence}")