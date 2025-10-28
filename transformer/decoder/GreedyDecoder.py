# transformer/decoder/GreedyDecoder.py
import torch

class GreedyDecoder:
    def __init__(self, model, tokenizer, device="mps", max_len=30):
        """
        Greedy Decoder for Transformer
        Args:
            model: Transformer model
            tokenizer: trained BPETokenizer
            device: "cpu", "cuda", or "mps"
            max_len: maximum generation length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    def decode(self, sentence):
        """
        Generate text from an input sentence using greedy decoding.
        """
        self.model.eval()
        with torch.no_grad():
            # 1. 입력 문장 토크나이징 및 텐서 변환
            src = self.tokenizer.batch_to_tensor([sentence], device=self.device)

            # 2. 시작 토큰(<sos>)으로 초기화
            tgt = torch.tensor([[self.tokenizer.token_to_id["<sos>"]]], device=self.device)

            # 3. 한 토큰씩 생성
            for _ in range(self.max_len):
                output = self.model(src, tgt)
                next_token = output[:, -1, :].argmax(-1).unsqueeze(0)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == self.tokenizer.token_to_id["<eos>"]:
                    break

            # 4. 토큰 → 문장으로 디코딩
            decoded = self.tokenizer.decode_sentence(tgt.squeeze(0).tolist())
            return decoded
