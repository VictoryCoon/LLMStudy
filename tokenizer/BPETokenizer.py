import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, special_token=None):
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        if special_token is None:
            self.special_token = ['<pad>', '<sos>','<eos>','<unk>']
        else:
            self.special_token = special_token

    def get_stats(self, vocab):
        pairs = defaultdict(int) # 23개가 나오는 근거는??? 46음절의 절반수치다.
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        bigram = re.escape(' '.join(pair))
        # (?<!\S) : 왼쪽에 공백이 아닌 문자(=Non-space) 가 없어야 함
        # (?!\S)  : 오른쪽에 공백이 아닌 문자가 없어야 함
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        v_out = {}
        for word in v_in:
            # pair 에 해당되는 v_in은 공백을 제거해서 읽는다.
            w_out = pattern.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
            # Dictionary라서 그런가?
        return v_out

    def train(self, corpus, num_merges=100):
        vocab = Counter(corpus.split())
        self.vocab = {' '.join(word) + ' </w>': freq for word, freq in vocab.items()}
        #tokens = Counter([' '.join(word) + ' </w>' for word in corpus])
        #받은 단어배열을 전부 공백으로 쪼개고, 단어의 끝에 </w>를 하는 것, 그리고 빈도수 입력
        for i in range(num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get) #가장 좋은 결과를 합쳐서, 다시 단어반환
            self.vocab = self.merge_vocab(best, self.vocab)
            self.merges.append(best) # self.merge의 Element는 마치, 입력,예측 관계같다.

        tokens = set()
        for word in self.vocab.keys(): # Key는 단어들, Values는 빈도
            tokens.update(word.split())
        #self.token_to_id = {token: idx for idx, token in enumerate(sorted(tokens))}
        #self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        idx = 0
        # 다시금 tokens로 인해, vocab사전과 유사한 결과를 얻었다(결과만 보고는 유사 뻘짓인데?)

        for token in self.special_token: # 스페셜 토큰은 4개다.
            # 현재까지 둘은 빈 상태였다.
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1 # idx는 3(0,1,2,3)에서 끝날것이다.

        for token in sorted(tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
        #학습을 끝으로 얻는것은, token_to_id, id_to_token이다.

    def encode_word(self, word):
        word = list(word) + ["</w>"] # 단어를 전부 음절로 자르고, 단어끼리는 </w>구분자를 준다.
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            pair_ranks = {pair: i for i, pair in enumerate(self.merges)}
            ranked = [(pair, pair_ranks[pair]) for pair in pairs if pair in pair_ranks]
            if not ranked:
                break
            best_pair = min(ranked, key=lambda x: x[1])[0]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(''.join(best_pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    # 여기서부터는 학습으로 얻은 token_to_id, id_to_token을 사용해서, 본격 Tokenizer를 진행한다.
    def encode_sentence(self, sentence):
        encoded_sentence = [self.token_to_id['<sos>']]
        for word in sentence.strip().split():
            tokens = self.encode_word(word)
            encoded_sentence.extend([self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens])
        encoded_sentence.append(self.token_to_id['<eos>'])
        return encoded_sentence

    def decode_sentence(self, token_ids):
        tokens = [self.id_to_token.get(i,"<unk>") for i in token_ids]
        tokens = [t for t in tokens if t not in ["<sos>","<eos>","<pad>"]]
        sentence = ' '.join(tokens).replace("</w>","").strip()
        return sentence


#corpus = "low lower lowest lowly lower newest wide wider"
corpus = "this these that thing those me you your my mine one"

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=30)

sentence = "This is your thing."
encoded = tokenizer.encode_sentence(sentence)
decoded = tokenizer.decode_sentence(encoded)

#print("원문:", sentence)
#print("인코딩:", encoded)
#print("디코딩:", decoded)
