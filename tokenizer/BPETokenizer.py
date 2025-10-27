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
        pairs = defaultdict(int)
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
            w_out = pattern.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, corpus, num_merges=100):
        vocab = Counter(corpus.split())
        self.vocab = {' '.join(word) + ' </w>': freq for word, freq in vocab.items()}
        #tokens = Counter([' '.join(word) + ' </w>' for word in corpus])
        for i in range(num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            self.merges.append(best)

        tokens = set()
        for word in self.vocab.keys():
            tokens.update(word.split())
        #self.token_to_id = {token: idx for idx, token in enumerate(sorted(tokens))}
        #self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        idx = 0
        for token in self.special_token:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        for token in sorted(tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

    def encode_word(self, word):
        word = list(word) + ["</w>"]
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


corpus = "low lower lowest lowly lower newest wide wider"

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=30)

sentence = "low lowest"
encoded = tokenizer.encode_sentence(sentence)
decoded = tokenizer.decode_sentence(encoded)

print("원문:", sentence)
print("인코딩:", encoded)
print("디코딩:", decoded)
