import re
from collections import Counter,defaultdict

class BPEWordTokenizer:
    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.vocab = {}
        self.bpe_codes = {}

    def get_stats(self, tokens):
        pairs = Counter()   # Iterable Object Return, Data Indexing.
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i],symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, tokens):
        pattern = re.escape(' '.join(pair))
        # (?<!\S) : 왼쪽에 공백이 아닌 문자(=Non-space) 가 없어야 함
        # (?!\S)  : 오른쪽에 공백이 아닌 문자가 없어야 함
        pattern = re.compile(r'(?<!\S)'+pattern +r'(?!\S)')
        new_tokens = {}
        for word, freq in tokens.items():
            new_word = pattern.sub(''.join(pair), word)
            new_tokens[new_word] = freq
        return new_tokens

    def fit(self, corpus):  # learning
        tokens = Counter([' '.join(word) + ' </w>' for word in corpus])
        for i in range(self.num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            tokens = self.merge_vocab(best,tokens)
            self.bpe_codes[best] = i
        self.vocab = list(tokens.keys())

    # Word
    def encode_word(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i],word[i+1]) for i in range(len(word)-1)]
            merges = {pair:self.bpe_codes[pair] for pair in pairs if pair in self.bpe_codes}
            if not merges:
                break
            best = min(merges, key=merges.get)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i],word[i+1]) == best:
                    new_word.append(''.join(best))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def decode_word(self, tokens):
        return ''.join(tokens).replace('</w>','')

#Test
corpus = ["low","lowest","newer","wider","shorter","latest"]
bpe = BPEWordTokenizer(num_merges=10)
bpe.fit(corpus)

print("📘 Learned Vocab:", bpe.vocab[:10])
print("🧩 Encode('lower'):", bpe.encode("lower"))
print("🧩 Decode:", bpe.decode(bpe.encode("lower")))