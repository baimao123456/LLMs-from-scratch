import json
import collections

class Encoder:
    def __init__(self, train_file, vocab_size):
        self.train_file = train_file
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.num_merge = self.vocab_size - 256
        self.text_bytes = self.get_vocab(train_file)
        self.merge_vocab(self.num_merge)
        self.build_vocab_index()
   
    # 读取训练样本,并转化为bytes格式
    def get_vocab(self, train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            text = f.read()
            text_bytes = [bytes([b]) for b in text.encode('utf-8')]
        return text_bytes

    # 对语料中所有字节对的出现次数进行计数
    def get_stats(self, byte_arr):
        count = collections.defaultdict(int)
        for pair in zip(byte_arr[:-1], byte_arr[1:]):
            count[pair] += 1
        return count

    # 合并频次最高的字节对
    def merge(self, text_bytes, pair):
        new_byte = b''.join(pair)
        new_bytes = []
        i = 0
        while i < len(text_bytes):
            if i < len(text_bytes)-1 and text_bytes[i] == pair[0] and text_bytes[i+1] == pair[1]:
                new_bytes.append(new_byte)
                i += 2
            else:
                new_bytes.append(text_bytes[i])
                i += 1
        return new_bytes
    
    def merge_vocab(self, num_merge):
        #print(f"Before: {self.text_bytes}. Length: {len(self.text_bytes)}.")
        assert num_merge > 0
        for i in range(num_merge):
            self.stats = self.get_stats(self.text_bytes)
            top_pair = max(self.stats, key=self.stats.get)
            new_byte = b''.join(top_pair)
            print(f"第{i + 1}轮合并：{top_pair} -> {new_byte}")
            self.text_bytes = self.merge(self.text_bytes, top_pair)
            self.merges[top_pair] = 256 + i		# 256 + i可以作为字节对top_pair的整数索引
        #print(f"After: {self.text_bytes}. Length: {len(self.text_bytes)}.")

    def build_vocab_index(self):
        #建立字节对和整数索引之间的映射关系，作为词表
        self.vocab = {i : bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            assert idx not in self.vocab
            self.vocab[idx] = b''.join(pair)
        assert len(self.vocab) == self.vocab_size

    def encode(self, text):
        bytes2id = {b : i for i, b in self.vocab.items()}
        sub_words = [bytes([b]) for b in text.encode('utf-8')]
        print(f"Before merged: {sub_words}")
        while len(sub_words) >= 2:
            pairs = [(x0, x1) for x0, x1 in zip(sub_words[:-1], sub_words[1:])]
            top_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if top_pair not in self.merges:
                break
            sub_words = self.merge(sub_words, top_pair)
            print(f"top pair: {top_pair}")
        print(f"After merged: {sub_words}")
        tokens = [bytes2id[b] for b in sub_words]
        return tokens, sub_words

    def decode(self, tokens):
        sub_words = [self.vocab[t] for t in tokens]
        text = b''.join(sub_words).decode('utf-8')
        return text

    def save_merges(self, filename):
        with open(filename, 'w') as f:
            for pair, idx in self.merges.items():
                f.write('\t'.join(map(str, list(pair) + [idx, '\n'])))

    def save_vocab(self, filename):
        with open(filename, 'w') as f:
            for v, idx in self.vocab.items():
                f.write('\t'.join(map(str, [v, idx, '\n'])))

if __name__ == '__main__':
    # 确定词汇表大小
    # 单个字节的值的范围为[0, 255]
    # 我们把词表的前256个位置留给单个字节，把后44个位置留给合并之后的字节对，也就是总共可以进行42次合并操作
    #train_file = './programmers-intro-to-unicode'
    train_file = './pg16457.txt'
    vocab_file = './train_vocab'
    merges_file = './train_merges'
    vocab_size = 1000

    encoder = Encoder(train_file, vocab_size)
    print('merge: ', encoder.merges)
    print('vocab: ', encoder.vocab)
    encoder.save_vocab(vocab_file)
    encoder.save_merges(merges_file)

    stats = encoder.get_stats(encoder.text_bytes)
    stats_sort = sorted(stats.items(), key=lambda x:x[1], reverse=True)
    print(f"stats[:5]: {stats_sort[:5]}")

    text = "good morning"
    tokens, sub_words = encoder.encode(text)
    print(f"{text} -> {tokens} {sub_words}")

    text = encoder.decode(tokens)
    print(f"{tokens} -> {text}")
