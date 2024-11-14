
train_file = './programmers-intro-to-unicode'

with open(train_file, 'r', encoding='utf-8') as f:
    text = f.read()

text_bytes = [bytes([b]) for b in text.encode('utf-8')]

# 确定词汇表大小
# 单个字节的值的范围为[0, 255]
# 我们把词表的前256个位置留给单个字节，把后44个位置留给合并之后的字节对，也就是总共可以进行42次合并操作
vocab_size = 300

# 对语料中所有字节对的出现次数进行计数
def get_stats(byte_arr):
    count = {}
    for pair in zip(byte_arr[:-1], byte_arr[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count
stats = get_stats(text_bytes)
stats_sort = sorted(stats.items(), key=lambda x:x[1], reverse=True)

print(f"stats[:5]: {stats_sort[:5]}")

# 确定出现频次最高的字节对
top_pair = max(stats, key=stats.get)
print(f"Top pair {top_pair} has {stats[top_pair]} counts.")

# 合并频次最高的字节对
def merge(text_bytes, pair):
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

tmp = text_bytes[30:50]
top_pair = [b'w', b'3']
new_bytes_tmp = merge(tmp, top_pair)
print(f"Before: {tmp}. Length: {len(tmp)}.")
print(f"After: {new_bytes_tmp}. Length: {len(new_bytes_tmp)}.")

# 重复上述merge操作，迭代进行替换
text_bytes = text_bytes[200:300]
print(f"Before: {text_bytes}. Length: {len(text_bytes)}.")
merges = {}
num_merge = vocab_size - 256
assert num_merge > 0
for i in range(num_merge):
    stats = get_stats(text_bytes)
    top_pair = max(stats, key=stats.get)
    new_byte = b''.join(top_pair)
    print(f"第{i + 1}轮合并：{top_pair} -> {new_byte}")
    text_bytes = merge(text_bytes, top_pair)
    merges[top_pair] = 256 + i		# 256 + i可以作为字节对top_pair的整数索引
print(f"After: {text_bytes}. Length: {len(text_bytes)}.")

#建立字节对和整数索引之间的映射关系，作为词表
vocab = {i : bytes([i]) for i in range(256)}
for pair, idx in merges.items():
    assert idx not in vocab
    vocab[idx] = b''.join(pair)
assert len(vocab) == vocab_size

# 打印一些词表内容出来看看
for i in range(256, 256 + 10):
    print(f"{i}: {vocab[i]}")

def encode(text, merges, vocab):
    bytes2id = {b : i for i, b in vocab.items()}
    sub_words = [bytes([b]) for b in text.encode('utf-8')]
    print(f"Before merged: {sub_words}")
    while len(sub_words) >= 2:
        pairs = [(x0, x1) for x0, x1 in zip(sub_words[:-1], sub_words[1:])]
        top_pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if top_pair not in merges:
            break
        sub_words = merge(sub_words, top_pair, new_byte)
        print(f"top pair: {top_pair}")
    tokens = [bytes2id[b] for b in sub_words]
    return tokens, sub_words

def decode(tokens, vocab):
    sub_words = [vocab[t] for t in tokens]
    text = b''.join(sub_words).decode('utf-8')
    return text

text = "good morning"
tokens, sub_words = encode(text, merges, vocab)
print(f"{text} -> {tokens} {sub_words}")

text = decode(tokens, vocab)
print(f"{tokens} -> {text}")
