"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache

@lru_cache
def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.
    """
    
    # the 188 integers that render fine in their original form and need no shifting
    # 188个常用字符的unicode代码点
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # all integers b in bs will simply map to chr(b) in the output dict
    n = 0
    # 词表的大小为255
    for b in range(2**8):
        # if this byte is "ugly" then map it to the next available "nice" character
        # 如果b代码点并不常用，或者是表示'\x00'，并不是单个字符，会增词表的大小
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)   # 将b映射到另外一个能表示单个字符的代码点
            n += 1
    cs = [chr(n) for n in cs]   # 将cs由代码点转化为字符
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    返回word中前后相邻的两个字符的pair对
    """ 
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        # 外部传入的encoder.json, 为字符到index的映射
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        # unicode代码点到字符的映射 
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        # 形成二元组对应共现频率的字典，其中bpe_merges是从已经统计好的文件中读取二元组频率数据
        # 读取的文件中每行是一个二元组，行号即为频率，行号越小频率越高
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # 对text进行切分
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        # cache缓存加速bpe算法
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # find the next lowest rank bigram that can be merged
            # 找到word字符pair中频次最高的二元组进行merge
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf'))) # 优先合并共现频率高的二元组
            #print('bigram: ', bigram, ', rank: ', self.bpe_ranks.get(bigram, float('inf')))
            # 如果二元组出现频率过低则跳过
            if bigram not in self.bpe_ranks:    # 如果剩下的二元组共现频率过低则跳过
                break
            # 获取二元组的前后两个字符
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):    # 合并二元组(考虑多次出现的情况)
                # find the next occurence of first in the sequence of current words
                # 从word[i:]找到first在word[i:]中的起始位置j, 能找到：将word[i:j]作为new_word，进行下一次merge
                print('word: ', word)
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j           # i指向first所在的位置
                except:
                    # 找不到，说明word[i:]均需要merge
                    new_word.extend(word[i:])
                    break

                # 如果first和second均在word[i:]内, 则将first和second进行合并
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    # 若剩余的字符不满足merge，将剩余的字符和new_word拼接，后续重新生成pair后再merge
                    new_word.append(word[i])
                    i += 1
            # 对于merge后的word重新生成pair进行merge
            new_word = tuple(new_word)
            word = new_word
            # 如果合并后只有一个word，则跳过，不需要合并
            if len(word) == 1:
                break
            else:
                # 将合并后的word重新生成pair进行合并
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        text: 原始训练文本数据
        0. 通过pat正则表达式提取有用文字
        1. 将token按照utf8编码，并转化为unicode, 并将一些ugly字符映射到正常字符，然后拼接
        2. 通过bpe函数对token的码点按照二元组的频率进行组合，并按照空格分割
        3. 通过encoder.json对合并后的token编码
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # token被encode为bypes，输出unicode代码点,并对某些ugly字符进行重映射
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对token里的两元组字符进行合并
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(model_name, model_dir):
    # 如果没有下载，要先下载这两个文件
    #!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
    #!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json

    # encoder.json为unicode代码点到字符的映射
    # vocab.bpe 为二元组频率，按照从高到低排序
    with open(os.path.join(model_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(model_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

if __name__ == '__main__':
    print('pairs: ', get_pairs(('hello', 'world', 'beijing')))
    print('pairs: ', get_pairs(('h', 'e', 'll')))
    print('bytes_to_unicode: ', bytes_to_unicode())
    encoder = get_encoder('gpt2_model', './')
    print(encoder.bpe('aahhhhelloolllo'))

