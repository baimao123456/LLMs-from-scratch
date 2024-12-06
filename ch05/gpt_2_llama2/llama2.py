
import torch
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

import torch 
import torch.nn as nn

#class GPTModel(nn.Module):
class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
        #self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        #self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[(TransformerBlock(cfg)) for _ in range(cfg['n_layers'])]
        )
        #self.final_norm = LayerNorm(cfg['emb_dim'])
        self.final_norm = RMSNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype']
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx)
        #pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds # + pos_embeds
        #x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj =  nn.Linear(d_out, d_out)    # Linear layer to combine head outputs
        #self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # (context_length,context_length)

        # rope 
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        querys = self.W_query(x)      # (B,S,D) * (D, D_out) -> (B,S,D_out)
        keys = self.W_key(x)          # (B,S,D) * (D, D_out) -> (B,S,D_out)
        values = self.W_value(x)      # (B,S,D) * (D, D_out) -> (B,S,D_out)
        
        # split matrix by num_heads
        querys = querys.view(b, num_tokens, self.num_heads, self.head_dim)  # (B,S,D_out) -> (B,S,H,D_head)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      # (B,S,D_out) -> (B,S,H,D_head)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  # (B,S,D_out) -> (B,S,H,D_head)

        # transpose (B,S,H,D) -> (B,H,S,D)
        querys = querys.transpose(1, 2)     # (B,S,H,D_head) -> (B,H,S,D_head)
        keys = keys.transpose(1, 2)         # (B,S,H,D_head) -> (B,H,S,D_head)
        values = values.transpose(1, 2)     # (B,S,H,D_head) -> (B,H,S,D_head)

        # new add rope position emb
        querys = compute_rope(querys, self.cos, self.sin)
        keys = compute_rope(keys, self.cos, self.sin)

        attn_scores = querys @ keys.transpose(2, 3) # (B,H,S,D_head) * (B,H,D_head,S) -> (B,H,S,S)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        #attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (B,H,S,S) * (B,H,S,D_head) -> (B,H,S,D_head)
        
        # commbine heads, contiguous改变了矩阵存储的位置，方便view使用
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) #  (B,H,S,D_head) -> (B,S,D_out)
        context_vec = self.out_proj(context_vec)    # (B,S,D_out) * (D_out,D_out) -> (B,S,D_out)

        return context_vec
 
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dtype=cfg['dtype']
            #dropout=cfg['drop_rate'],
            #qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        #self.norm1 = LayerNorm(cfg['emb_dim'])
        #self.norm2 = LayerNorm(cfg['emb_dim'])
        self.norm1 = RMSNorm(cfg['emb_dim'])
        self.norm2 = RMSNorm(cfg['emb_dim'])

        #self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        #x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        #x = self.drop_shortcut(x)
        x = x + shortcut

        return x

#class LayerNorm(nn.Module):
#    def __init__(self, emb_dim):
#        super().__init__()
#        self.eps = 1e-5
#        self.scale = nn.Parameter(torch.ones(emb_dim)) 
#        self.shift = nn.Parameter(torch.zeros(emb_dim)) 
#
#    def forward(self, x):
#        mean = x.mean(dim=-1, keepdim=True)
#        var = x.var(dim=-1, keepdim=True, unbiased=False)
#        norm_x = (x - mean) / torch.sqrt(var + self.eps)
#        return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed  * self.weight).to(dtype=x.dtype)

#class GELU(nn.Module):
#    def __init__(self):
#        super().__init__()
#    
#    def forward(self, x):
#        return 0.5 * x * (1 + torch.tanh(
#            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
#            (x + 0.044715 * torch.pow(x, 3))
#        ))

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

#class FeedForward(nn.Module):
#    def __init__(self, cfg):
#        super().__init__()
#        self.layers = nn.Sequential(
#            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
#            GELU(),
#            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
#        )
#    
#    def forward(self, x):
#        return self.layers(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc2 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc3 = nn.Linear(cfg['hidden_dim'], cfg['emb_dim'], dtype=cfg['dtype'], bias=False)
        self.silu = SiLU()
    
    def forward(self, x):
        x_fc1 = self.fc1(x)     # (B,S,D) x (D,H) -> (B,S,H)
        x_fc2 = self.fc2(x)     # (B,S,D) x (D,H) -> (B,S,H)
        x = self.silu(x_fc1) * x_fc2    # (B,S,H) * (B,S,H) -> (B,S,H)
        return self.fc3(x)      # (B,S,H) x (H,D) -> (B,S,D)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 送到模型的context长度，超出则截断
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # 取序列最后一个token
        logits = logits[:, -1, :]
        
        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 将多次预估的结果进行合并，上一步预估结果作为下一步预估的输入
        idx = torch.cat((idx, idx_next), dim=1) # (batch, n_tokens+1)

    return idx

def main():
    LLAMA2_CONFIG_7B = {
        "vocab_size": 2,     # Vocabulary size
        "context_length": 4096,  # Context length
        "emb_dim": 128,         # Embedding dimension
        "n_heads": 2,           # Number of attention heads
        "n_layers": 2,          # Number of layers
        "hidden_dim": 128,     # NEW: Size of the intermediate dimension in FeedForward
        "dtype": torch.bfloat16  # NEW: Lower-precision dtype to save memory
    }
    model = Llama2Model(LLAMA2_CONFIG_7B)
    print(model)
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"Total number of parameters: {total_params:,}")

if __name__ == '__main__':
    main()
