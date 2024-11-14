import torch
import torch.nn as nn
import math

class SelfAttention_v0(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        querys = x @ self.W_query   # (B,S,D) * (D, D_out) -> (B,S,D_out)
        keys = x @ self.W_key       # (B,S,D) * (D, D_out) -> (B,S,D_out)
        values = x @ self.W_value   # (B,S,D) * (D, D_out) -> (B,S,D_out)
        
        attn_scores = querys @ keys.permute(0, 2, 1)   # (B,S,D_out) * (B, D_out, S) -> (B,S,S)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values # (B,S,S) * (B,S,D_out) -> (B,S,D_out)
        return context_vec
 
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        querys = self.W_query(x)   # (B,S,D) * (D, D_out) -> (B,S,D_out)
        keys = self.W_key(x)       # (B,S,D) * (D, D_out) -> (B,S,D_out)
        values = self.W_value(x)   # (B,S,D) * (D, D_out) -> (B,S,D_out)
        
        attn_scores = querys @ keys.permute(0, 2, 1)   # (B,S,D_out) * (B, D_out, S) -> (B,S,S)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values # (B,S,S) * (B,S,D_out) -> (B,S,D_out)
        return context_vec
 
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # (context_length,context_length)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        querys = self.W_query(x)      # (B,S,D) * (D, D_out) -> (B,S,D_out)
        keys = self.W_key(x)          # (B,S,D) * (D, D_out) -> (B,S,D_out)
        values = self.W_value(x)      # (B,S,D) * (D, D_out) -> (B,S,D_out)
        
        attn_scores = querys @ keys.permute(0, 2, 1)   # (B,S,D_out) * (B, D_out, S) -> (B,S,S)
        #attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # in-place，填充上三角mask
        print('attn_scores: ', attn_scores)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values # (B,S,S) * (B,S,D_out) -> (B,S,D_out)
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj =  nn.Linear(d_out, d_out)    # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # (context_length,context_length)

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

        attn_scores = querys @ keys.transpose(2, 3) # (B,H,S,D_head) * (B,H,D_head,S) -> (B,H,S,S)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (B,H,S,S) * (B,H,S,D_head) -> (B,H,S,D_head)
        
        # commbine heads, contiguous改变了矩阵存储的位置，方便view使用
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) #  (B,H,S,D_head) -> (B,S,D_out)
        context_vec = self.out_proj(context_vec)    # (B,S,D_out) * (D_out,D_out) -> (B,S,D_out)

        return context_vec
 
class MultiHeadAttentionCombinedQKV(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # combine qkv
        self.qkv = nn.Linear(d_in, 3*d_out, bias=qkv_bias)
        self.out_proj =  nn.Linear(d_out, d_out)    # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # (context_length,context_length)

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        
        qkv = self.qkv(x)   # (B,S,D) * (D,3*D_out) -> (B,S,3*D_out)

        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim) # (B,S,3*D_out) -> (B,S,3,H,D_head)

        qkv = qkv.permute(2, 0, 3, 1, 4) # (B,S,3,H,D_head) -> (3,B,H,S,D_head)

        querys, keys, values = qkv.unbind(0) # 3 * (B,H,S,D_head)

        attn_scores = querys @ keys.transpose(2, 3) # (B,H,S,D_head) * (B,H,D_head,S) -> (B,H,S,S)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (B,H,S,S) * (B,H,S,D_head) -> (B,S,H,D_head)
        
        # commbine heads, contiguous改变了矩阵存储的位置，方便view使用
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) #  (B,S,H,D_head) -> (B,S,D_out)
        context_vec = self.out_proj(context_vec)    # (B,S,D_out) * (D_out,D_out) -> (B,S,D_out)

        return context_vec

class MHAEinsum(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)
        
        self.out_proj =  nn.Linear(d_out, d_out)    # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # (context_length,context_length)

        # Initialize parameters
        self.reset_parameters() 
   
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)
        
    def forward(self, x):
        b, n, _ = x.shape # New batch dimension b

        Q = torch.einsum("bnd,di->bni", x, self.W_query) # (B,S,D) * (D,D_out) -> (B,S,D_out)
        K = torch.einsum("bnd,di->bni", x, self.W_key)   # (B,S,D) * (D,D_out) -> (B,S,D_out)
        V = torch.einsum("bnd,di->bni", x, self.W_value) # (B,S,D) * (D,D_out) -> (B,S,D_out)

        if self.bias_q is not None:
            Q += self.bias_q
            K += self.bias_k
            V += self.bias_v

        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (B,S,D_out) -> (B,S,H,D_head) -> (B,H,S,D_head)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (B,S,D_out) -> (B,S,H,D_head) -> (B,H,S,D_head)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (B,S,D_out) -> (B,S,H,D_head) -> (B,H,S,D_head)

        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5) # (B,H,S,D_head) * (B,H,S,D_head) -> (B,H,S,S)

        mask = self.mask[:n, :n].unsqueeze(0).unsqueeze(1).expand(b, self.num_heads, n, n) # (S,S) -> (B,H,S,S)
        score = scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V) # (B,H,S,S) * (B,H,S,D_head) ->  (B,H,S,D_head)

        # combine 
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)  # (B,H,S,D_head) -> (B,S,H,D_head) -> (B,S,D_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length
        self.d_out = d_out

        # combine qkv
        self.qkv = nn.Linear(d_in, 3*d_out, bias=qkv_bias)
        self.out_proj =  nn.Linear(d_out, d_out)    # Linear layer to combine head outputs
        self.dropout = dropout

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        
        qkv = self.qkv(x)   # (B,S,D) * (D,3*D_out) -> (B,S,3*D_out)

        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim) # (B,S,3*D_out) -> (B,S,3,H,D_head)

        qkv = qkv.permute(2, 0, 3, 1, 4) # (B,S,3,H,D_head) -> (3,B,H,S,D_head)

        querys, keys, values = qkv.unbind(0) # 3 * (B,H,S,D_head)

        use_dropout = 0.0 if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            querys, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True) # 

        # commbine heads, contiguous改变了矩阵存储的位置，方便view使用
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out) #  (B,H,S,D_head) -> (B,S,H,D_head) -> (B,S,D_out)
        context_vec = self.out_proj(context_vec)    # (B,S,D_out) * (D_out,D_out) -> (B,S,D_out)

        return context_vec

class MHAPyTorchSDPAWithoutFlash(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length
        self.d_out = d_out

        # combine qkv
        self.qkv = nn.Linear(d_in, 3*d_out, bias=qkv_bias)
        self.out_proj =  nn.Linear(d_out, d_out)    # Linear layer to combine head outputs
        self.dropout = dropout
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        
        qkv = self.qkv(x)   # (B,S,D) * (D,3*D_out) -> (B,S,3*D_out)

        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim) # (B,S,3*D_out) -> (B,S,3,H,D_head)

        qkv = qkv.permute(2, 0, 3, 1, 4) # (B,S,3,H,D_head) -> (3,B,H,S,D_head)

        querys, keys, values = qkv.unbind(0) # 3 * (B,H,S,D_head)

        use_dropout = 0.0 if not self.training else self.dropout

        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        context_vec = nn.functional.scaled_dot_product_attention(
            querys, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False) # 

        # commbine heads, contiguous改变了矩阵存储的位置，方便view使用
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out) #  (B,H,S,D_head) -> (B,S,H,D_head) -> (B,S,D_out)
        context_vec = self.out_proj(context_vec)    # (B,S,D_out) * (D_out,D_out) -> (B,S,D_out)

        return context_vec

class MHAPyTorchClass(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, need_weights=True):
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        output = self.proj(attn_output)

        return output

#from torch.nn.attention.flex_attention import flex_attention, create_block_mask
#def causal(b, h, q_idx, kv_idx):
#    return q_idx >= kv_idx
#class MHAPyTorchFlexAttention(nn.Module):
#
#    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
#        super().__init__()
#
#        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"
#
#        self.num_heads = num_heads
#        self.context_length = context_length
#        self.head_dim = d_out // num_heads
#        self.d_out = d_out
#
#        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
#        self.out_proj = nn.Linear(d_out, d_out)
#        self.dropout = dropout
#        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
#        # `create_block_mask` function does not support buffers, yet
#        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length)
#
#    def forward(self, x):
#        b, num_tokens, embed_dim = x.shape
#
#        qkv = self.qkv(x)   # (B,S,D) -> (B,S,3*D)
#
#        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim) # (B,S,3*D) -> (B,S,3,H,D_head)
#
#        qkv = qkv.permute(2, 0, 3, 1, 4)    # (B,S,3,H,D_head) -> (3,B,H,S,D_head)
#
#        queries, keys, values = qkv # (3,B,H,S,D_head) -> 3 * (B,H,S,D_head)
#
#        use_dropout = 0. if not self.training else self.dropout
#
#        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
#        # No need to manually adjust for num_heads; ensure it's right for the sequence
#        if self.context_length >= num_tokens:
#            attn_mask = self.block_mask[:num_tokens, :num_tokens]
#        else:
#            attn_mask = self.block_mask[:self.context_length, :self.context_length]
#
#        context_vec = flex_attention(queries, keys, values, block_mask=attn_mask) #  (B,H,S,D_head)
#
#        # Combine heads, where self.d_out = self.num_heads * self.head_dim
#        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out) # (B,H,S,D_head) -> (B,S,H,D_head) -> (B,S,D_out)
#
#        context_vec = self.out_proj(context_vec)    # (B,S,D_out) * (D_out,D_out) -> (B,S,D_out)
#
#        return context_vec

if __name__ == '__main__':
    inputs = torch.tensor(
    [
      [[0.43, 0.15, 0.89, 0.02], # Your     (x^1)
       [0.55, 0.87, 0.66, 0.02], # journey  (x^2)
       [0.57, 0.85, 0.64, 0.02], # starts   (x^3)
       [0.22, 0.58, 0.33, 0.02], # with     (x^4)
       [0.77, 0.25, 0.10, 0.02], # one      (x^5)
       [0.05, 0.80, 0.55, 0.02]], # step     (x^6)
      [[0.43, 0.15, 0.89, 0.02], # Your     (x^1)
       [0.55, 0.87, 0.66, 0.02], # journey  (x^2)
       [0.57, 0.85, 0.64, 0.02], # starts   (x^3)
       [0.22, 0.58, 0.33, 0.02], # with     (x^4)
       [0.77, 0.25, 0.10, 0.02], # one      (x^5)
       [0.05, 0.80, 0.55, 0.02]] # step     (x^6)
    ]
    )
    print('inputs.shape', inputs.shape)
    d_in = inputs.shape[-1] # the input embedding size, d=4
    d_out = 3 # the output embedding size, d=3



    print('SelfAttention_v0 test')
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v0(d_in, d_out)
    print(sa_v1(inputs[0]))

    print('SelfAttention_v1 test')
    torch.manual_seed(456)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    print('SelfAttention_v2 test')
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    print('CausalAttention test')
    context_length = inputs.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


    print('MultiHeadAttentionWrapper test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 3
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    print('MultiHeadAttention test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 6
    mha = MultiHeadAttention(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    print('MultiHeadAttentionCombinedQKV test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 6
    mha = MultiHeadAttentionCombinedQKV(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    print('MHAEinsum test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 6
    mha = MHAEinsum(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    print('MHAPyTorchScaledDotProduct test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 6
    mha = MHAPyTorchScaledDotProduct(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    print('HAPyTorchSDPAWithoutFlash test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 6
    mha = MHAPyTorchSDPAWithoutFlash(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    print('MHAPyTorchClass test')
    torch.manual_seed(123)
    context_length = inputs.shape[1] # This is the number of tokens
    d_in, d_out = 4, 6
    mha = MHAPyTorchClass(
        d_in=d_in, d_out=d_in, num_heads=2, context_length=context_length, dropout=0.0 
    )
    context_vecs = mha(inputs)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    #print('MHAPyTorchFlexAttention test')
    #torch.manual_seed(123)
    #context_length = inputs.shape[1] # This is the number of tokens
    #d_in, d_out = 4, 6
    #mha = MHAPyTorchFlexAttention(
    #    d_in=d_in, d_out=d_in, num_heads=2, context_length=context_length, dropout=0.0 
    #)
    #context_vecs = mha(inputs)
    #print(context_vecs)
    #print("context_vecs.shape:", context_vecs.shape)
