"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Causal Self-Attention with previous probabilities adjustment."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False  # Disable flash attention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, prev_probs=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        if prev_probs is not None:
            epsilon = 0.1  # Adjustable parameter for attention adjustment
            assert prev_probs.size() == (B, T), f"Expected prev_probs shape ({B}, {T}), got {prev_probs.size()}"
            adjustment = epsilon * (-torch.log(prev_probs + 1e-10))  # (B, T)
            adjustment = adjustment.view(B, 1, 1, T)  # Broadcast to (B, 1, 1, T)
            att = att + adjustment

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Feed-forward network within each transformer block."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, prev_probs=None):
        x = x + self.attn(self.ln_1(x), prev_probs=prev_probs)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    """Configuration class for GPT model parameters."""
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = kwargs.get('n_layer', 12)
        self.n_head = kwargs.get('n_head', 12)
        self.n_embd = kwargs.get('n_embd', 768)
        self.dropout = kwargs.get('dropout', 0.1)
        self.bias = kwargs.get('bias', True)

class GPT(nn.Module):
    """Main GPT model class."""
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, prev_probs=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, prev_probs=prev_probs)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, prev_probs=None):
        """Generate tokens with attention adjusted by previous probabilities."""
        B = idx.size(0)
        T_init = idx.size(1)
        if prev_probs is None:
            prev_probs = torch.ones(B, T_init, device=idx.device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            T_cond = idx_cond.size(1)
            prev_probs_cond = prev_probs[:, -T_cond:]
            logits, _ = self(idx_cond, prev_probs=prev_probs_cond)
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                kth_value = v[:, :, -1, None]
                logits = torch.where(logits < kth_value, torch.full_like(logits, float('-inf')), logits)

            probs = F.softmax(logits, dim=-1)
            next_probs = probs[:, -1, :]
            idx_next = torch.multinomial(next_probs, num_samples=1)
            p_next = next_probs.gather(1, idx_next)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            prev_probs = torch.cat((prev_probs, p_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    

########################################Example Usage########################################
# Define a small vocabulary
vocab = ['hello', 'world', 'this', 'is', 'a', 'test']
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Tokenizer functions
def encode(text):
    return [word_to_idx[word] for word in text.split()]

def decode(tokens):
    return ' '.join([idx_to_word[idx] for idx in tokens])

device = 'cpu'
config = GPTConfig(
    vocab_size=len(vocab),
    block_size=8,
    n_layer=2,
    n_head=2,
    n_embd=64,
    dropout=0.0,
    bias=False
)
# Initialize model
model = GPT(config).to(device)

# Initial context
initial_text = "hello world"
initial_tokens = encode(initial_text)
print("Initial token indices:",initial_tokens)
print("Initial text:", initial_text)

idx = torch.tensor([initial_tokens], dtype=torch.long, device=device)

# Generate tokens
generated_idx = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=3)

# Decode and print
generated_text = decode(generated_idx[0].tolist())
print("Generated token indices:", generated_idx[0].tolist())
print("Generated text:", generated_text)