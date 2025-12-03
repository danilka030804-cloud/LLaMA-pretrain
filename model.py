import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer
from huggingface_hub import login
from torch.backends.cuda import sdp_kernel
from dataclasses import dataclass

with open("configurate.yaml", "r") as f:
    cfg2 = yaml.safe_load(f)

@dataclass
class ModelConfig:
    embed_size: int = 960
    vocab_size: int = 32000
    num_heads: int = 15


class RoPE(nn.Module):
    def __init__(self, embed_size:int):
        super().__init__()
        self.base = 10000
        self.embed_size = embed_size
        self.cos_cached = None
        self.sin_cached = None

    def build_cache(self, x, seq_len):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        half_dim = self.embed_size // 2
        theta = 1.0 / (self.base ** (torch.arange(0, half_dim, dtype=torch.float32, device=x.device) / half_dim))
        seq_idx = torch.arange(seq_len, dtype=torch.float32, device=x.device)

        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[:, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, :]

    def neg_half(self, x):
        emb_2 = self.embed_size // 2
        return torch.cat([-x[..., emb_2:], x[..., :emb_2]], dim=-1)

    def forward(self, x):
        """
        x: [B, S, H, Hd]
        """
        seq_len = x.shape[1]
        self.build_cache(x, seq_len)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        x_rope = (x * cos[None, :, :, :]) + (self.neg_half(x) * sin[None, :, :, :])
        return x_rope




class AttentionWithPos(nn.Module):
    def __init__(self, embed_size:int, num_heads:int = 32, training: bool = True):
        super().__init__()
        self.training = training
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)

        self.rope = RoPE(embed_size=self.head_dim)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        q = self.q_proj(x).view(B, L, H, Hd)
        k = self.k_proj(x).view(B, L, H, Hd)
        v = self.v_proj(x).view(B, L, H, Hd)

        q = self.rope(q)
        k = self.rope(k)
        
        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = 0.1 if self.training else 0.0,
                is_causal=True
            )

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)




class Encoder(nn.Module):
    def __init__(self, embed_size:int, num_heads:int):
        super().__init__()

        self.silu = nn.SiLU()

        hidden_size = int(8 * embed_size / 3)
        self.net1 = nn.Linear(embed_size, hidden_size, bias=False)
        self.net2 = nn.Linear(embed_size, hidden_size, bias=False)
        self.net3 = nn.Linear(hidden_size, embed_size, bias=False)

        self.norm = nn.RMSNorm(embed_size)
        self.attention = AttentionWithPos(embed_size, num_heads=num_heads)

    def ff(self, x):
        x = self.net3(self.silu(self.net1(x)) * self.net2(x))
        return x

    def forward(self, x):
        x = self.norm(x)
        out = self.attention(x)
        out = out + x
        out2 = self.ff(self.norm(x))
        return out + out2


configurate = ModelConfig()

class LLaMA(nn.Module):
    def __init__(self, config: ModelConfig = configurate, token_conf = cfg2['tokenizer']):
        super().__init__()
        
        login(token=token_conf['login'])
        tokenizer = AutoTokenizer.from_pretrained(token_conf['name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.embed = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_size, padding_idx=tokenizer.pad_token_id)
        self.net = nn.Sequential(*[
            Encoder(embed_size=config.embed_size, num_heads=config.num_heads)
            for _ in range(15)
        ])
        self.fc = nn.Linear(config.embed_size, config.vocab_size)
        self.norm = nn.RMSNorm(normalized_shape=config.embed_size)
        
        
    def forward(self, x):
        x = self.embed(x)
        for layer in self.net:
            x = layer(x)
        logits = self.fc(self.norm(x))
        return logits
    
    @torch.no_grad()
    def generate_simple(self, max_new_tokens:int = 50, temperature:float = 1.0):
        self.eval()
        device = next(self.parameters()).device

        if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
        else:
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self(input_ids)

            next_token_logits = logits[0, -1, :] / temperature

            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).to(device)

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            if hasattr(self.tokenizer, "eos_token_id") and next_token_id.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text