import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch, heads, seq_len, d_k]
        d_k     = Q.shape[-1]
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn   = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) trebuie să fie multiplu de n_heads ({n_heads})"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout   = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.shape[0]

        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        x, attn = self.attention(Q, K, V, mask)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.dropout(self.W_O(x)), attn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1]])

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask=None, src_mask=None):
        attn1, _ = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(attn1))

        attn2, cross_attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x, cross_attn_weights

class TransformerNL2SQL(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 d_model:  int = 256,
                 n_heads:  int = 8,
                 n_layers: int = 3,
                 d_ff:     int = 512,
                 dropout:  float = 0.1,
                 max_len:  int = 512):
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model, padding_idx=0)
        self.pos_encoding  = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(d_model, trg_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg, pad_idx=0):
        T = trg.shape[1]
        pad_mask   = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(T, T, device=trg.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        return pad_mask & causal_mask

    def encode(self, src, src_mask=None):
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, trg, enc_out, trg_mask=None, src_mask=None):
        x = self.pos_encoding(self.trg_embedding(trg) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x, _ = layer(x, enc_out, trg_mask, src_mask)
        return x

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(trg, enc_out, trg_mask, src_mask)

        return self.fc_out(dec_out)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)