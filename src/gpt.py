from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


class GPTConfig(BaseModel):
    vocab_size: int = Field(..., description="词表大小")
    block_size: int = Field(..., description="块大小")
    n_layer: int = Field(4, description="层数")
    n_head: int = Field(4, description="头数")
    n_embd: int = Field(256, description="嵌入维度")
    dropout: float = Field(0.1, description=" dropout 率")


class CausalSelfAttention(nn.Module):
    """
    GPT 风格的多头自注意力：
    - 输入: x 形状 (B, T, C)
    - 只允许关注到当前 token 及之前（因果 mask）
    - 可选叠加 padding mask（attention_mask）

    你可以把它理解成三步：
    1) 用线性层一次性算出 Q/K/V（这里用 `qkv` 合并在一起）
    2) 计算注意力权重：softmax( QK^T / sqrt(d) )，并用 mask 把“未来位置”变成不可见
    3) 用权重对 V 做加权求和得到输出，再做一次线性投影回 n_embd 维

    维度符号说明：
    - B: batch size（批大小）
    - T: sequence length（序列长度）
    - C: embedding dim（等于 n_embd）
    - nh: number of heads（等于 n_head）
    - hs: head size（每个头的维度，hs = C / nh）
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd 必须能被 n_head 整除，但得到 n_embd={n_embd}, n_head={n_head}")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 预先注册最大长度的 causal mask（推理/训练更方便）
        mask = torch.tril(torch.ones((block_size, block_size),
                                     dtype=torch.bool)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, T, C)
        b, t, c = x.shape

        # 线性层一次性输出 (Q, K, V)，形状 (B, T, 3C)
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(c, dim=-1)  # 各自 (B, T, C)

        # 把通道维 C 拆成多头：(B, T, nh, hs) 然后转置成 (B, nh, T, hs)
        # 这样就可以对每个 head 单独做注意力
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力打分：QK^T，得到 (B, nh, T, T)
        # 第 3/4 维的 T 表示：对序列中每个位置 i，去看所有位置 j 的相关性分数
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)

        # 因果 mask：禁止看未来
        # causal_mask: (1, 1, block_size, block_size)
        # 取前 t 截断后变成 (1, 1, T, T)，下三角为 True（允许关注），上三角为 False（禁止看未来）
        causal = self.causal_mask[:, :, :t, :t]
        # 把禁止位置填成一个非常小的数（近似 -inf），softmax 后这些位置权重约等于 0
        att = att.masked_fill(~causal, torch.finfo(att.dtype).min)

        # 可选：叠加 padding mask
        # 支持形状：
        # - (B, T) 其中 1=保留，0=mask
        # - (B, 1, 1, T) 或 (B, 1, T, T) 的可广播形状
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, T) -> (B, 1, 1, T)，用于“mask 掉 padding token”
                am = attention_mask[:, None, None, :].to(dtype=torch.bool)
            else:
                am = attention_mask.to(dtype=torch.bool)
            att = att.masked_fill(~am, torch.finfo(att.dtype).min)

        # softmax 得到注意力权重（每一行对所有 key 位置归一化）
        weights = F.softmax(att, dim=-1)
        weights = self.attn_dropout(weights)
        # 用权重对 V 加权求和，得到 (B, nh, T, hs)
        y = weights @ v  # (B, nh, T, hs)

        # 把多头再合并回 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # (B, T, C)
        # 最后做一次输出投影 + dropout（这是“残差分支”的输出）
        y = self.resid_dropout(self.proj(y))
        return y, (weights if need_weights else None)


class MLP(nn.Module):

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT 里的前馈网络（FFN/MLP）通常是：
        # Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> Dropout
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.block_size)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 这是“Pre-LN”结构（LayerNorm 在子层之前）：
        # - 优点：训练更稳定，是 GPT 系列里常见选择
        #
        # 残差连接的直觉：
        # - 每个子层只需要学习“在原表示基础上做一点增量修正”
        # - 梯度传播更顺畅
        a, _ = self.attn(self.ln1(x), attention_mask=attention_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重共享（常见做法）：embedding 和 lm_head 共享参数
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,  # (B, T) token ids
        targets: Optional[torch.Tensor] = None,  # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        训练时：
        - idx 是输入 token 序列 x
        - targets 是“右移一位”的 y（即让模型预测下一个 token）
        - 返回 logits + loss

        推理/生成时：
        - targets=None，只返回 logits
        """
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"序列长度 T={t} 超过 block_size={self.cfg.block_size}，请截断或增大 block_size")

        # 位置编码：用可学习的 position embedding（wpe）
        # pos: (T,) -> wpe(pos): (T, C) -> broadcast 到 (B, T, C)
        pos = torch.arange(0, t, device=idx.device, dtype=torch.long)  # (T,)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        x = self.drop(x)

        # 堆叠 N 层 Transformer Block
        for blk in self.blocks:
            x = blk(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            # CrossEntropy 期望输入形状 (N, vocab)，所以把 (B, T, vocab) 展平到 (B*T, vocab)
            # targets 同理展平成 (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        经典自回归生成：
        - 每次把当前序列 idx 喂给模型
        - 取最后一个位置的 logits（预测下一个 token 的分布）
        - 采样得到 next_id，再拼回 idx，循环 max_new_tokens 次

        这里做了两点常见技巧：
        - temperature：调节分布“尖锐程度”，越小越确定、越大越随机
        - top_k：只保留概率最高的 k 个 token，减少胡言乱语
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 个 token，避免超过位置编码长度
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # 只取最后一个时间步 (B, vocab)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))

            probs = F.softmax(logits, dim=-1)
            # multinomial = 按概率采样（而不是 argmax 贪心）
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
