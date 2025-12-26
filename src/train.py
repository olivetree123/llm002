import argparse
import math
import time
from pathlib import Path
from typing import Tuple

import torch
import tiktoken

# 兼容两种运行方式：
# 1) 直接脚本运行：python src/train.py（此时 sys.path[0] 指向 src/，可以用 `from gpt import ...`）
# 2) 模块方式运行：python -m src.train（此时需要相对导入 `from .gpt import ...`）
try:
    from .gpt import GPT, GPTConfig  # type: ignore
except Exception:  # pragma: no cover
    from gpt import GPT, GPTConfig


def load_tokens(text_path: Path) -> Tuple[torch.Tensor, int]:
    """
    把纯文本文件编码成 token 序列（1D long tensor）。

    - 使用 `tiktoken.get_encoding("gpt2")`：这会给你一个经典的 GPT-2 BPE tokenizer
    - 返回：
      - data: 形状 (N,) 的 token ids（N 是整本文本的 token 数）
      - vocab_size: 词表大小（模型输出 logits 的最后一维）
    """
    text = text_path.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    return data, enc.n_vocab


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    在训练中“抽样”估计 train/val loss。

    说明：
    - 这里不是对整个数据集做遍历（那会很慢），而是随机采样 eval_iters 个 batch 求平均
    - 使用 @torch.no_grad() 避免构建计算图，速度更快、显存更省
    """
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, targets=y)
            losses[k] = loss.detach().float().cpu()
        out[split] = losses.mean().item()
    model.train()
    return out["train"], out["val"]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # 随机选择起点，形成 (B, T) 的输入与标签（右移一位）
    # 注意：必须保证能取到 y 的最后一个位置
    #
    # 为什么 y 要“右移一位”？
    # - 如果 x 是: [t0, t1, t2, t3]
    # - 那么目标 y 是: [t1, t2, t3, t4]
    # 模型在每个位置 i 都在预测“下一个 token”。
    max_start = data.numel() - block_size - 1
    ix = torch.randint(0, max_start, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="the-verdict.txt", help="训练语料路径（txt）")
    p.add_argument("--out_dir", type=str, default="checkpoints", help="checkpoint 输出目录")
    p.add_argument("--seed", type=int, default=1337)

    # 模型超参（先用小模型，确保能跑通）
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # 训练超参
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # 生成
    p.add_argument("--prompt", type=str, default="I had always thought", help="训练后用于生成的 prompt")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)

    # 恢复训练
    p.add_argument("--resume", action="store_true", help="从 out_dir/ckpt.pt 恢复训练")

    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1) 加载并编码 token
    data, vocab_size = load_tokens(data_path)
    if data.numel() < args.block_size + 2:
        raise ValueError(f"语料 token 数太少：N={data.numel()}，但 block_size={args.block_size}。"
                         f"请减小 block_size，或提供更长的文本。")
    n = int(0.9 * data.numel())
    train_data = data[:n]
    val_data = data[n:]

    # 2) 构建模型
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    iter_num = 0
    best_val = math.inf

    if args.resume and ckpt_path.exists():
        # 恢复训练：把模型权重、优化器状态、迭代步数一起恢复
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = int(ckpt.get("iter_num", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[resume] iter={iter_num} best_val={best_val:.4f} from {ckpt_path}")

    # 3) 训练循环
    t0 = time.time()
    model.train()
    while iter_num < args.max_iters:
        if iter_num % args.eval_interval == 0:
            # 定期评估：看看模型是否在“泛化”而不是只记住训练片段
            train_loss, val_loss = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                eval_iters=args.eval_iters,
                device=device,
            )
            dt = time.time() - t0
            print(f"iter {iter_num:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")

            # 保存 best checkpoint
            # 这里只按 val loss 变好来保存，避免每次都写盘
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val": best_val,
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"[save] best checkpoint -> {ckpt_path} (val={best_val:.4f})")

        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(xb, targets=yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：防止梯度爆炸（尤其是刚开始训练或学习率偏大时）
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        iter_num += 1

    # 4) 训练后生成一段看看效果
    # 注意：这是“无缓存”的朴素生成（每步都重新 forward 整个上下文），适合学习与小模型
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n=== sample ===")
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
