# llm002：从 0 实现最小 GPT（支持 tiktoken）

这份代码的目标是：用尽量少的代码把 **GPT（自回归 Transformer）训练闭环**跑通，并且在关键位置写了注释，方便你按“张量形状/数据流”去理解。

项目结构（当前版本）：
- `src/gpt.py`：模型本体（注意力、Block、GPT、生成）
- `src/train.py`：训练脚本（tiktoken 编码、batch 采样、训练/评估/保存、训练后生成）
- `the-verdict.txt`：示例语料

## 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

> 说明：如果你已经装了 `torch`，只装 `tiktoken` 也可以：`pip install tiktoken`

## 快速上手：跑一个最小训练

你已经把 `the-verdict.txt` 放进仓库后，可以直接跑训练（先用很小的迭代数验证 pipeline）：

```bash
python src/train.py --data the-verdict.txt --max_iters 30 --eval_interval 10 --eval_iters 5 --batch_size 8 --block_size 64 --n_layer 2 --n_head 4 --n_embd 128
```

正常训练（CPU 会慢一些，先跑 2000 次看看 loss 下降）：

```bash
python src/train.py --data the-verdict.txt --max_iters 2000
```

继续训练（从 `checkpoints/ckpt.pt` 恢复）：

```bash
python src/train.py --data the-verdict.txt --resume --max_iters 4000
```

训练结束后脚本会自动用 `--prompt` 生成一段文本；你也可以自定义：

```bash
python src/train.py --data the-verdict.txt --resume --max_iters 4000 --prompt "I had always thought" --max_new_tokens 200
```

## 概念速览（建议边看边对照代码）

下面这些概念在 `src/gpt.py` / `src/train.py` 都能找到对应实现位置。

### Token、词表（vocab）
- **Token**：文本被 tokenizer 切分后的最小单位（不是“字/词”，而是 BPE 子词片段）
- **vocab_size**：词表大小；模型最后输出 logits 的维度就是 `vocab_size`

### 自回归语言模型（next-token prediction）
- 训练目标是最大化：
  - `p(t1, t2, ..., tT) = ∏_i p(t_i | t_<i)`（自回归分解）
- 具体到训练数据就是：
  - 输入 `x = [t0, t1, ..., t(T-1)]`
  - 标签 `y = [t1, t2, ..., tT]`（也就是“右移一位”）
- 对应代码：`src/train.py:get_batch` + `src/gpt.py:GPT.forward` 里的 `cross_entropy`

### 注意力（Q/K/V）与多头
- 先把输入 embedding 线性映射成 Q/K/V
- 打分：`QK^T / sqrt(d)`，softmax 得到权重
- 多头：把通道维拆成 `n_head` 份并行做注意力，最后再拼回去
- 对应代码：`src/gpt.py:CausalSelfAttention.forward`

### 因果 Mask（为什么“不能看未来”）
- 自回归生成时，位置 i 只能用到 `t_{<=i}`（也就是只看当前位置及之前的 token）
- 所以注意力矩阵 (T×T) 必须是**下三角可见**
- 对应代码：`src/gpt.py` 里 `causal_mask` + `masked_fill(~causal, ...)`

### Transformer Block（残差 + LayerNorm + MLP）
- GPT 常用 **Pre-LN**：LayerNorm 放在子层之前
- 两条残差分支：
  - `x = x + Attn(LN(x))`
  - `x = x + MLP(LN(x))`
- 对应代码：`src/gpt.py:Block.forward`

### 生成（sampling）
- 每次取最后一个位置 logits 作为 next-token 分布
- **temperature** 控制随机性
- **top-k** 只保留概率最高的 k 个 token
- 对应代码：`src/gpt.py:GPT.generate`

## 教程：推荐的学习顺序（按“数据流”走）

1. **先读训练数据怎么喂进去**
   - 从 `src/train.py:get_batch` 开始，搞懂 x/y 右移
   - 再看 `src/gpt.py:GPT.forward` 如何算 logits/loss

2. **再读注意力的张量形状**
   - `src/gpt.py:CausalSelfAttention.forward`
   - 重点看：`(B,T,C)` -> `(B,nh,T,hs)` -> `(B,nh,T,T)` -> 合并回 `(B,T,C)`

3. **最后看生成**
   - `src/gpt.py:GPT.generate`：循环采样并拼接 token

1. **数据管道**
   - 用 `tiktoken` 把大文本 encode 成 token 序列
   - 实现“滑动窗口”采样：输入 `x=ids[i:i+T]`，标签 `y=ids[i+1:i+T+1]`
   - 用 `DataLoader` 做 batch、shuffle、多进程加载（可选）

2. **训练循环**
   - 优化器：`AdamW`
   - 学习率策略：warmup + cosine / linear decay
   - 梯度裁剪：`clip_grad_norm_`
   - 混合精度（可选）：`torch.cuda.amp`

3. **评估与推理**
   - 指标：perplexity（由 cross-entropy 换算）
   - 生成策略：temperature / top-k / top-p（当前提供了 top-k）
   - 保存/加载：`state_dict` + 配置

4. **模型能力/稳定性增强**
   - 更合理的初始化与残差缩放
   - 更大模型时的 checkpoint、gradient accumulation
   - 长上下文：RoPE/ALiBi（进阶）

## 常见坑（你很快会遇到）

- **mask 形状/广播**：注意 (B,T)、(B,1,1,T)、(B,1,T,T) 的差别
- **数值稳定**：被 mask 的位置用 `torch.finfo(dtype).min`，避免 `-inf` 在某些 dtype 上的问题
- **block_size**：训练/生成时都要截断到 block_size

## 下一步练习（建议按优先级）

1. **把 top-k 改成 top-p（nucleus sampling）**：更常用
2. **加学习率调度（warmup + cosine）**：训练更稳更快
3. **加 AMP 混合精度（cuda）**：显存更省、速度更快
4. **加入 KV cache**：生成速度会明显提升（进阶）


