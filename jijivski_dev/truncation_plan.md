# Truncation Plan v2 (SFT)

## Goals
- In SFT, compute per-token NLL and optional PPL logging.
- Add truncation/masking strategies driven by environment variables.
- Default behavior is naive; protections are opt-in for later iterations.

## Non-goals / Compatibility
- Do not support `use_dft_loss`, `use_eaft_loss`, `enable_liger_kernel`, `use_kt`.
  If any is enabled, truncation is auto-disabled with a warning.
- Multimodal processors are ignored for now.

## Unified env vars
```bash
TRUNCATION_ENABLE=false

# threshold selection
TRUNCATION_THRESHOLD_MODE=absolute   # absolute | ratio
TRUNCATION_THRESHOLD=2.0            # used when absolute
TRUNCATION_RATIO=0.1                # used when ratio (top-p of losses)

# masking strategy
TRUNCATION_MASK_MODE=after_first_high  # after_first_high | mask_high_only | mask_random
TRUNCATION_INCLUDE_TRIGGER=true

# protections (defaults = naive)
TRUNCATION_MIN_PREFIX_TOKENS=0
TRUNCATION_MAX_MASK_RATIO=1.0
TRUNCATION_MIN_HIGH_RUN=1

# random mask params
TRUNCATION_RANDOM_RATIO=0.1

# logging / debug
TRUNCATION_LOG_EVERY_N_STEPS=50
TRUNCATION_LOG_DETAILED=false
TRUNCATION_LOG_DETAILED_EVERY_N_STEPS=100
TRUNCATION_DEBUG=false
```

## Masking logic (core)
1. Compute per-token loss on shifted labels (ignore `IGNORE_INDEX`).
2. Build `high_loss_positions` via `TRUNCATION_THRESHOLD_MODE`:
   - `absolute`: `loss > TRUNCATION_THRESHOLD`
   - `ratio`: compute threshold from valid losses, then `loss >= threshold`
3. Apply `TRUNCATION_MASK_MODE`:
   - `after_first_high`: find first position with `TRUNCATION_MIN_HIGH_RUN` consecutive highs.
     Start index = trigger index (or +1 if `TRUNCATION_INCLUDE_TRIGGER=false`).
     Then clamp with protections:
       - `start_idx >= TRUNCATION_MIN_PREFIX_TOKENS`
       - `start_idx >= ceil(valid_len * (1 - TRUNCATION_MAX_MASK_RATIO))`
     If `start_idx >= valid_len`, no truncation for this sequence.
   - `mask_high_only`: mask only high-loss tokens.
     If `TRUNCATION_MIN_PREFIX_TOKENS > 0`, clear highs in the prefix.
   - `mask_random`: randomly mask valid tokens by `TRUNCATION_RANDOM_RATIO`,
     excluding the prefix if `TRUNCATION_MIN_PREFIX_TOKENS > 0`.

## Strategy ladder (experiments)
- Baseline: `TRUNCATION_ENABLE=false`
- A1 (naive): `absolute + after_first_high`, no protections (defaults)
- A2 (protected): set `TRUNCATION_MIN_PREFIX_TOKENS`, `TRUNCATION_MAX_MASK_RATIO`, and `TRUNCATION_MIN_HIGH_RUN`
- A3: `absolute + mask_high_only`
- A4: `mask_random`

## Logging policy
- Always log batch-level ratios (every `TRUNCATION_LOG_EVERY_N_STEPS`):
  - `truncation/batch_ratio`, `truncation/mean_seq_ratio`,
    `truncation/max_seq_ratio`, `truncation/truncated_seqs`
  - Optional: `truncation/raw_loss`, `truncation/raw_ppl` (pre-mask)
- Detailed logging is optional and sampled by step interval.
- Use rank-0 guard to avoid duplicate file writes.

## Quick verification notes (for min-prefix / max-mask)
Example 1:
- losses = [0.1, 0.2, 5.0, 5.1, 0.3]
- threshold = 4.0, min_high_run=2, include_trigger=true
- min_prefix=3 => trigger idx=2, start_idx=max(2, 3)=3
- result: tokens from idx=3 masked

Example 2:
- valid_len=10, max_mask_ratio=0.5
- min_keep=ceil(10 * (1 - 0.5))=5
- start_idx is clamped to >= 5 (so mask at most 50%)




TRUNCATION_ENABLE=false              # 是否启用截断

# 阈值选择方式
TRUNCATION_THRESHOLD_MODE=absolute   # absolute (绝对值) | ratio (比例)
TRUNCATION_THRESHOLD=2.0            # 绝对阈值模式下使用的值
TRUNCATION_RATIO=0.1                # 比例模式下使用的值 (Loss 的 Top-P 比例)

# 掩码策略
TRUNCATION_MASK_MODE=after_first_high  # 策略：首个高点后截断 | 仅掩码高点 | 随机掩码
TRUNCATION_INCLUDE_TRIGGER=true        # 是否包含触发点本身

# 保护机制 (默认值为 0/1.0，即原始简单模式)
TRUNCATION_MIN_PREFIX_TOKENS=0       # 最小保留的前缀 Token 数
TRUNCATION_MAX_MASK_RATIO=1.0        # 最大掩码比例限制
TRUNCATION_MIN_HIGH_RUN=1            # 触发截断所需的连续高 Loss Token 数量

# 随机掩码参数
TRUNCATION_RANDOM_RATIO=0.1          # 随机掩码的比例

# 日志与调试
TRUNCATION_LOG_EVERY_N_STEPS=50            # 批量日志记录频率
TRUNCATION_LOG_DETAILED=false              # 是否记录详细日志
TRUNCATION_LOG_DETAILED_EVERY_N_STEPS=100  # 详细日志采样步长
TRUNCATION_DEBUG=false                     # 调试模式



  Ablation set (建议顺序)

  1. Baseline（对照）

  - 目的：确认所有指标变化来自 truncation
  - 配置：TRUNCATION_ENABLE=false

  2. A1 Naive: absolute + after_first_high

  - 目的：验证“高 loss 后截断”是否整体有正向信号
  - 配置：

  TRUNCATION_ENABLE=true
  TRUNCATION_THRESHOLD_MODE=absolute
  TRUNCATION_THRESHOLD=2.0
  TRUNCATION_MASK_MODE=after_first_high
  TRUNCATION_INCLUDE_TRIGGER=true
  TRUNCATION_MIN_PREFIX_TOKENS=0
  TRUNCATION_MAX_MASK_RATIO=1.0
  TRUNCATION_MIN_HIGH_RUN=1

  - 预期：mask 比例可能偏高，容易伤前部；这是“最简单验证有效性”的版本。

  3. A2 Protected: absolute + after_first_high + 保护

  - 目的：缓解“前部高 loss 更高”的问题，避免学不到前缀
  - 配置：

  TRUNCATION_ENABLE=true
  TRUNCATION_THRESHOLD_MODE=absolute
  TRUNCATION_THRESHOLD=2.0
  TRUNCATION_MASK_MODE=after_first_high
  TRUNCATION_INCLUDE_TRIGGER=true
  TRUNCATION_MIN_PREFIX_TOKENS=32        # 或 0.2*len 的等价（先固定）
  TRUNCATION_MAX_MASK_RATIO=0.5
  TRUNCATION_MIN_HIGH_RUN=2

  - 预期：更稳，mask 比例更可控。

  4. A3 Ratio: ratio + after_first_high

  - 目的：测试对尺度不敏感的阈值（loss 分布变化时更稳）
  - 配置：

  TRUNCATION_ENABLE=true
  TRUNCATION_THRESHOLD_MODE=ratio
  TRUNCATION_RATIO=0.1
  TRUNCATION_MASK_MODE=after_first_high
  TRUNCATION_MIN_PREFIX_TOKENS=32
  TRUNCATION_MAX_MASK_RATIO=0.5
  TRUNCATION_MIN_HIGH_RUN=2

  - 预期：比 absolute 更“自适应”，但分布变化可能导致截断比例变化。

  5. A4 Mask‑high‑only

  - 目的：只移除高 loss token，不截断后续，衡量“噪声过滤”本身的价值
  - 配置（绝对阈值或 ratio 均可）：

  TRUNCATION_ENABLE=true
  TRUNCATION_THRESHOLD_MODE=absolute
  TRUNCATION_THRESHOLD=2.0
  TRUNCATION_MASK_MODE=mask_high_only
  TRUNCATION_MIN_PREFIX_TOKENS=32

  6. A5 Random mask（负对照）

  - 目的：排除“降低有效 token 数”带来的假收益
  - 配置：

  TRUNCATION_ENABLE=true
  TRUNCATION_MASK_MODE=mask_random
  TRUNCATION_RANDOM_RATIO=0.1
  TRUNCATION_MIN_PREFIX_TOKENS=32


  7. A6 Mask‑high‑only

  - 目的：在前面绝对值或者ratio的基础上, 看看是否要带上困惑本身,




  为什么这样设计

  - A1 vs Baseline：验证“高 loss 截断”是否有用。
  - A2 vs A1：验证“保护前缀 + 连续触发”是否抑制过度截断。
  - A3：验证“比例阈值”是否比绝对阈值稳定。
  - A4：验证“局部去噪”是否比“后缀截断”更安全。
  - A5：负对照，证明收益不是来自“纯减少训练 token”。
  - A6：验证是否要带上触发的token

  建议的观察指标

  - 训练 loss / eval loss / eval ppl
  - truncation 比例（truncation/batch_ratio、mean_seq_ratio）
  - 训练稳定性（是否出现 0 loss 过多、梯度异常）

  如果你愿意，我们还可以加一个 A1b：TRUNCATION_INCLUDE_TRIGGER=false 来判断“触发点是否保留”对学习影响是否显著。
   我认为学个5epo应该能看出东西来, 把test加上, 然后看看这里的应该如何设置