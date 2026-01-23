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
