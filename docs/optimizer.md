# Optimizers

The following optimizers are supported:
- **torch**:
  - `Adam`
  - `AdamW`
- **bitsandbytes**:
  - `Adam`
  - `AdamW`
  - `Adam8Bit`
  - `AdamW8Bit`

> [!NOTE]
> Not all optimizers have been tested with all models/parallel settings. They may or may not work, but this will gradually improve over time.

On Apple Silicon, the bitsandbytes choices require the MPS fork and build described in
[`apple_silicon.md`](apple_silicon.md). The reference MPS recipes use `AdamW8Bit`.
