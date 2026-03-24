# Legacy: Backprop Prefix CPU Offload (Removed)

This file stores the removed CPU-offload path from `backprop` so it can be restored later.

## 1) `BackpropConfig` field removed from `backprop/base.py`

```python
@dataclass
class BackpropConfig:
    top_frac: float = 0.25
    query_chunk: int = 4096
    logit_chunk: int = 64
    use_grad_checkpoint: bool = True
    offload_prefix_cpu: bool = True
```

## 2) Conditional CPU offload removed from `backprop/streaming.py`

Removed from `_run_frozen_prefix(...)` tail:

```python
if self.config.offload_prefix_cpu:
    return hidden.cpu(), position_ids.cpu()
return hidden, position_ids
```

Current behavior keeps prefix tensors on GPU:

```python
return hidden, position_ids
```

## 3) Callsite/config references removed

### `config.py`

Removed in `build_backprop_config()`:

```python
offload_prefix_cpu=True,
```

### `sft_primer/main.py`

Removed in `BackpropConfig(...)`:

```python
offload_prefix_cpu=True,
```

### `backprop/test_grad_parity.py`

Removed explicit arg usage:

```python
cfg_stream = BackpropConfig(top_frac=args.top_frac, logit_chunk=64, offload_prefix_cpu=True)
cfg_lora = BackpropConfig(top_frac=1.0, logit_chunk=64, offload_prefix_cpu=False)
```

### `trainer.py`

Removed logging placeholder/value:

```python
"backprop=streaming top_frac=%.3f grad_ckpt=%s offload_prefix_cpu=%s"
...
bp_cfg.offload_prefix_cpu,
```

## Restore guide

To restore CPU offload behavior:
1. Add `offload_prefix_cpu: bool = True` back into `BackpropConfig`.
2. Re-add the conditional `hidden.cpu(), position_ids.cpu()` return in `_run_frozen_prefix`.
3. Re-add callsite config args and trainer log fields listed above.
