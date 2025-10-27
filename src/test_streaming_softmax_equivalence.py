
import torch, json
from neuromorphic_srwkv_tpu import NeuromorphicSRWKVTpu, get_device, DEFAULT_DTYPE

def run(cfg_overrides):
    cfg = {
        'embedding_dim': 64,
        'num_heads': 4,
        'block_size_q': 16,
        'block_size_kv': 16,
        'spike_threshold': 0.0,  # disable spike gating for equivalence check
        'decay_factor': 0.0,     # disable temporal mixing effects
    }
    cfg.update(cfg_overrides)
    m = NeuromorphicSRWKVTpu(cfg)
    B, T, D = 2, 33, cfg['embedding_dim']
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=get_device(), dtype=DEFAULT_DTYPE)
    ids = torch.randint(0, 10, (B, T), device=get_device())

    # Bypass neuromorphic path by feeding x after projections directly into attention blocks equivalently
    # For a fair comparison we set attention over identical tensors.
    with torch.no_grad():
        temporal = x  # identical inputs to attention

        # Heads
        def split(t):
            H = m.num_heads
            Hd = m.head_dim
            return t.view(B, T, H, Hd).transpose(1, 2).contiguous()

        qh = split(temporal)
        kh = split(temporal)
        vh = split(temporal)

        # Streaming
        ys = m.streaming_attention(qh, kh, vh, m.block_size_q, m.block_size_kv)
        # Dot
        yd = m.scaled_dot_attention(qh, kh, vh)

        # Merge heads
        ys = ys.transpose(1,2).contiguous().view(B, T, D)
        yd = yd.transpose(1,2).contiguous().view(B, T, D)

        diff = (ys.float() - yd.float()).abs()
        rel = diff / (yd.float().abs() + 1e-6)
        return {
            'max_abs_diff': float(diff.max().item()),
            'mean_rel_err': float(rel.mean().item()),
            'shapes_equal': tuple(ys.shape) == tuple(yd.shape)
        }

res = run({'attn_mode': 'streaming'})
print(json.dumps(res, indent=2))
