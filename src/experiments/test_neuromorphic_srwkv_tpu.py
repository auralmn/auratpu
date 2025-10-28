
import torch, json
from neuromorphic_srwkv_tpu import NeuromorphicSRWKVTpu, get_device, DEFAULT_DTYPE

cfg = {
    'embedding_dim': 128,
    'num_heads': 8,
    'attn_mode': 'chunked',
    'block_size_q': 16,
    'block_size_kv': 32,
    'k_winners': 4,
}
m = NeuromorphicSRWKVTpu(cfg)
B, T, D = 2, 33, cfg['embedding_dim']
x = torch.randn(B, T, D, device=get_device(), dtype=DEFAULT_DTYPE)
ids = torch.randint(0, 500, (B, T), device=get_device())
y = m(x, ids)
ok, msg = m.validate()
print(json.dumps({'y_shape': list(y.shape), 'validate': msg}))
