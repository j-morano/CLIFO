from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

MAX_SLICES = 256
SIZE, PATCH, DIM, DEPTH, HEADS = 512, 32, 768, 12, 12
GRID = SIZE // PATCH  # 16x16 patches per B-scan
NUM_GLOBAL = 10       # global tokens appended by the encoder
NUM_QUERIES = 16


def sincos_posemb(h, w, dim, temperature=10000.):
    """2D sine-cosine positional embedding, shape [1, dim, h, w]."""
    gw, gh = torch.meshgrid(torch.arange(w, dtype=torch.float32),
                            torch.arange(h, dtype=torch.float32), indexing='ij')
    omega = 1. / temperature ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    ow = gw.flatten()[:, None] * omega
    oh = gh.flatten()[:, None] * omega
    pos = torch.cat([ow.sin(), ow.cos(), oh.sin(), oh.cos()], dim=1)[None]
    return rearrange(pos, 'b (h w) d -> b d h w', h=h, w=w)


class PatchAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Parameter(sincos_posemb(GRID, GRID, DIM), requires_grad=False)
        self.proj = nn.Conv2d(1, DIM, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):
        x = rearrange(self.proj(x) + self.pos_emb, 'b d h w -> b (h w) d')
        return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """ViT over a single B-scan. Returns [B, patches + global tokens, D]."""
    def __init__(self):
        super().__init__()
        self.input_adapters = nn.ModuleDict({'bscan': PatchAdapter()})
        self.global_tokens = nn.Parameter(torch.zeros(1, NUM_GLOBAL, DIM))
        self.encoder = nn.Sequential(*[Block(DIM, HEADS) for _ in range(DEPTH)])

    def forward(self, x):
        tokens = self.input_adapters['bscan'](x)
        tokens = torch.cat([tokens, self.global_tokens.expand(len(x), -1, -1)], dim=1)
        return self.encoder(tokens)


class Aggregator(nn.Module):
    """Learned queries cross-attend over all slice (and patch) tokens.
    Patch version (use_patches=True) adds an extra self-attention block."""
    def __init__(self, num_queries, heads, use_patches):
        super().__init__()
        self.use_patches = use_patches
        self.queries = nn.Parameter(torch.randn(1, num_queries, DIM))
        self.pos_z_embed = nn.Embedding(MAX_SLICES, DIM)
        self.cross_attn = nn.MultiheadAttention(DIM, heads, batch_first=True)
        self.ln_kv = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, DIM * 4), nn.GELU(), nn.Linear(DIM * 4, DIM))
        if use_patches:
            self.pos_xy_embed = nn.Parameter(torch.randn(1, 1, GRID ** 2, DIM) * 0.02)
            self.ln_q1 = nn.LayerNorm(DIM)
            self.ln_post1 = nn.LayerNorm(DIM)
            self.self_attn = nn.MultiheadAttention(DIM, heads, batch_first=True)
            self.ln_q2 = nn.LayerNorm(DIM)
            self.ln_post2 = nn.LayerNorm(DIM)
        else:
            self.ln_q = nn.LayerNorm(DIM)
            self.ln_post = nn.LayerNorm(DIM)

    def forward(self, x):
        """x: [B, S, D] (lesion) or [B, S, P, D] (patch)."""
        S = x.shape[1]
        pos_ids = torch.linspace(0, MAX_SLICES - 1, steps=S, device=x.device).round().long()
        z = self.pos_z_embed(pos_ids)
        q = self.queries.expand(len(x), -1, -1)

        if not self.use_patches:
            x = x + z[None]
            attn, _ = self.cross_attn(self.ln_q(q), self.ln_kv(x), self.ln_kv(x))
            q = q + attn
            return q + self.mlp(self.ln_post(q))

        x = x + z[None, :, None, :] + self.pos_xy_embed
        x = rearrange(x, 'b s p d -> b (s p) d')
        attn, _ = self.cross_attn(self.ln_q1(q), self.ln_kv(x), self.ln_kv(x))
        q = q + attn
        q = q + self.mlp(self.ln_post1(q))
        # NOTE: three separate calls on purpose. Passing one tensor as q/k/v makes
        # nn.MultiheadAttention use a fused projection, changing bf16 numerics.
        attn, _ = self.self_attn(self.ln_q2(q), self.ln_q2(q), self.ln_q2(q))
        q = q + attn
        return q + self.mlp(self.ln_post2(q))


class CLIFO(nn.Module):
    """Takes volumes [S, H, W] (numpy) and returns pooled features [N, 1536]."""
    def __init__(self, device='cuda', weights='_weights/CLIFO_B.pth', bridge='_weights/CLIFO_VFE.pth'):
        super().__init__()
        self.device = device
        self.vm = Encoder()
        msg = self.vm.load_state_dict(
            torch.load(weights, map_location=device, weights_only=False)['model'], strict=False)
        print('Missing keys:', msg.missing_keys)
        self.patch_aggregator = Aggregator(NUM_QUERIES, 8, use_patches=True)
        self.lesion_aggregator = Aggregator(2, 4, use_patches=False)
        for m in (self.patch_aggregator, self.lesion_aggregator):
            m.to(torch.bfloat16)
        ckpt = torch.load(bridge, map_location=device, weights_only=False)
        self.patch_aggregator.load_state_dict(ckpt['patch_aggregator'])
        self.lesion_aggregator.load_state_dict(ckpt['lesion_aggregator'])
        self.norm = nn.LayerNorm(DIM, eps=1e-6)
        self.eval().to(device)

    def to_tensor(self, x):
        """[S, H, W] numpy -> [1, 1, S, SIZE, SIZE] float in [0, 1]."""
        x = torch.from_numpy(x).float()
        if x.max() > 255:
            x = x / 65535.0
        elif x.max() > 1.0:
            x = x / 255.0
        x = x[None, None]
        S = min(x.shape[2], MAX_SLICES)
        if x.shape[2:] != (S, SIZE, SIZE):
            x = F.interpolate(x, size=(S, SIZE, SIZE), mode='trilinear', align_corners=False)
        return x

    @torch.no_grad()
    def encode(self, x, output='features'):
        """x: [B, 1, S, H, W] -> pooled [B, 1536] ('features'), raw patch tokens
        [B, S, 256, 768] ('patches'), (features, patches) ('both'), or
        (features, aggregated tokens [B, 18, 768], patches) ('all')."""
        B, S = x.shape[0], x.shape[2]
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            x = rearrange(x.to(self.device).to(torch.bfloat16), 'b c s h w -> (b s) c h w')
            tokens = self.vm(x)                                   # [B*S, P+G, D]
            patch_tokens = rearrange(tokens[:, :-NUM_GLOBAL], '(b s) p d -> b s p d', b=B, s=S)
            if output == 'patches':
                return patch_tokens.float()
            lesion = rearrange(tokens[:, -NUM_GLOBAL:][:, 4], '(b s) d -> b s d', b=B, s=S)
            feats = torch.cat([self.patch_aggregator(patch_tokens),
                               self.lesion_aggregator(lesion)], dim=1).float()
        agg = self.norm(feats)                                        # [B, 18, D]
        feats = torch.cat([agg[:, :NUM_QUERIES].mean(1), agg[:, NUM_QUERIES:].mean(1)], dim=1)
        if output == 'both':
            return feats, patch_tokens.float()
        if output == 'all':
            return feats, agg, patch_tokens.float()
        return feats

    def forward(self, volumes, output='features'):
        """volumes: [S, H, W] array, a single B-scan [H, W], or a list of them.
        output='features': pooled global features [N, 1536] (default).
        output='patches':  list of N raw patch-token tensors [S_i, 256, 768]
                           (16x16 patch grid per B-scan, row-major; encoder only).
        output='both':     (features [N, 1536], patches list), in a single encoder pass.
        output='all':      (features [N, 1536], tokens [N, 18, 768], patches list).
                           tokens = normed aggregator outputs (16 patch queries + 2 lesion
                           queries) whose two means give the 1536-d features."""
        if isinstance(volumes, np.ndarray):
            volumes = [volumes]
        volumes = [v[None] if v.ndim == 2 else v for v in volumes]  # single B-scan -> 1 slice
        tensors = [self.to_tensor(v) for v in volumes]
        # One batch per distinct slice count; original order restored.
        groups = {}
        for i, t in enumerate(tensors):
            groups.setdefault(t.shape[2], []).append(i)
        out = [None] * len(tensors)
        for idxs in groups.values():
            res = self.encode(torch.cat([tensors[i] for i in idxs], dim=0), output)
            for j, i in enumerate(idxs):
                out[i] = tuple(r[j] for r in res) if output in ('both', 'all') else res[j]
        if output == 'patches':
            return out
        if output in ('both', 'all'):
            cols = list(zip(*out))                    # one column per returned component
            return (*[torch.stack(c) for c in cols[:-1]], list(cols[-1]))  # patches stay a list
        return torch.stack(out)


if __name__ == "__main__":
    from tqdm import tqdm
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    NUM_VOLUMES = 3  # volumes processed per forward call
    model = CLIFO()
    path = Path('./_datasets/Classification_3D/OCTAVE/test')

    # --- FEATURE EXTRACTION + t-SNE ---
    features, targets = [], []
    for idx, class_dir in enumerate(sorted(path.iterdir())):
        print(f"Processing class: {class_dir.name}")
        files = sorted(class_dir.iterdir())
        for start in tqdm(range(0, len(files), NUM_VOLUMES)):
            chunk = files[start:start + NUM_VOLUMES]
            features.append(model([np.load(f)['vol'] for f in chunk]).cpu())  # [len(chunk), F]
            targets.extend([idx] * len(chunk))
    features = torch.cat(features)
    print("Features shape:", features.shape, "| Targets:", len(targets))

    features_2d = TSNE(n_components=2, random_state=42).fit_transform(features.numpy())
    plt.figure(figsize=(10, 8))
    plt.colorbar(plt.scatter(features_2d[:, 0], features_2d[:, 1], c=targets, cmap='tab10', alpha=0.7),
                 ticks=range(len(set(targets))))
    plt.title('t-SNE of CLIFO Features')
    plt.show()

    # --- BATCH CONSISTENCY CHECK (equal and mixed slice counts) ---
    files = [f for c in sorted(path.iterdir()) for f in sorted(c.iterdir())][:4]
    vols = [np.load(f)['vol'] for f in files]
    for volumes in (vols, [vols[0], vols[1][:15], vols[2], vols[3][:10]]):
        print("\n--- Batch consistency check ---")
        print("Slices per volume:", [v.shape[0] for v in volumes])
        single = torch.cat([model(v).cpu() for v in volumes])   # one volume per call
        batched = model(volumes).cpu()                          # whole list in one call
        cos = F.cosine_similarity(single, batched, dim=1)
        print("max abs diff:", (single - batched).abs().max().item(), "| min cos:", cos.min().item())
        assert cos.min() > 0.99, "Batched != single"
        rev = model(volumes[::-1]).cpu()                        # reversed in -> reversed out
        assert F.cosine_similarity(batched.flip(0), rev, dim=1).min() > 0.99, "Output order not preserved"
        cross = F.cosine_similarity(batched[0:1], batched[1:], dim=1)
        assert cross.max() < cos.min(), "Volumes look identical; check for batch mixing"
        print("OK: batched inference is consistent")
