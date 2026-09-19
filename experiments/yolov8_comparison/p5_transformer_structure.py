"""Independent P5 Transformer ablation on the frozen P2 task-fusion graph."""
import math
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.utils import RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from p2_task_structure import P2TaskModel


class AttentionProducts(nn.Module):
    """Explicit attention products; custom profiling counts both matrix products."""
    def forward(self, q, k, v):
        # FP32 softmax for AMP stability; QK and AV retain autocast behavior.
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        return weights @ v


def count_attention_products(module, inputs, output):
    q, k, v = inputs
    b, heads, n, d = q.shape
    module.total_ops += 2 * b * heads * n * k.shape[-2] * d


class P5Transformer(nn.Module):
    def __init__(self, channels=192, dim=96, heads=4):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.reduce = nn.Conv2d(channels, dim, 1, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention = AttentionProducts()
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.expand = nn.Conv2d(dim, channels, 1, bias=False)

    def position(self, h, w, device, dtype):
        # Dynamic 2D sine/cosine positions, including rectangular validation input.
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        omega = torch.arange(self.dim // 4, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (self.dim // 4)))
        x, y = x.flatten().float()[:, None] * omega, y.flatten().float()[:, None] * omega
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).to(dtype)[None]

    def forward(self, x):
        b, _, h, w = x.shape
        tokens = self.reduce(x).flatten(2).transpose(1, 2)
        z = self.norm1(tokens) + self.position(h, w, x.device, tokens.dtype)
        qkv = self.qkv(z).reshape(b, h*w, 3, self.heads, self.dim // self.heads).permute(2, 0, 3, 1, 4)
        attn = self.attention(*qkv.unbind(0)).transpose(1, 2).reshape(b, h*w, self.dim)
        tokens = tokens + self.proj(attn)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return x + self.expand(tokens.transpose(1, 2).reshape(b, self.dim, h, w))


class P5TransformerModel(P2TaskModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=False)
        if self.yaml.get('p5_transformer_design') != 'post_sppf_d96_h4_ffn192_v1':
            raise ValueError('Expected independent P5 Transformer YAML.')
        original = self.model[7]
        wrapped = nn.Sequential(original, P5Transformer())
        wrapped.i, wrapped.f = original.i, original.f
        wrapped.type = 'SPPF+P5Transformer'
        wrapped.np = sum(p.numel() for p in wrapped.parameters())
        self.model[7] = wrapped
        if verbose:
            self.info()


class P5TransformerTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P5TransformerModel(cfg, ch=self.data['channels'], nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights is not None:
            model.load(weights)
        return model


class P5TransformerYOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping['detect']['model'] = P5TransformerModel
        mapping['detect']['trainer'] = P5TransformerTrainer
        return mapping
