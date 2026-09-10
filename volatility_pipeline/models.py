from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class SelectiveSSMBlock(nn.Module):
    """Mamba-style selective SSM block used in the original experiments."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        inner = cfg.expand * cfg.d_model
        self.inner = inner
        self.dt_rank = math.ceil(cfg.d_model / 16)
        self.in_proj = nn.Linear(cfg.d_model, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, cfg.d_conv, groups=inner, padding=cfg.d_conv - 1)
        self.x_proj = nn.Linear(inner, self.dt_rank + cfg.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        a = torch.arange(1, cfg.d_state + 1, dtype=torch.float32).repeat(inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        values, gate = self.in_proj(x).chunk(2, dim=-1)
        values = F.silu(self.conv1d(values.transpose(1, 2))[..., :length].transpose(1, 2))
        return self.out_proj(self._ssm(values) * F.silu(gate))

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        a = -torch.exp(self.a_log.float())
        delta, b, c = self.x_proj(x).split([self.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        da = torch.exp(torch.einsum("bld,dn->bldn", delta, a))
        dbu = torch.einsum("bld,bln,bld->bldn", delta, b, x)
        return self.parallel_scan(da, dbu, c, self.cfg.scan_chunk) + x * self.skip.float()

    @staticmethod
    def _chunk_scan(a: torch.Tensor, bx: torch.Tensor) -> torch.Tensor:
        """Inclusive Hillis-Steele scan for h_t=a_t*h_(t-1)+b_t, h_-1=0."""
        length = a.shape[1]
        shift = 1
        while shift < length:
            previous_a = F.pad(a, (0, 0, 0, 0, shift, 0), value=1.0)[:, :length]
            previous_b = F.pad(bx, (0, 0, 0, 0, shift, 0), value=0.0)[:, :length]
            bx = a * previous_b + bx
            a = a * previous_a
            shift *= 2
        return bx

    @classmethod
    def parallel_scan(
        cls, da: torch.Tensor, dbu: torch.Tensor, c: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        length = da.shape[1]
        chunk_size = min(chunk_size, length)
        carry = None
        outputs = []
        for start in range(0, length, chunk_size):
            a = da[:, start : start + chunk_size]
            bx = dbu[:, start : start + chunk_size]
            cc = c[:, start : start + chunk_size]
            state = cls._chunk_scan(a, bx)
            if carry is not None:
                state = state + torch.cumprod(a, dim=1) * carry.unsqueeze(1)
            outputs.append(torch.einsum("bldn,bln->bld", state, cc))
            carry = state[:, -1]
        return torch.cat(outputs, dim=1)

    @staticmethod
    def sequential_states(da: torch.Tensor, dbu: torch.Tensor) -> torch.Tensor:
        """Reference recurrence used by unit tests, not by training."""
        state = torch.zeros_like(dbu[:, 0])
        outputs = []
        for step in range(da.shape[1]):
            state = da[:, step] * state + dbu[:, step]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class ResidualSSM(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ssm = SelectiveSSMBlock(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(self.norm(x))


class BiMambaStyle(nn.Module):
    """Custom bidirectional Mamba-style corrector (not the official Mamba implementation)."""

    def __init__(self, cfg: ExperimentConfig, n_features: int, n_scans: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_scans = cfg.n_scans if n_scans is None else n_scans
        self.embedding = nn.Linear(n_features, cfg.d_model)
        self.scans = nn.ModuleList(
            [nn.ModuleList([ResidualSSM(cfg) for _ in range(cfg.n_layers)]) for _ in range(self.n_scans)]
        )
        self.time_mix = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.seq_len),
                    nn.Linear(cfg.seq_len, cfg.hidden),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden, cfg.seq_len),
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in range(self.cfg.n_layers):
            aggregate = x
            for scan in range(self.n_scans):
                if scan % 2 == 1:
                    scanned = self.scans[scan][layer](x.flip(1)).flip(1)
                else:
                    scanned = self.scans[scan][layer](x)
                aggregate = aggregate + scanned
            x = aggregate
            x = x + self.time_mix[layer](x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.final_norm(x))


class RecurrentCorrector(nn.Module):
    def __init__(self, kind: str, n_features: int, hidden: int):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(n_features, hidden, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden * 2, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.projection(out)


class WindowMLP(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(n_features * seq_len, hidden), nn.GELU(), nn.Linear(hidden, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)


class PatchTransformer(nn.Module):
    def __init__(self, n_features: int, seq_len: int, width: int, patch_len: int = 10):
        super().__init__()
        patch_len = math.gcd(seq_len, patch_len)
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        width = max(4, (width // 4) * 4)
        self.input = nn.Linear(patch_len * n_features, width)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, width))
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(width, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, features = x.shape
        patches = x.reshape(batch, self.n_patches, self.patch_len * features)
        encoded = self.encoder(self.input(patches) + self.position)
        return self.head(encoded.mean(1)).unsqueeze(1)


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)
        self.time = nn.Sequential(nn.Linear(seq_len, hidden), nn.GELU(), nn.Linear(hidden, seq_len))
        self.feature = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.feature(self.norm2(x))


class TSMixer(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden: int):
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock(seq_len, n_features, hidden) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x.mean(1, keepdim=True)


class ITransformer(nn.Module):
    def __init__(self, n_features: int, seq_len: int, width: int):
        super().__init__()
        width = max(4, (width // 4) * 4)
        self.embedding = nn.Linear(seq_len, width)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.projection = nn.Linear(width, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        return self.projection(self.encoder(self.embedding(tokens))).transpose(1, 2)


def n_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_corrector(cfg: ExperimentConfig, name: str, n_features: int, n_scans: int | None = None) -> nn.Module:
    if name == "bimamba":
        return BiMambaStyle(cfg, n_features, n_scans=n_scans)
    target = n_parameters(BiMambaStyle(cfg, n_features, n_scans=n_scans))

    def build(hidden: int) -> nn.Module:
        choices = {
            "lstm": lambda: RecurrentCorrector("lstm", n_features, hidden),
            "gru": lambda: RecurrentCorrector("gru", n_features, hidden),
            "mlp": lambda: WindowMLP(n_features, cfg.seq_len, hidden),
            "patchtf": lambda: PatchTransformer(n_features, cfg.seq_len, hidden),
            "tsmixer": lambda: TSMixer(n_features, cfg.seq_len, hidden),
            "itransformer": lambda: ITransformer(n_features, cfg.seq_len, hidden),
        }
        if name not in choices:
            raise ValueError(f"unknown corrector={name}")
        return choices[name]()

    low, high = 4, 1024
    while low < high:
        middle = (low + high) // 2
        if n_parameters(build(middle)) < target:
            low = middle + 1
        else:
            high = middle
    candidates = [max(4, low - 1), low]
    hidden = min(candidates, key=lambda value: abs(n_parameters(build(value)) - target))
    return build(hidden)


class AnchoredCorrector(nn.Module):
    def __init__(
        self,
        cfg: ExperimentConfig,
        n_features: int,
        n_core: int,
        core_coef: tuple[torch.Tensor | object, float],
        corrector: str = "bimamba",
        freeze_core: bool = True,
        anchor: bool = True,
        use_core: bool = True,
        n_scans: int | None = None,
        zero_init_readout: bool | None = None,
    ):
        super().__init__()
        self.backbone = build_corrector(cfg, corrector, n_features, n_scans=n_scans)
        self.use_core = bool(use_core)
        # ``anchor`` is kept for compatibility with the earlier experiments.
        # Matched-control runs pass this setting explicitly for both modes.
        self.zero_init_readout = bool(anchor if zero_init_readout is None else zero_init_readout)
        self.readout = nn.Linear(n_features, 1)
        self.core = nn.Linear(n_core, 1)
        weights, bias = core_coef
        with torch.no_grad():
            self.core.weight.copy_(torch.as_tensor(weights, dtype=torch.float32).reshape(1, -1))
            self.core.bias.fill_(float(bias))
            if self.zero_init_readout:
                nn.init.zeros_(self.readout.weight)
                nn.init.zeros_(self.readout.bias)
        if freeze_core:
            self.core.weight.requires_grad_(False)
            self.core.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
        correction_sequence = self.backbone(x)
        correction = self.readout(correction_sequence.mean(1))
        if self.use_core:
            prediction = self.core(core) + correction
        else:
            # A genuine direct/pure-deep model: core inputs and coefficients do
            # not contribute to the forward pass.
            prediction = correction
        return prediction.unsqueeze(-1)
