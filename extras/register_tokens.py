"""
Register tokens — implemented, NEVER exercised as an experiment.

Status
------
Unlike the other files in extras/, this is not a separate rejected module: it is live code
that still exists inside `BiMamba` (the corrector backbone used throughout notebooks/ and
extras/) in the current, in-use codebase. `Config` carries three fields controlling it --
`use_registers: bool = False`, `n_registers: int = 4`, `reg_control: bool = False` -- and
`BiMamba` implements the full mechanism (evenly-spaced learnable register tokens inserted into
the sequence, a reduce-and-concat readout pulled from their final states, plus a `reg_control`
variant that keeps the same parameter count without actual registers, as a param-matched control).

We could not find any run, in any notebook or backup covering this project's history, that ever
set `use_registers=True` or `reg_control=True`. The feature was built as part of the general
"ablation knobs" set on the BiMamba backbone but was never exercised as a reported ablation --
it is dormant, not "evaluated and rejected". It is reproduced here standalone (excerpted from the
live `BiMamba.__init__`/`_insert`/`forward`) purely for documentation, in case a future run wants
to actually test it.

The excerpt below is NOT standalone-runnable as printed -- it depends on `ResidualBlock`, `RMSNorm`,
`Config`, and the surrounding `BiMamba.__init__`/`forward` control flow in
notebooks/01_main_ablation_tables_I_II_IV.ipynb (or any of the other notebooks: they all carry an
identical copy of this class, unchanged, since register tokens were never toggled on).
"""

# --- relevant __init__ fragment (inside BiMamba.__init__, after self.global_dim = 0) -----------
#
#     self.use_reg = cfg.use_registers
#     self.L_ext = self.seq_len + (cfg.n_registers if self.use_reg else 0)
#     ...
#     if self.use_reg:
#         self.registers = nn.Parameter(torch.randn(cfg.n_registers, cfg.d_model) * 0.02)
#         pos = torch.linspace(0, self.L_ext - 1, cfg.n_registers).round().long()
#         assert len(torch.unique(pos)) == cfg.n_registers, "n_registers too large for seq_len"
#         mask = torch.zeros(self.L_ext, dtype=torch.bool); mask[pos] = True
#         self.register_buffer("reg_mask", mask)
#         self.r = max(1, cfg.d_model // 4); self.reg_reduce = nn.Linear(cfg.d_model, self.r)
#         self.global_dim = cfg.n_registers * self.r
#     elif cfg.reg_control:                    # param-matched control: same head size, no registers
#         self.r = max(1, cfg.d_model // 4); self.reg_reduce = nn.Linear(cfg.d_model, self.r)
#         self.global_dim = cfg.n_registers * self.r

def _insert(self, x):
    # x: [b, seq_len, d_model] -> [b, L_ext, d_model], with `n_registers` learnable tokens
    # spliced in at evenly-spaced positions (`reg_mask`).
    b, S, d = x.shape
    out = torch.empty(b, self.L_ext, d, device=x.device, dtype=x.dtype)
    out[:, self.reg_mask] = self.registers.to(x.dtype).expand(b, -1, -1)
    out[:, ~self.reg_mask] = x
    return out

# --- relevant forward() fragment ----------------------------------------------------------------
#
#     if self.use_reg: x = self._insert(x)          # right after the embedding layer
#     ...                                            # (n_layer bidirectional-scan blocks, unchanged)
#     x = self.norm_f(x)
#     gvec = None
#     if self.use_reg:
#         gvec = self.reg_reduce(x[:, self.reg_mask]).flatten(1)   # reduce-and-concat readout
#         x = x[:, ~self.reg_mask]                                  # strip registers before the head
#     elif self.cfg.reg_control:
#         gvec = self.reg_reduce(x).mean(1).repeat(1, self.cfg.n_registers)  # param-matched control
#     out = self.head(x)
#     return out, gvec                              # gvec feeds BiMambaNet.reg_head if global_dim > 0

# To actually run this ablation: replace(Config(), use_registers=True, n_registers=4, ...) or
# replace(Config(), reg_control=True, ...) as the corrector config, same harness as every other row.
