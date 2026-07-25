# extras/ — evaluated-but-rejected modules

These support claims made in the paper about what was tried and why it isn't part of the final
method. Nothing here appears in any table or figure of the paper — Tables I-V and the VXN section
come from `notebooks/`; both figures (Fig. 1: TikZ architecture diagram; Fig. 2: pgfplots
proxy-invariance bar chart, values from `notebooks/05_vxn_robustness.ipynb`) are drawn directly in
the paper source. See the root `README.md`'s reproduction map. Everything here depends on the
shared `Config`/`BiMamba`/`make_backbone`/
`_core_frame`/`Standardizer` classes defined identically in every notebook in this repo; the `.py`
files are not standalone-runnable, they're excerpts for reference and reuse.

| File | What it is | Why it isn't adopted |
|---|---|---|
| `har_ssm_corrector.ipynb` | HAR-SSM: a diagonal selective-SSM corrector with state-decay rates initialized at the HAR half-lives (1/5/22 days), kept learnable. Includes the full repair history: the original optimizer bug (weight decay was pulling `b_dt` toward zero, freezing the half-lives), the fix (a separate param group with `weight_decay=0`), and a decisive mis-initialization control (an internal diagnostic plot, not a paper figure: planting half-lives at a flat 10 days or at random and checking whether training migrates them back). | No significant QLIKE improvement over the BI-Mamba corrector on any index, **and** the mis-init control showed channels mostly stay near wherever they're planted rather than migrating back toward 1/5/22 — the "recovers HAR time-scales from data" interpretability claim doesn't survive the control. Per the project's own pre-registered decision rule, that means HAR-SSM is dropped and the paper proceeds corrector-agnostic; it appears in no figure or table of the paper. |
| `joint_graph_static.py` | Joint+Graph v1: one shared backbone over all three markets' concatenated features, with a static learned 3x3 row-softmax adjacency mixing the three markets' residuals, gated near-off at init. | The gate never moved off its near-zero initialization and the adjacency stayed uniform (~1/3 everywhere) — no significant DM improvement over the single-market hybrid on any index. Ambiguous as to *why* (see v2). |
| `joint_graph_dynamic_attention.py` | Joint+Graph v2: redesigned so cross-market leakage is structurally impossible except through the graph (per-market backbones, each seeing only its own features) and the graph itself is a real dynamic Q/K/V attention layer over market-embedding nodes (not a static matrix), with a per-market gate. | Still converged to near-uniform attention and near-off gates on every market — a more decisive negative result than v1, since it rules out "the backbone already saw everything" as the explanation. Points to a genuine absence of exploitable cross-sectional spillover between these three large-cap, heavily-overlapping US indices at this frequency. |
| `core_guard_experiment.ipynb` | A validation-time guard on the rolling-origin checkpoint selection (Table III): after training, compare the restored model's validation QLIKE against the frozen-core-only validation QLIKE, and fall back to core-only predictions for that seed's test window if the model is worse. Meant to catch an occasional bad-seed failure. | Fired far more broadly than intended — on folds/indices that were already fine, sometimes making results measurably worse — while only partially fixing the one fold it targeted. The paper reports Table III **without** this guard. |
| `register_tokens.py` | **Not a rejected module** — this is live code still present inside the current `BiMamba` class (`use_registers`/`n_registers`/`reg_control` in `Config`), reproduced here standalone for documentation. It was built as one of BiMamba's general ablation knobs but, as far as we can determine from the project's full history, was never actually toggled on in any reported run. Dormant, not evaluated. | N/A — no experiment exists to report a verdict on. |

## A gap we could not fill

An earlier version of the corrector backbone (predating this project's recorded history) appears to
have included some form of feature-dimension graph module: `Config` still carries vestigial fields
(`cheb_k`, `embed`, `hid`) that match common hyperparameter names for Chebyshev/adaptive graph-conv
designs (e.g. AGCRN-style), and the current backbone class is explicitly commented
`# Pure bidirectional Mamba (graph removed)`. We could not locate any surviving implementation of
that module in any notebook or backup available to us — it was already removed before this project's
recorded work began. We're reporting this honestly rather than reconstructing a plausible-looking
replacement: if this module's code exists somewhere outside what we had access to, it should be
added here; otherwise the vestigial `Config` fields should probably be removed in a future cleanup
pass (not done here, since that would be a code change beyond this packaging task's scope).
