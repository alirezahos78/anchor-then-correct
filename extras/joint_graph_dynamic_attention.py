"""
Joint+Graph (v2) — dynamic graph-attention over per-market embeddings. EVALUATED, NOT ADOPTED.

Status
------
Redesign of joint_graph_static.py meant to close a specific ambiguity in the v1 result: v1's
shared backbone saw ALL markets' features from the input layer, so it was unclear whether the
static graph failed to engage because the mechanism doesn't help, or because the backbone had
already "used up" any cross-market signal before the graph got a chance to add anything.

This version removes that ambiguity by construction: each market gets its OWN backbone seeing
ONLY its own features (cross-market leakage is structurally impossible except through the graph),
pooled to a node embedding; a real dynamic graph-attention layer (Q/K/V softmax attention over the
market nodes, GAT-style, not a static learned matrix) is the ONLY channel for cross-market
information, with a per-market learnable gate (vs. v1's single global gate).

Finding: even with full capacity and no way for the backbone to "steal" the job, the graph still
converged to near-uniform attention (~1/3 to every market) and every per-market gate stayed at
its near-off initialization (sigmoid(g) ~= 0.02, same as v1). This is the more decisive negative
result: it rules out the "backbone already saw everything" explanation and points to a genuine
absence of exploitable, non-redundant cross-sectional volatility spillover between these three
large-cap, heavily-overlapping US equity indices at this frequency.

This module depends on the shared `Config`/`make_backbone`/`_core_frame`/`Standardizer` machinery
(same classes as the main notebooks, unchanged) and is NOT standalone-runnable without them.
"""

MARKETS = list(INDICES)

def prepare_joint(cfg):
    # per-market feature tensors are kept SEPARATE (stacked on a market axis) instead of concatenated:
    # the only path for cross-market information is the graph-attention layer in GraphCoupledNet below.
    per, feats_list, common = [], [], None
    for idx in MARKETS:
        df = pd.read_csv(f"{idx}_vol.csv", index_col=0, parse_dates=True).sort_index()
        common = df.index if common is None else common.intersection(df.index)
        per.append(df)
    per = [df.reindex(common) for df in per]
    mask = pd.Series(True, index=common)
    cores, ys = [], []
    for idx, df in zip(MARKETS, per):
        y = df[cfg.target_col].astype("float32")
        f = df.drop(columns=[c for c in cfg.not_features if c in df.columns]).astype("float32")
        cfg_i = replace(cfg, data_path=f"{idx}_vol.csv")
        core = _core_frame(df, cfg_i)[CORE_SETS["full"]]
        mask &= f.notna().all(1) & y.notna() & core.notna().all(1)
        feats_list.append(f); ys.append(y); cores.append(core)
    Y_ = np.stack([y[mask].values for y in ys], 1).astype("float32")             # [N, n_mkt]
    C_ = np.stack([c[mask].values for c in cores], 1).astype("float32")          # [N, n_mkt, 5]
    F_list = [f[mask].values.astype("float32") for f in feats_list]              # n_mkt x [N, nf]
    N = len(Y_); nf = F_list[0].shape[1]                                          # same builder -> same nf per market
    n_test = int(N * cfg.test_ratio); n_val = int(N * cfg.val_ratio); n_train = N - n_val - n_test
    tr_end, val_end = n_train, n_train + n_val

    fscalers = [Standardizer().fit(F[:tr_end]) for F in F_list]
    Xs_list = [sc.transform(F).astype("float32") for sc, F in zip(fscalers, F_list)]

    tscalers, coefs, Ys, Cs = [], [], np.empty_like(Y_), np.empty_like(C_)
    for i in range(len(MARKETS)):
        cs = Standardizer().fit(C_[:tr_end, i]); Cs[:, i] = cs.transform(C_[:, i])
        ts_ = Standardizer1D().fit(Y_[:tr_end, i]); Ys[:, i] = ts_.transform(Y_[:, i]); tscalers.append(ts_)
        A = np.c_[np.ones(tr_end), Cs[:tr_end, i]]
        b, *_ = np.linalg.lstsq(A, Ys[:tr_end, i], rcond=None)
        coefs.append((b[1:].astype("float32"), float(b[0])))

    Xw, Cw, Yw, tgt = [], [], [], []
    for i in range(N - cfg.seq_in + 1):
        last = i + cfg.seq_in - 1
        Xw.append(np.stack([Xs_list[m][i:i+cfg.seq_in] for m in range(len(MARKETS))], 0))  # [n_mkt, seq_in, nf]
        Cw.append(Cs[last]); Yw.append(Ys[last]); tgt.append(last)
    Xw = np.asarray(Xw, "float32")                        # [n_windows, n_mkt, seq_in, nf]
    Cw, Yw, tgt = np.asarray(Cw, "float32"), np.asarray(Yw, "float32"), np.asarray(tgt)

    def split(lo, hi):
        s = (tgt >= lo) & (tgt < hi)
        return torch.from_numpy(Xw[s]), torch.from_numpy(Cw[s]), torch.from_numpy(Yw[s])
    return {"train": split(0, tr_end), "val": split(tr_end, val_end), "test": split(val_end, N)}, nf, tscalers, coefs

class GraphCoupledNet(nn.Module):
    # One backbone PER MARKET, each seeing only its own features -> no built-in cross-market leakage.
    # Pooled backbone outputs become node embeddings; a single dynamic graph-attention layer (GAT-style
    # Q/K/V softmax over the n_mkt nodes) is the ONLY channel for cross-market (spillover) information.
    # Per-market gate starts ~off (training begins as independent per-market models); per-market zero-init
    # head starts AT the frozen OLS core baseline, same "can't do worse than the core" philosophy as before.
    def __init__(self, cfg, nf, n_mkt, n_core, coefs, d_node=16):
        super().__init__()
        self.n_mkt = n_mkt
        self.backbones = nn.ModuleList([make_backbone(cfg, nf) for _ in range(n_mkt)])
        self.proj = nn.ModuleList([nn.Linear(nf, d_node) for _ in range(n_mkt)])
        self.q = nn.Linear(d_node, d_node); self.k = nn.Linear(d_node, d_node); self.v = nn.Linear(d_node, d_node)
        self.attn_out = nn.Linear(d_node, d_node)
        self.gate = nn.Parameter(torch.full((n_mkt, 1), -4.0))                    # per-market graph gate, starts ~off
        self.heads = nn.ModuleList([nn.Linear(d_node, 1) for _ in range(n_mkt)])
        for h in self.heads: nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)      # start AT the per-market baseline
        self.cores = nn.ModuleList([nn.Linear(n_core, 1) for _ in range(n_mkt)])
        for lin, (w, b) in zip(self.cores, coefs):
            with torch.no_grad(): lin.weight.copy_(torch.as_tensor(w).view(1, -1)); lin.bias.fill_(b)
            lin.weight.requires_grad_(False); lin.bias.requires_grad_(False)       # frozen OLS cores
        self.d_node = d_node
    def forward(self, x, Cc, return_attn=False):                                  # x:[b,n_mkt,L,nf]  Cc:[b,n_mkt,5]
        nodes = []
        for i in range(self.n_mkt):
            h, _ = self.backbones[i](x[:, i])                                     # [b, L, nf]
            nodes.append(self.proj[i](h.mean(1)))                                 # [b, d_node]
        E = torch.stack(nodes, 1)                                                 # [b, n_mkt, d_node]
        Q, K, V = self.q(E), self.k(E), self.v(E)
        attn = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(self.d_node), dim=-1)   # [b, n_mkt, n_mkt]
        msg = self.attn_out(attn @ V)                                             # [b, n_mkt, d_node]
        g = torch.sigmoid(self.gate).unsqueeze(0)                                 # [1, n_mkt, 1]
        Eg = E + g * msg                                                          # per-market gated graph update
        r = torch.cat([self.heads[i](Eg[:, i]) for i in range(self.n_mkt)], 1)    # [b, n_mkt]
        core = torch.cat([self.cores[i](Cc[:, i]) for i in range(Cc.size(1))], 1)
        out = r + core
        return (out, attn) if return_attn else out

def run_joint(cfg, seeds=SEEDS):
    data, nf, tscalers, coefs = prepare_joint(cfg)
    dl = {k: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*v), batch_size=cfg.batch_size,
          shuffle=(k == "train")) for k, v in data.items()}
    per_seed = []
    for s in seeds:
        set_seed(s)
        model = GraphCoupledNet(replace(cfg, seed=s), nf, len(MARKETS), 5, coefs).to(cfg.device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
        lossf = nn.MSELoss(); best, best_state, wait = float("inf"), None, 0
        for ep in range(cfg.epochs):
            model.train()
            for X, Cc, Y in dl["train"]:
                X, Cc, Y = X.to(cfg.device), Cc.to(cfg.device), Y.to(cfg.device)
                opt.zero_grad(); loss = lossf(model(X, Cc), Y); loss.backward()
                if cfg.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            with torch.no_grad():
                model.eval(); es = []
                for X, Cc, Y in dl["val"]:
                    es.append(((model(X.to(cfg.device), Cc.to(cfg.device)).cpu() - Y) ** 2).mean(0))
                v = float(torch.stack(es).mean().sqrt())
            if v < best: best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if cfg.patience and wait >= cfg.patience: break
        if best_state: model.load_state_dict(best_state)
        with torch.no_grad():
            model.eval(); ps, tsx, attns = [], [], []
            for X, Cc, Y in dl["test"]:
                p, a = model(X.to(cfg.device), Cc.to(cfg.device), return_attn=True)
                ps.append(p.cpu()); tsx.append(Y); attns.append(a.cpu())
            per_seed.append((torch.cat(ps).numpy(), torch.cat(tsx).numpy(), model, torch.cat(attns).mean(0)))
    P = np.mean(np.stack([p for p, _, _, _ in per_seed]), 0)                       # seed-ensemble [n,3]
    T = per_seed[0][1]
    rows = {}
    for i, idx in enumerate(MARKETS):
        mets = [_vol_metrics(torch.tensor(tscalers[i].inverse(p[:, i])), torch.tensor(tscalers[i].inverse(T[:, i])))
                for p, _, _, _ in per_seed]
        rows[idx] = {k: f"{np.mean([m[k] for m in mets]):.4f} +/- {np.std([m[k] for m in mets]):.4f}" for k in mets[0]}
    gates = torch.sigmoid(per_seed[-1][2].gate).detach().cpu().numpy().flatten()
    mean_attn = torch.stack([a for _, _, _, a in per_seed]).mean(0).numpy()
    print("per-market graph gate sigmoid(g):", {idx: round(float(g), 4) for idx, g in zip(MARKETS, gates)})
    print("mean test-set attention (rows=query market, cols=key market):\n",
          pd.DataFrame(np.round(mean_attn, 3), index=MARKETS, columns=MARKETS))
    return pd.DataFrame(rows).T

# Example usage (requires Config/make_backbone/etc. from notebooks/01_main_ablation_tables_I_II_IV.ipynb):
#   joint_table = run_joint(replace(Config(), hybrid=True, weight_decay=1e-3, epochs=EPOCHS, patience=PAT), seeds=SEEDS)
