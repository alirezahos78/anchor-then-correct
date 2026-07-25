"""
Joint+Graph (v1) — static cross-market spillover graph. EVALUATED, NOT ADOPTED.

Status
------
This is the FIRST cross-market graph design tried. One shared BiMamba backbone saw the
CONCATENATED features of all three markets (DJI/IXIC/NYA) at once; a single learned 3x3
row-softmax adjacency matrix, gated near-off at init (sigmoid(-4) ~= 0.018), mixed the three
markets' final residual scalars before adding each market's frozen per-market OLS core.

Finding: the graph gate never moved off its near-zero initialization (converged sigmoid(g) ~=
0.019, essentially unchanged) and the adjacency stayed at uniform ~1/3 everywhere — the model
never learned to use it. Diebold-Mariano tests against the single-market hybrid were not
significant on any index. The leading hypothesis for why it never engaged: the shared backbone,
having full access to all three markets' features from the input layer onward, could already
extract whatever cross-market signal existed on its own, leaving nothing for a residual-level
graph correction to add.

This module is kept for transparency; it depends on the shared `Config`/`BiMamba`/`make_backbone`
/`_core_frame`/`Standardizer` machinery defined in `notebooks/01_main_ablation_tables_I_II_IV.ipynb`
(same classes, unchanged) and is NOT standalone-runnable without them.

Superseded by: joint_graph_dynamic_attention.py (v2), which was also evaluated and not adopted —
see that file for the redesigned, higher-capacity attempt and why it also didn't work.
"""

MARKETS = list(INDICES)

def prepare_joint(cfg):
    per, feats_all, common = [], [], None
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
        f.columns = [f"{idx}:{c}" for c in f.columns]
        cfg_i = replace(cfg, data_path=f"{idx}_vol.csv")
        core = _core_frame(df, cfg_i)[CORE_SETS["full"]]
        mask &= f.notna().all(1) & y.notna() & core.notna().all(1)
        feats_all.append(f); ys.append(y); cores.append(core)
    F_ = pd.concat(feats_all, axis=1)[mask].values.astype("float32")
    Y_ = np.stack([y[mask].values for y in ys], 1).astype("float32")            # [N, 3]
    C_ = np.stack([c[mask].values for c in cores], 1).astype("float32")         # [N, 3, 5]
    N, Ftot = F_.shape
    n_test = int(N * cfg.test_ratio); n_val = int(N * cfg.val_ratio); n_train = N - n_val - n_test
    tr_end, val_end = n_train, n_train + n_val
    fs = Standardizer().fit(F_[:tr_end]); Xs = fs.transform(F_).astype("float32")
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
        Xw.append(Xs[i:i+cfg.seq_in]); Cw.append(Cs[last]); Yw.append(Ys[last]); tgt.append(last)
    Xw, Cw, Yw, tgt = np.asarray(Xw, "float32"), np.asarray(Cw, "float32"), np.asarray(Yw, "float32"), np.asarray(tgt)
    def split(lo, hi):
        s = (tgt >= lo) & (tgt < hi)
        return torch.from_numpy(Xw[s]), torch.from_numpy(Cw[s]), torch.from_numpy(Yw[s])
    return {"train": split(0, tr_end), "val": split(tr_end, val_end), "test": split(val_end, N)}, Ftot, tscalers, coefs

class JointGraphNet(nn.Module):
    def __init__(self, cfg, nf_total, n_mkt, n_core, coefs):
        super().__init__()
        self.backbone = make_backbone(cfg, nf_total)
        self.heads = nn.Linear(nf_total, n_mkt)                                 # per-market residuals
        nn.init.zeros_(self.heads.weight); nn.init.zeros_(self.heads.bias)      # start AT the baselines
        self.adj = nn.Parameter(torch.zeros(n_mkt, n_mkt))                      # spillover graph (row-softmax)
        self.g = nn.Parameter(torch.tensor(-4.0))                               # graph gate, starts ~off
        self.cores = nn.ModuleList([nn.Linear(n_core, 1) for _ in range(n_mkt)])
        for lin, (w, b) in zip(self.cores, coefs):
            with torch.no_grad(): lin.weight.copy_(torch.as_tensor(w).view(1, -1)); lin.bias.fill_(b)
            lin.weight.requires_grad_(False); lin.bias.requires_grad_(False)    # frozen OLS cores
    def forward(self, x, Cc):                                                   # x:[b,L,Ftot]  Cc:[b,3,5]
        h, _ = self.backbone(x); r = self.heads(h.mean(1))                      # [b,3]
        g = torch.sigmoid(self.g)
        r = (1 - g) * r + g * (r @ torch.softmax(self.adj, -1).T)               # graph-coupled residuals
        core = torch.cat([self.cores[i](Cc[:, i]) for i in range(Cc.size(1))], 1)
        return r + core                                                         # [b,3] (standardized per market)

def run_joint(cfg, seeds=SEEDS):
    data, ftot, tscalers, coefs = prepare_joint(cfg)
    dl = {k: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*v), batch_size=cfg.batch_size,
          shuffle=(k == "train")) for k, v in data.items()}
    per_seed = []
    for s in seeds:
        set_seed(s)
        model = JointGraphNet(replace(cfg, seed=s), ftot, len(MARKETS), 5, coefs).to(cfg.device)
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
            model.eval(); ps, tsx = [], []
            for X, Cc, Y in dl["test"]:
                ps.append(model(X.to(cfg.device), Cc.to(cfg.device)).cpu()); tsx.append(Y)
            per_seed.append((torch.cat(ps).numpy(), torch.cat(tsx).numpy(), model))
    P = np.mean(np.stack([p for p, _, _ in per_seed]), 0)                       # seed-ensemble [n,3]
    T = per_seed[0][1]
    rows = {}
    for i, idx in enumerate(MARKETS):
        pr = tscalers[i].inverse(P[:, i]); tr = tscalers[i].inverse(T[:, i])
        mets = [_vol_metrics(torch.tensor(tscalers[i].inverse(p[:, i])), torch.tensor(tr)) for p, _, _ in per_seed]
        rows[idx] = {k: f"{np.mean([m[k] for m in mets]):.4f} +/- {np.std([m[k] for m in mets]):.4f}" for k in mets[0]}
    m = per_seed[-1][2]
    print("graph gate sigmoid(g) =", round(float(torch.sigmoid(m.g)), 4))
    print("adjacency (row-softmax):\n", np.round(torch.softmax(m.adj, -1).detach().cpu().numpy(), 3))
    return pd.DataFrame(rows).T

# Example usage (requires Config/make_backbone/etc. from notebooks/01_main_ablation_tables_I_II_IV.ipynb):
#   joint_table = run_joint(replace(Config(), hybrid=True, weight_decay=1e-3, epochs=EPOCHS, patience=PAT), seeds=SEEDS)
