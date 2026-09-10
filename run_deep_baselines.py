#!/usr/bin/env python3
"""One-command runner for the matched direct/residual control experiment.

The script uses the active environment as-is. Missing data are built automatically
with the frozen cutoff before training. No packages or environments are installed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# PyTorch/CuBLAS reads this before torch is imported. Preserve a user override.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "torch": "torch",
    "arch": "arch",
    "tqdm": "tqdm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-, initialization-, and checkpoint-matched direct/residual models"
    )
    parser.add_argument("--config", default="configs/matched_control.json")
    parser.add_argument(
        "--data-dir",
        help="optional dataset directory; missing data are built automatically",
    )
    parser.add_argument(
        "--output-dir",
        help="result directory (overrides config)",
    )
    parser.add_argument("--force", action="store_true", help="ignore compatible cached cases")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="diagnostic only: first market/backbone, one seed, two epochs, separate output",
    )
    return parser.parse_args()


def dependency_preflight(extra_modules: tuple[str, ...] = ()) -> None:
    missing: list[str] = []
    for module, package in {**REQUIRED_IMPORTS, **{name: name for name in extra_modules}}.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{package} ({type(exc).__name__}: {exc})")
    if missing:
        raise SystemExit(
            "The active environment is missing required packages. Nothing was installed or changed.\n"
            "Missing:\n  - " + "\n  - ".join(missing)
        )


def verify_frozen_data(config: dict, project_root: Path, markets: list[str]) -> None:
    data_dir = project_root / config["data_dir"]
    manifest_path = data_dir / "data_manifest.json"
    universe_path = project_root / config["market_universe"]
    if not manifest_path.exists():
        raise SystemExit(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    expected = {
        "download_start_inclusive": universe.get("start_inclusive"),
        "download_end_exclusive": universe.get("end_exclusive"),
        "requested_cutoff_inclusive": universe.get("requested_cutoff_inclusive"),
    }
    mismatches = {
        key: {"manifest": manifest.get(key), "universe": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"frozen data/universe dates disagree: {mismatches}")
    if manifest.get("omitted_markets"):
        raise SystemExit(f"data manifest records omitted markets: {manifest['omitted_markets']}")
    if int(manifest.get("target_horizon", -1)) != int(config["primary_horizon"]):
        raise SystemExit(
            "data manifest target_horizon differs from the configured primary_horizon"
        )
    recorded = manifest.get("files", {})
    errors: list[str] = []
    for market in markets:
        path = data_dir / f"{market}_vol.csv"
        if market not in recorded:
            errors.append(f"{market}: absent from data_manifest.json")
            continue
        if not path.exists():
            errors.append(f"{market}: missing {path}")
            continue
        expected_hash = recorded[market].get("sha256")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            errors.append(f"{market}: SHA-256 differs from the frozen manifest")
    if errors:
        raise SystemExit("frozen dataset is incomplete or modified:\n  - " + "\n  - ".join(errors))


def _build_frozen_data(config: dict, project_root: Path, markets: list[str], destination: Path) -> None:
    dependency_preflight(extra_modules=("yfinance",))
    command = [
        sys.executable, "-u", str(project_root / "data" / "build_market_dataset.py"),
        "--universe", str((project_root / config["market_universe"]).resolve()),
        "--out-dir", str(destination), "--horizon", str(config["primary_horizon"]),
        "--markets", *markets,
        "--cache-dir", str(project_root / "data" / "raw_cache"),
    ]
    subprocess.run(command, cwd=project_root, check=True)


def ensure_data(
    config: dict, project_root: Path, markets: list[str], explicit_data_dir: str | None = None,
) -> Path:
    """Reuse valid local data or build and verify a fresh dataset before publishing it."""
    project_root = project_root.resolve()
    preferred = (project_root / (explicit_data_dir or config["data_dir"])).resolve()
    candidates = [preferred]
    if explicit_data_dir is None:
        candidates.extend([project_root / "data" / "vol_dataset", project_root / "vol_dataset"])
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    for candidate in candidates:
        if not (candidate / "data_manifest.json").is_file():
            continue
        try:
            verify_frozen_data({**config, "data_dir": str(candidate)}, project_root, markets)
        except (SystemExit, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"[data] existing dataset is not compatible: {candidate}\n{exc}", flush=True)
        else:
            print(f"[data] reusing verified dataset: {candidate}", flush=True)
            return candidate

    universe = json.loads((project_root / config["market_universe"]).read_text(encoding="utf-8"))
    print(
        f"[data] preparing {len(markets)} markets automatically; "
        f"fixed cutoff {universe['requested_cutoff_inclusive']}", flush=True,
    )
    preferred.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{preferred.name}_building_", dir=preferred.parent))
    try:
        _build_frozen_data(config, project_root, markets, staging)
        verify_frozen_data({**config, "data_dir": str(staging)}, project_root, markets)
        backup = None
        if preferred.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = preferred.with_name(preferred.name + "_backup_" + stamp)
            preferred.rename(backup)
        try:
            staging.rename(preferred)
        except OSError:
            if backup is not None:
                backup.rename(preferred)
            raise
        if backup is not None:
            print(f"[data] previous files preserved in {backup}", flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[data] dataset ready: {preferred}", flush=True)
    return preferred


def main() -> None:
    project_root = Path(__file__).resolve().parent
    invocation_directory = Path.cwd()
    args = parse_args()
    os.chdir(project_root)
    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"configuration not found: {config_path}")

    dependency_preflight()
    from volatility_pipeline.config import load_config
    from volatility_pipeline.deep_baselines import run_deep_backbone_suite

    cfg = load_config(config_path)
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = str((invocation_directory / Path(args.data_dir).expanduser()).resolve())
    if args.output_dir:
        overrides["output_dir"] = str((invocation_directory / Path(args.output_dir).expanduser()).resolve())
    if overrides:
        cfg = cfg.with_updates(**overrides)
    markets = list(cfg.confirmatory_markets)
    backbones = list(cfg.correctors)
    data_dir = ensure_data(
        cfg.to_dict(), project_root, markets,
        explicit_data_dir=overrides.get("data_dir"),
    )
    cfg = cfg.with_updates(data_dir=str(data_dir))

    if args.smoke:
        smoke_output = str(Path(cfg.output_dir).with_name(Path(cfg.output_dir).name + "_smoke"))
        cfg = cfg.with_updates(
            seeds=[cfg.seeds[0]],
            epochs=2,
            patience=1,
            output_dir=smoke_output,
            force=args.force,
        )
        markets = markets[:1]
        backbones = backbones[:1]
        print("[smoke] diagnostic output only; do not report it in the paper")
    elif args.force:
        cfg = cfg.with_updates(force=True)

    print(f"[data] verified frozen files for {len(markets)} markets; starting training")
    print(
        f"[suite] {len(markets)} markets x {len(backbones)} backbones x 2 modes x "
        f"{len(cfg.seeds)} seeds"
    )
    generated = run_deep_backbone_suite(cfg, markets=markets, backbones=backbones)
    print("\nCompleted. Main outputs:")
    for name, path in generated.items():
        print(f"  {name:14s} {path}")


if __name__ == "__main__":
    main()
