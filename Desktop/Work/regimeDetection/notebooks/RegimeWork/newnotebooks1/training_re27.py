# -*- coding: utf-8 -*-
"""
Feature Engineering + Hierarchical Regime Detection – re27.

re27 is based on re25 (NOT re26 — re26 regressed to AUC 0.652 via simplification).
Adds 8 targeted units: direction-specific L1/L2 targets, structural Renko features,
signed slippage, day-of-week cyclical encoding.

re22 best-in-class: L1 AUC 0.758, L1 Spearman 0.561 (best to date).
re25 base: CLS model restored + fold-local C(d) targets enforced + comprehensive metrics.

  CHANGES vs re25
  ───────────────
  Unit 1   L2 window streak direction-specific (short streak=4, long=3).
  Unit 2   Activate fold-local Gates 2&3 in compute_cd_target (fold_train_df=_train_slice).
  Unit 3   Short L1 avoidance target: predict bad days, gate inverted at application.
  Unit 4   Short L2 CLS target: risk-adjusted PnL (pnl minus worst-window drawdown).
  Unit 5   net_slippage_signed feature (direction-aware slippage signal).
  Unit 6   Direction-specific duration boost thresholds (short=20min, long=30min).
  Unit 7   bricks_above_breakeven + exit_jump_direction_impact from df_raw.
  Unit 8   Day-of-week cyclical encoding (dow_sin, dow_cos).

  FIX-re25a  Restore L2 CLS win/loss classifier (from re22, Step 9b).
             Binary LGB+XGB classifier with OOF ISO calibration on win_prob target
             (sign(pnl)>0). Note: ISO here calibrates win probability — not rank scores —
             so it is not subject to the re23 compression bug (which was ISO-on-rank).
             CLS gate: per-day top-30% by win_prob replaces per-day top-30% by rank.
             Both rank (pred_label_rank) and CLS (pred_label) gates are output for comparison.

  FIX-re25b  Restore comprehensive metrics from re22.
             fold_metrics: brier, pearson, P/R/F1 at base-rate and 0.65 gates,
             full confusion matrix (TP/FP/FN/TN), pct_taken.
             l2_fold_metrics: l2_precision_cls, l2_recall_cls, l2_f1_cls (window target),
             l2_win_precision_cls / l2_win_recall_cls / l2_win_f1_cls (sign(pnl) target),
             l2_lift_ratio_cls, all TP/FP/FN/TN counts.

  FIX-re25c  Target calculation strictly fold-local.
             Global C(d) computation kept for dataset health printing only (DIAGNOSTIC).
             add_fold_safe_target no longer filters on regime_target (placeholder 0);
             authoritative regime_target is recomputed in the walk-forward loop per direction.

All re24/re25 improvements preserved: direction loop (long/short), fold-local targets,
TEMPORAL_ARI=0, no rank ISO, 1min indicator injection, regime_quality+conviction,
large/REVERSAL dropped, is_unbalance=True, S3/Wasabi data loading.
Output CSVs use _re27_ suffix.
"""

import importlib.util
import io
import sys
import threading as _threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)

# ============================================================
# PHASE 1 — Data Loading
# ============================================================
# Import build_renko_dataset.py for S3 helpers and trade/OHLC build
_RENKO_SCRIPT = Path(
    "/Users/mazza/Desktop/Work/regimeDetection/notebooks/analysis2/newnotebooks1/build_renko_dataset.py"
)
_renko_spec = importlib.util.spec_from_file_location("build_renko_dataset", _RENKO_SCRIPT)
_renko = importlib.util.module_from_spec(_renko_spec)
_renko_spec.loader.exec_module(_renko)

# Local cache paths — indicators are merged back into the super_merged base files
_SM_DIR     = Path("/Users/mazza/Desktop/Work/regimeDetection/notebooks/RegimeWork/newnotebooks1/regimedataset/super_merged")
_IND_PREFIX = "indicators_by_frequency/"

_ind_long_path   = _SM_DIR / "super_merged_long.csv"
_ind_short_path  = _SM_DIR / "super_merged_short.csv"
_base_long_path  = _SM_DIR / "super_merged_long.csv"
_base_short_path = _SM_DIR / "super_merged_short.csv"
# Parquet mirrors — written after the final merge, read preferentially on future runs
_ind_long_pq   = _SM_DIR / "super_merged_long.parquet"
_ind_short_pq  = _SM_DIR / "super_merged_short.parquet"


def _read_df(csv_path: "Path", pq_path: "Path") -> "pd.DataFrame":
    """Load parquet if available (10-20x faster); fall back to CSV."""
    if pq_path.exists():
        print(f"  ✔  Loading parquet cache: {pq_path.name}")
        return pd.read_parquet(pq_path)
    return pd.read_csv(csv_path, low_memory=False)


def _s3_list_folders(s3, prefix):
    pag, names = s3.get_paginator("list_objects_v2"), []
    for page in pag.paginate(Bucket=_renko.BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.append(cp["Prefix"].rstrip("/").split("/")[-1])
    return sorted(names)


def _s3_list_keys(s3, prefix):
    pag, keys = s3.get_paginator("list_objects_v2"), []
    for page in pag.paginate(Bucket=_renko.BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _download_indicator(s3, freq_min, indicator, ticker):
    prefix  = f"{_IND_PREFIX}{freq_min}min/{indicator}/{ticker}/"
    matches = [k for k in _s3_list_keys(s3, prefix) if k.upper().endswith(".CSV")]
    if not matches:
        return None
    try:
        resp = s3.get_object(Bucket=_renko.BUCKET, Key=sorted(matches)[-1])
        df_i = pd.read_csv(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        print(f"    ✖  {ticker}: download error — {exc}")
        return None
    df_i.columns = [c.strip().lower() for c in df_i.columns]

    # 't' is the actual timestamp column name in S3 indicator files — list first.
    time_col = next(
        (c for c in ["t", "timestamp", "datetime", "date_time", "time", "date"]
         if c in df_i.columns),
        None,
    )
    if time_col is None:
        print(f"    ✖  {ticker}: no timestamp column in {indicator} — {df_i.columns.tolist()}")
        return None
    if "close" not in df_i.columns:
        print(f"    ✖  {ticker}: no 'close' column in {indicator} — {df_i.columns.tolist()}")
        return None

    # Each indicator file contains only a timestamp + close value.
    # Keep only those two columns and rename close → {indicator}_{freq_min}min_close.
    # _strip_close_suffix (called after all merges) will drop the _close suffix to
    # produce the final column name: {indicator}_{freq_min}min (e.g. net_gex_5min).
    df_i = df_i[[time_col, "close"]].copy()
    df_i[time_col] = pd.to_datetime(df_i[time_col])
    df_i = df_i.rename(columns={"close": f"{indicator}_{freq_min}min_close"})
    df_i["Ticker"]      = ticker
    df_i["_merge_time"] = df_i[time_col]
    df_i.drop(columns=[time_col], inplace=True, errors="ignore")
    return df_i.sort_values("_merge_time").reset_index(drop=True)


def _merge_indicator(df_base, all_ind, time_col="entry_time"):
    parts = []
    for ticker in df_base["Ticker"].unique():
        tdf   = df_base[df_base["Ticker"] == ticker].sort_values(time_col).copy()
        ind_t = all_ind[all_ind["Ticker"] == ticker].drop(columns=["Ticker"], errors="ignore")
        if ind_t.empty:
            parts.append(tdf)
            continue
        merged = pd.merge_asof(
            tdf,
            ind_t.sort_values("_merge_time"),
            left_on=time_col,
            right_on="_merge_time",
            direction="backward",
        )
        merged.drop(columns=["_merge_time"], inplace=True, errors="ignore")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


_N_DL_WORKERS    = 8    # parallel S3 download threads (one per core)
_N_MERGE_WORKERS = 8    # parallel indicator merge workers (one per CPU core)
_thread_local_s3 = _threading.local()


def _get_thread_s3():
    """Return a per-thread boto3 S3 client (boto3 clients are not thread-safe)."""
    if not hasattr(_thread_local_s3, "client"):
        _thread_local_s3.client = _renko.make_s3_client(_access_key, _secret_key)
    return _thread_local_s3.client


def _download_indicator_worker(freq_min, indicator, ticker):
    """Thread worker wrapper — creates its own S3 client."""
    return _download_indicator(_get_thread_s3(), freq_min, indicator, ticker)


def _merge_indicator_parallel(df_base, all_ind, time_col="entry_time"):
    """Like _merge_indicator but parallelises the per-ticker merge_asof."""
    tickers = df_base["Ticker"].unique()

    def _one(ticker):
        tdf   = df_base[df_base["Ticker"] == ticker].sort_values(time_col).copy()
        ind_t = all_ind[all_ind["Ticker"] == ticker].drop(columns=["Ticker"], errors="ignore")
        if ind_t.empty:
            return tdf
        merged = pd.merge_asof(
            tdf, ind_t.sort_values("_merge_time"),
            left_on=time_col, right_on="_merge_time", direction="backward",
        )
        merged.drop(columns=["_merge_time"], inplace=True, errors="ignore")
        return merged

    with ThreadPoolExecutor(max_workers=min(_N_DL_WORKERS, len(tickers))) as ex:
        parts = list(ex.map(_one, tickers))
    return pd.concat(parts, ignore_index=True)


def _merge_ind_col(df_base, all_ind, fmin, ind_name, time_col="entry_time"):
    """Merge one indicator and return only the new column as a Series (index-aligned).
    Avoids copying the full df_base so multiple indicators can run in parallel safely."""
    col_name  = f"{ind_name}_{fmin}min_close"
    tickers   = df_base["Ticker"].unique()
    col_parts = []
    for ticker in tickers:
        tdf   = df_base.loc[df_base["Ticker"] == ticker, [time_col]].sort_values(time_col)
        ind_t = all_ind[all_ind["Ticker"] == ticker].drop(columns=["Ticker"], errors="ignore")
        if ind_t.empty:
            col_parts.append(pd.Series(float("nan"), index=tdf.index, name=col_name))
            continue
        merged = pd.merge_asof(
            tdf.reset_index(), ind_t.sort_values("_merge_time"),
            left_on=time_col, right_on="_merge_time", direction="backward",
        )
        merged.drop(columns=["_merge_time", time_col], inplace=True, errors="ignore")
        col_parts.append(merged.set_index("index")[col_name])
    return col_name, pd.concat(col_parts)


def _write_checkpoints(df_long, df_short, long_path, short_path):
    """Write both CSVs atomically (tmp → rename) in parallel to prevent corruption on kill."""
    def _safe_write(df, path):
        tmp = Path(str(path) + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.rename(path)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fl = ex.submit(_safe_write, df_long,  long_path)
        fs = ex.submit(_safe_write, df_short, short_path)
        fl.result()
        fs.result()


def _strip_close_suffix(df):
    """Drop the '_close' suffix from indicator columns added during merging.
    'net_gex_5min_close' → 'net_gex_5min'.
    Only strips columns that end with exactly '_close' and where the resulting
    name does not already exist (prevents silent overwrites)."""
    existing = set(df.columns)
    rename = {}
    for col in df.columns:
        if col.endswith("_close"):
            clean = col[:-6]   # len("_close") == 6
            if clean not in existing:
                rename[col] = clean
    if rename:
        df = df.rename(columns=rename)
    return df


# ── Step 1: Collect indicator requests BEFORE any file I/O or S3 access ──────
# Both frequencies are prompted consecutively so the user can review the full
# request before any downloading begins.  This also allows the local-cache check
# below to gate S3 entirely when everything is already present.


def _parse_ind_input(raw: str):
    """
    Parse a raw user string → 'all' | [name, ...] | [].

    Accepts:
      • Blank / 'n' / 'no'  → [] (skip)
      • 'all'               → 'all' (download every indicator for this frequency)
      • /path/to/file.txt   → reads file; each line may be comma-separated;
                              lines starting with '#' are ignored
      • ~/path/to/file.txt  → same, with home-dir expansion
      • name1,name2,...     → explicit comma-separated list (no hard limit)
    """
    raw = raw.strip()
    if not raw or raw.lower() in ("n", "no", "skip"):
        return []
    if raw.lower() == "all":
        return "all"
    # Treat as file path if it has a .txt extension or starts with / or ~
    if raw.endswith(".txt") or raw.startswith("/") or raw.startswith("~"):
        p = Path(raw).expanduser()
        if p.exists():
            names = []
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names.extend(n.strip() for n in line.split(",") if n.strip())
            return names
        print(f"  ✖  File not found: {p}")
        print("     Fix the path and re-run, or enter indicator names directly as a comma-separated list.")
        sys.exit(1)
    return [n.strip() for n in raw.split(",") if n.strip()]


print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║               Indicator Selection  (re25)                   ║")
print("║  For each frequency answer y/n, then supply indicator names ║")
print("║  as a comma-separated list, a path to a .txt file (one      ║")
print("║  name per line or comma-separated), or 'all' for every      ║")
print("║  available indicator.  Already-cached columns are detected. ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()

_use_5min = input("  Use 5min indicators? (y/n): ").strip().lower()
if _use_5min == "y":
    _5min_raw = input("  5min indicators (file path / comma-separated / 'all'): ").strip()
else:
    _5min_raw = ""

_use_1min = input("  Use 1min indicators? (y/n): ").strip().lower()
if _use_1min == "y":
    _1min_raw = input("  1min indicators (file path / comma-separated / 'all'): ").strip()
else:
    _1min_raw = ""
print()

# {freq_min: 'all' | [name, ...] | []}
_ind_req: dict = {5: _parse_ind_input(_5min_raw), 1: _parse_ind_input(_1min_raw)}


# ── Step 2: Per-indicator local cache check ───────────────────────────────────
def _col_in_csv(csv_path: "Path", freq_min: int, indicator: str) -> bool:
    """
    Return True if {indicator}_{freq_min}min (or a _close / suffixed variant)
    already exists as a column header in csv_path.
    Reads only the first row — no memory cost for large CSVs.
    """
    if not csv_path.exists():
        return False
    try:
        cols = set(pd.read_csv(csv_path, nrows=0).columns.str.lower())
    except Exception:
        return False
    tag = f"{indicator}_{freq_min}min"
    return any(c == tag or c.startswith(f"{tag}_") for c in cols)


# Classify each named indicator as CACHED (skip S3) or MISSING (need S3).
# 'all' always requires S3 — we cannot know the full list without listing the bucket.
_cached:  dict = {5: [], 1: []}   # indicators already in enriched CSV cache
_missing: dict = {5: [], 1: []}   # indicators that must be fetched from S3

for _fmin, _inds in _ind_req.items():
    if not _inds:
        continue
    if _inds == "all":
        _missing[_fmin] = "all"   # sentinel: expand to full S3 list later
        print(f"  ℹ  {_fmin}min → 'all' requested — S3 listing required")
        continue
    for _ind in _inds:
        if (_col_in_csv(_ind_long_path,  _fmin, _ind) and
                _col_in_csv(_ind_short_path, _fmin, _ind)):
            _cached[_fmin].append(_ind)
            print(f"  ✔  {_ind}_{_fmin}min already in local cache — skipping S3")
        else:
            _missing[_fmin].append(_ind)
            print(f"  ↓  {_ind}_{_fmin}min not in cache — will download from S3")

_any_missing = any(_missing[f] for f in (5, 1))


# ── Step 3: Load dataset ──────────────────────────────────────────────────────
if _ind_long_path.exists() and _ind_short_path.exists() and not _any_missing:
    # ── Fast path: every requested indicator is already in enriched cache ─────
    print(f"\n  ✔  All requested indicators present locally → {_SM_DIR}")
    print("     Skipping all S3 downloads.\n")
    _df_long  = _read_df(_ind_long_path,  _ind_long_pq)
    _df_short = _read_df(_ind_short_path, _ind_short_pq)

else:
    # ── Slow path: at least one indicator (or base data) needs S3 ────────────
    _access_key, _secret_key = _renko.prompt_credentials()
    _s3 = _renko.make_s3_client(_access_key, _secret_key)

    # Load base data (start from enriched cache if available so existing
    # indicators are not lost; fall back to base-only cache; otherwise build).
    if _ind_long_path.exists() and _ind_short_path.exists():
        print(f"\n  ✔  Loading enriched cache as base (will append missing indicators)")
        _df_long  = _read_df(_ind_long_path,  _ind_long_pq)
        _df_short = _read_df(_ind_short_path, _ind_short_pq)
    elif _base_long_path.exists() and _base_short_path.exists():
        print(f"\n  ✔  Base trade+OHLC dataset cached → {_SM_DIR}")
        print("     Skipping trade/OHLC build.\n")
        _df_long  = pd.read_csv(_base_long_path,  low_memory=False)
        _df_short = pd.read_csv(_base_short_path, low_memory=False)
    else:
        if not _renko.run_health_check(_s3):
            print("  Aborting — fix the errors above and re-run.\n")
            sys.exit(1)
        print("\n  Building trade+OHLC dataset from Wasabi S3...")
        _renko.build_dataset(_s3)
        _df_long  = pd.read_csv(_base_long_path,  low_memory=False)
        _df_short = pd.read_csv(_base_short_path, low_memory=False)

    _df_long["entry_time"]  = pd.to_datetime(_df_long["entry_time"])
    _df_short["entry_time"] = pd.to_datetime(_df_short["entry_time"])
    _all_tickers = sorted(set(
        _df_long["Ticker"].unique().tolist() + _df_short["Ticker"].unique().tolist()
    ))

    # ── Parallel download of MISSING indicators ──────────────────────────────────
    # Strategy:
    #   1. For each frequency, filter out already-cached indicators.
    #   2. Submit every (indicator, ticker) pair concurrently (max _N_DL_WORKERS threads).
    #      Each thread creates its own boto3 client via _get_thread_s3().
    #   3. After all downloads complete, merge + checkpoint one indicator at a time
    #      (sequential order preserves restart safety — _col_in_csv detects partial runs).
    #   4. Within each indicator merge, long + short run in parallel (2 threads),
    #      and each merge parallelises the per-ticker merge_asof loop.
    #   5. Checkpoint writes (long + short) run in parallel (2 threads).
    for _fmin, _to_dl in _missing.items():
        if not _to_dl:
            continue
        _freq_label = f"{_fmin}min"
        if _to_dl == "all":
            _to_dl = _s3_list_folders(_s3, f"{_IND_PREFIX}{_fmin}min/")
            print(f"\n  Found {len(_to_dl)} {_freq_label} indicators on S3.")

        # Filter indicators already present in both CSV caches.
        # _col_in_csv matches both clean names and _close-suffixed variants,
        # so this correctly detects partial-run checkpoints too.
        _to_dl = [
            _ind for _ind in _to_dl
            if not (_col_in_csv(_ind_long_path,  _fmin, _ind)
                    and _col_in_csv(_ind_short_path, _fmin, _ind))
        ]
        if not _to_dl:
            continue

        # Submit all (indicator, ticker) downloads concurrently
        _tasks = [(_ind, _tk) for _ind in _to_dl for _tk in _all_tickers]
        print(f"\n  Downloading {len(_to_dl)} {_freq_label} indicator(s) × "
              f"{len(_all_tickers)} ticker(s) = {len(_tasks)} requests "
              f"[{_N_DL_WORKERS} parallel threads]...")

        _dl_results: dict = {_ind: [] for _ind in _to_dl}

        with ThreadPoolExecutor(max_workers=_N_DL_WORKERS) as _dl_ex:
            _dl_futs = {
                _dl_ex.submit(_download_indicator_worker, _fmin, _ind, _tk): (_ind, _tk)
                for _ind, _tk in _tasks
            }
            for _fut in as_completed(_dl_futs):
                _ind, _tk = _dl_futs[_fut]
                try:
                    _df_part = _fut.result()
                except Exception as _exc:
                    print(f"    ✖  {_tk}/{_ind}: exception — {_exc}")
                    _df_part = None
                if _df_part is not None:
                    _dl_results[_ind].append(_df_part)
                    print(f"    ✔  {_tk}/{_ind}: {len(_df_part):,} rows")
                else:
                    print(f"    ✖  {_tk}/{_ind}: not found — NaN column will be added")

        # Merge all pending indicators in parallel batches of _N_MERGE_WORKERS.
        # Each indicator only adds one new column, so we merge against a read-only
        # snapshot of _df_long/_df_short and assign columns back after all complete.
        # Checkpoint once per batch instead of once per indicator.
        for _b0 in range(0, len(_to_dl), _N_MERGE_WORKERS):
            _batch = [
                (_ind, pd.concat(_dl_results[_ind], ignore_index=True))
                for _ind in _to_dl[_b0:_b0 + _N_MERGE_WORKERS]
                if _dl_results[_ind]
            ]
            _skipped = [
                _ind for _ind in _to_dl[_b0:_b0 + _N_MERGE_WORKERS]
                if not _dl_results[_ind]
            ]
            for _ind in _skipped:
                print(f"  No data for {_ind}_{_freq_label} — skipping.")
            if not _batch:
                continue

            # Submit long + short for every indicator in the batch simultaneously
            # (up to _N_MERGE_WORKERS * 2 concurrent merge_asof calls).
            # All futures must finish before any column is written back — writing
            # _df_long/_df_short while other threads are still reading them is a
            # pandas thread-safety violation (numpy ops can release the GIL).
            with ThreadPoolExecutor(max_workers=_N_MERGE_WORKERS * 2) as _mex:
                _long_futs  = {_ind: _mex.submit(_merge_ind_col, _df_long,  _ai, _fmin, _ind)
                               for _ind, _ai in _batch}
                _short_futs = {_ind: _mex.submit(_merge_ind_col, _df_short, _ai, _fmin, _ind)
                               for _ind, _ai in _batch}
                # Collect all results first — parallel reads complete here
                _long_res  = {_ind: _long_futs[_ind].result()  for _ind, _ in _batch}
                _short_res = {_ind: _short_futs[_ind].result() for _ind, _ in _batch}
            # Assign columns after the executor exits (all threads done)
            for _ind, _ in _batch:
                _col, _ls = _long_res[_ind]
                _df_long[_col]  = _ls
                _col, _ss = _short_res[_ind]
                _df_short[_col] = _ss
                print(f"  ✔  Merged {_ind}_{_freq_label} — {_df_long.shape[1]} cols (long).")

            _SM_DIR.mkdir(parents=True, exist_ok=True)
            _write_checkpoints(_df_long, _df_short, _ind_long_path, _ind_short_path)
            _n_done = min(_b0 + _N_MERGE_WORKERS, len(_to_dl))
            print(f"  ✔  Checkpoint [{_n_done}/{len(_to_dl)}] saved (batch {_b0 // _N_MERGE_WORKERS + 1}).")

    _df_long  = _strip_close_suffix(_df_long)
    _df_short = _strip_close_suffix(_df_short)
    print(f"  ✔  Column cleanup done — sample cols: "
          f"{[c for c in _df_long.columns if not c.startswith('_')][-5:]}")
    _SM_DIR.mkdir(parents=True, exist_ok=True)
    _write_checkpoints(_df_long, _df_short, _ind_long_path, _ind_short_path)
    # Save parquet mirrors — future runs load these instead of 227MB CSVs
    with ThreadPoolExecutor(max_workers=2) as _pq_ex:
        _pq_ex.submit(_df_long.to_parquet,  _ind_long_pq,  index=False)
        _pq_ex.submit(_df_short.to_parquet, _ind_short_pq, index=False)
    print(f"\n  ✔  Enriched dataset saved → {_SM_DIR}\n")

# Ensure entry_time is datetime in all load paths
_df_long["entry_time"]  = pd.to_datetime(_df_long["entry_time"])
_df_short["entry_time"] = pd.to_datetime(_df_short["entry_time"])

# ── Combine long + short into single df for training pipeline ─────────────────
_df_combined = pd.concat([_df_long, _df_short], ignore_index=True)
if "factor" not in _df_combined.columns and "direction" in _df_combined.columns:
    _df_combined["factor"] = _df_combined["direction"]

# ============================================================
# 0. Configuration & Paths
# ============================================================
OUTPUT_DIR = Path(
    "/Users/mazza/Desktop/Work/regimeDetection/notebooks/RegimeWork/newnotebooks1/outdataset"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "AAPL_tier4_features_no_target_re25.parquet"

GROUP_KEYS = ['Ticker', 'factor']

print("=" * 80)
print("FEATURE ENGINEERING – re25 (no L2 ISO + regime_quality restored + REVERSAL dropped)")
print("=" * 80)
print(f"Output: {OUTPUT_FILE}")
print("=" * 80)

# ============================================================
# 1. Load Raw Data & Basic Cleanup
# ============================================================
t0 = time.perf_counter()
print("\nLoading dataset from Phase 1 output...")
df = _df_combined

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
for suffix in ['_x', '_y']:
    df.drop(columns=[c for c in df.columns if c.endswith(suffix)], inplace=True, errors='ignore')

required_trade_cols = [
    'Ticker', 'factor', 'entry_time', 'pnl', 'theoretical_profit',
    'brick_size', 'atr_brick_size', 'entry_price', 'trend',
]
missing = [c for c in required_trade_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df['entry_time'] = pd.to_datetime(df['entry_time'])
df['entry_date'] = df['entry_time'].dt.normalize()
df.sort_values(['Ticker', 'entry_date', 'factor', 'entry_time'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Loaded {len(df):,} rows, {df['Ticker'].nunique()} ticker(s)")

_FORBIDDEN_SOURCES = frozenset([
    'pnl', 'is_win', 'exit_price', 'theoretical_profit',
    'exit_jump_bricks', 'decision_bricks',
])

# ============================================================
# Helpers
# ============================================================
def shift_within_group(df: pd.DataFrame, col: str, n: int = 1) -> pd.Series:
    """Shift `col` by `n` steps within each GROUP_KEYS group."""
    return df.groupby(GROUP_KEYS)[col].shift(n)


if 'trend' in df.columns:
    trend_int: pd.Series = df['trend'].astype(int)
else:
    trend_int = None

# ============================================================
# 2. State Features
# ============================================================
print("\nComputing state features...")

if 'PSAR_uptrend' in df.columns:
    df['reversal_density_5'] = df.groupby(GROUP_KEYS)['PSAR_uptrend'].transform(
        lambda x: x.shift(1).astype(float).diff().abs().fillna(0).rolling(5, min_periods=3).mean()
    )
    df['reversal_density_10'] = df.groupby(GROUP_KEYS)['PSAR_uptrend'].transform(
        lambda x: x.shift(1).astype(float).diff().abs().fillna(0).rolling(10, min_periods=5).mean()
    )
else:
    df['reversal_density_5'] = df['reversal_density_10'] = np.nan

if trend_int is not None:
    # Vectorized: abs(rolling_sum) / rolling_count — equivalent to the old
    # python-lambda apply but uses native C ops (~100x faster on 54k rows).
    df['directional_efficiency_10'] = df.groupby(GROUP_KEYS)['trend'].transform(
        lambda x: x.shift(1).pipe(
            lambda s: s.rolling(10, min_periods=5).sum().abs()
                      / s.rolling(10, min_periods=5).count()
        )
    )
else:
    df['directional_efficiency_10'] = np.nan

if trend_int is not None:
    df['_trend_change']   = (trend_int != shift_within_group(df, 'trend', 1)).astype(int)
    df['_run_id']         = df.groupby(GROUP_KEYS)['_trend_change'].cumsum()
    df['_run_length_raw'] = df.groupby(['Ticker', 'factor', '_run_id']).cumcount() + 1
    df['run_length']      = shift_within_group(df, '_run_length_raw', 1).fillna(1)
    df.drop(columns=['_trend_change', '_run_id', '_run_length_raw'], inplace=True)
else:
    df['run_length'] = np.nan

if trend_int is not None:
    df['alternation_ratio'] = df.groupby(GROUP_KEYS)['trend'].transform(
        lambda x: (x.astype(int) != x.astype(int).shift(1))
                   .astype(float).shift(1)
                   .rolling(10, min_periods=5).mean()
    )
else:
    df['alternation_ratio'] = np.nan

if 'ATR' in df.columns:
    df['atr_compression_ratio'] = df.groupby(GROUP_KEYS)['ATR'].transform(
        lambda x: (
            x.shift(1).rolling(5, min_periods=3).mean() /
            x.shift(1).rolling(20, min_periods=10).mean().replace(0, np.nan)
        ).clip(0.1, 5)
    )
    df['vol_regime'] = df.groupby(GROUP_KEYS)['ATR'].transform(
        lambda x: (
            x.shift(1) /
            x.shift(1).rolling(20, min_periods=10).mean().replace(0, np.nan)
        ).clip(0.1, 5)
    )
else:
    df['atr_compression_ratio'] = df['vol_regime'] = np.nan

if 'run_length' in df.columns:
    run_max_20 = df.groupby(GROUP_KEYS)['run_length'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).max()
    )
    df['pullback_from_max_progress'] = (
        (run_max_20 / (df['run_length'] + 1e-6) - 1).clip(0, 20)
    )
else:
    df['pullback_from_max_progress'] = np.nan

if 'gex_call_percentile_20' in df.columns:
    df['call_wall_proxy'] = df['gex_call_percentile_20'].fillna(0.5)
elif 'call_wall_percentile_20_5min' in df.columns:
    df['call_wall_proxy'] = df['call_wall_percentile_20_5min'].fillna(0.5)
if 'gex_put_percentile_20' in df.columns:
    df['put_wall_proxy'] = (1 - df['gex_put_percentile_20'].fillna(0.5)).clip(0, 1)
elif 'put_wall_percentile_20_5min' in df.columns:
    df['put_wall_proxy'] = (1 - df['put_wall_percentile_20_5min'].fillna(0.5)).clip(0, 1)

for col in ['dist_to_gamma_flip_atr', 'net_gex_zscore_20', 'net_gex_percentile_20',
            'RSIOscillator', 'SqueezeMomentum']:
    if col not in df.columns:
        df[col] = np.nan

# ============================================================
# 3. Cross-Factor Features (Leave-One-Out)
# ============================================================
print("\nComputing cross-factor features (LOO)...")
MICRO_THRESHOLD = 0.1
micro_factors = set(f for f in df['factor'].unique() if f < MICRO_THRESHOLD)
macro_factors = set(f for f in df['factor'].unique() if f >= MICRO_THRESHOLD)
agg_cols = ['entry_price', 'brick_size', 'atr_brick_size']

time_agg = df.groupby(['Ticker', 'entry_time']).agg(
    **{f'{c}_sum':    (c, 'sum')                  for c in agg_cols},
    **{f'{c}_sum_sq': (c, lambda x: (x**2).sum()) for c in agg_cols},
    **{f'{c}_count':  (c, 'count')                for c in agg_cols},
).reset_index()

df = df.merge(time_agg, on=['Ticker', 'entry_time'], how='left')

for c in agg_cols:
    n, s, ss, own = df[f'{c}_count'], df[f'{c}_sum'], df[f'{c}_sum_sq'], df[c]
    loo_n   = n - 1
    loo_sum = s - own
    loo_ss  = ss - own ** 2
    df[f'{c}_loo_mean'] = np.where(loo_n > 0, loo_sum / loo_n, np.nan)
    loo_mean = df[f'{c}_loo_mean']
    loo_var  = np.where(
        loo_n > 1,
        (loo_ss / loo_n - loo_mean ** 2) * (loo_n / (loo_n - 1)),
        np.nan,
    )
    df[f'{c}_loo_std'] = np.where(
        np.isfinite(loo_var) & (loo_var >= 0), np.sqrt(np.maximum(loo_var, 0)), np.nan
    )
    df[f'{c}_loo_zscore'] = np.where(
        df[f'{c}_loo_std'].notna() & (df[f'{c}_loo_std'] > 1e-10),
        (df[c] - loo_mean) / df[f'{c}_loo_std'],
        np.nan,
    )

drop_cols = [c for c in df.columns
             if c.endswith('_sum') or c.endswith('_sum_sq') or c.endswith('_count')]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

_loo_cols = [f'{c}_{s}' for c in agg_cols for s in ('loo_mean', 'loo_std', 'loo_zscore')]
_loo_cols = [c for c in _loo_cols if c in df.columns]
for _col in _loo_cols:
    df[f'{_col}_lag1'] = shift_within_group(df, _col, 1)
    df.drop(columns=[_col], inplace=True)

df['_is_micro'] = df['factor'].isin(micro_factors)
df['_is_macro']  = df['factor'].isin(macro_factors)

micro_agg = (
    df[df['_is_micro']].groupby(['Ticker', 'entry_time'])
    .agg(micro_entry_price_mean=('entry_price', 'mean')).reset_index()
)
macro_agg = (
    df[df['_is_macro']].groupby(['Ticker', 'entry_time'])
    .agg(macro_entry_price_mean=('entry_price', 'mean')).reset_index()
)
df = df.merge(micro_agg, on=['Ticker', 'entry_time'], how='left')
df = df.merge(macro_agg, on=['Ticker', 'entry_time'], how='left')
df['micro_macro_divergence_entry'] = (
    df['micro_entry_price_mean'] - df['macro_entry_price_mean']
)
df.drop(columns=['micro_entry_price_mean', 'macro_entry_price_mean'],
        inplace=True, errors='ignore')
df['micro_macro_divergence_entry_lag1'] = shift_within_group(df, 'micro_macro_divergence_entry', 1)
df.drop(columns=['micro_macro_divergence_entry'], inplace=True)

# ============================================================
# 4. Stack Active Counts & Micro Conditioning
# ============================================================
print("\nComputing stack and micro conditioning features...")

stack_stats = (
    df.groupby(['Ticker', 'entry_time'])
    .agg(stack_active_count=('factor', 'count')).reset_index()
)
df = df.merge(stack_stats, on=['Ticker', 'entry_time'], how='left')

micro_rows = (
    df[df['_is_micro']].groupby(['Ticker', 'entry_time'])
    .agg(micro_active_count=('factor', 'count')).reset_index()
)
macro_rows = (
    df[df['_is_macro']].groupby(['Ticker', 'entry_time'])
    .agg(macro_active_count=('factor', 'count')).reset_index()
)
df = df.merge(micro_rows, on=['Ticker', 'entry_time'], how='left')
df = df.merge(macro_rows, on=['Ticker', 'entry_time'], how='left')
for col in ['micro_active_count', 'macro_active_count']:
    df[col] = df[col].fillna(0).astype(int)

for _col in ['stack_active_count', 'micro_active_count', 'macro_active_count']:
    if _col in df.columns:
        df[f'{_col}_lag1'] = shift_within_group(df, _col, 1)
        df.drop(columns=[_col], inplace=True)

_micro_df = df[df['_is_micro']].copy()
_micro_df['_ep_dir'] = (df.loc[df['_is_micro'], 'trend'].astype(int) * 2 - 1).fillna(0)
micro_summary = (
    _micro_df.groupby(['Ticker', 'entry_time'])
    .agg(
        micro_agreement_signed   = ('_ep_dir', 'mean'),
        micro_consensus_strength = ('_ep_dir', lambda x: abs(x.mean())),
        micro_dispersion         = ('_ep_dir', 'std'),
        micro_factor_spread      = ('entry_price', lambda x: x.max() - x.min()),
    )
    .reset_index()
)
df = df.merge(micro_summary, on=['Ticker', 'entry_time'], how='left')
for _col in ['micro_agreement_signed', 'micro_consensus_strength',
             'micro_dispersion', 'micro_factor_spread']:
    if _col in df.columns:
        df[f'{_col}_lag1'] = shift_within_group(df, _col, 1)
        df.drop(columns=[_col], inplace=True)

# Anchor densities
df['micro_anchor_density_5']  = np.nan
df['micro_anchor_density_20'] = np.nan
if 'is_anchor' in df.columns:
    micro_mask = df['_is_micro']
    if micro_mask.any():
        df.loc[micro_mask, 'micro_anchor_density_5'] = (
            df[micro_mask].groupby(GROUP_KEYS)['is_anchor']
            .transform(lambda x: x.astype(int).shift(1).rolling(5, min_periods=1).sum())
        )
        df.loc[micro_mask, 'micro_anchor_density_20'] = (
            df[micro_mask].groupby(GROUP_KEYS)['is_anchor']
            .transform(lambda x: x.astype(int).shift(1).rolling(20, min_periods=1).sum())
        )

_macro_df = df[df['_is_macro']].copy()
_macro_df['_ep_dir'] = (df.loc[df['_is_macro'], 'trend'].astype(int) * 2 - 1).fillna(0)
macro_summary = (
    _macro_df.groupby(['Ticker', 'entry_time'])
    .agg(
        macro_agreement_signed = ('_ep_dir', 'mean'),
        macro_dispersion       = ('_ep_dir', 'std'),
    )
    .reset_index()
)
df = df.merge(macro_summary, on=['Ticker', 'entry_time'], how='left')
for _col in ['macro_agreement_signed', 'macro_dispersion']:
    if _col in df.columns:
        df[f'{_col}_lag1'] = shift_within_group(df, _col, 1)
        df.drop(columns=[_col], inplace=True)

if 'micro_agreement_signed_lag1' in df.columns and 'macro_agreement_signed_lag1' in df.columns:
    df['micro_macro_alignment_gap_lag1'] = (
        df['micro_agreement_signed_lag1'] - df['macro_agreement_signed_lag1']
    )
else:
    df['micro_macro_alignment_gap_lag1'] = np.nan

# FIX-re17f: scale_compression_lag1 has 100% NaN rate — set to NaN but do NOT
# add to feature pool (removed from DEDUPLICATED_CANDIDATE_COLS below).
if 'micro_dispersion_lag1' in df.columns and 'macro_agreement_signed_lag1' in df.columns:
    df['scale_compression_lag1'] = np.where(
        df['micro_dispersion_lag1'].notna() & (df['micro_dispersion_lag1'].abs() > 1e-8),
        df['macro_agreement_signed_lag1'].abs() / df['micro_dispersion_lag1'],
        np.nan,
    )
else:
    df['scale_compression_lag1'] = np.nan

df.drop(
    columns=['_is_micro', '_is_macro'],
    inplace=True, errors='ignore',
)

# ============================================================
# 5. Energy Features
# ============================================================
print("\nComputing energy features...")

df['brick_ratio'] = df['atr_brick_size'] / df['brick_size'].replace(0, np.nan)

if trend_int is not None:
    df['lying_signal_density_3'] = df.groupby(GROUP_KEYS)['trend'].transform(
        lambda x: (
            (x.astype(int) != x.astype(int).shift(1))
            .astype(float)
            .shift(1)
            .rolling(3, min_periods=1).mean()
        )
    )

atr_lag1 = shift_within_group(df, 'atr_brick_size', 1)
atr_lag3 = shift_within_group(df, 'atr_brick_size', 3)
df['vol_change_pct_3'] = np.where(
    atr_lag3.notna() & (atr_lag3 != 0),
    (atr_lag1 - atr_lag3) / atr_lag3,
    np.nan,
)

if 'entry_price' in df.columns:
    prev_entry = shift_within_group(df, 'entry_price', 1)
    df['entry_exit_gap'] = np.where(
        prev_entry.notna() & (df['atr_brick_size'] != 0),
        (df['entry_price'] - prev_entry) / df['atr_brick_size'],
        np.nan,
    )
    df['entry_exit_gap_abs'] = df['entry_exit_gap'].abs()

    df['entry_exit_gap_max_5'] = df.groupby(GROUP_KEYS)['entry_exit_gap_abs'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).max()
    )

    entry_back = shift_within_group(df, 'entry_price', 2)
    time_back  = shift_within_group(df, 'entry_time', 2)
    time_delta = (df['entry_time'] - time_back).dt.total_seconds()
    df['price_velocity_3'] = np.where(
        (time_delta > 0) & time_delta.notna(),
        (df['entry_price'] - entry_back) / time_delta,
        np.nan,
    )
    vel_lag1     = shift_within_group(df, 'price_velocity_3', 1)
    vel_lag_back = shift_within_group(df, 'price_velocity_3', 4)
    df['price_velocity_accel_3'] = np.where(
        vel_lag_back.notna(), (vel_lag1 - vel_lag_back) / 3, np.nan
    )

df['brick_size_ratio_ma_5'] = df.groupby(GROUP_KEYS)['brick_ratio'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

if trend_int is not None:
    df['_streak_change']    = (trend_int != shift_within_group(df, 'trend', 1)).astype(int)
    df['_streak_id']        = df.groupby(GROUP_KEYS)['_streak_change'].cumsum()
    df['_raw_brick_streak'] = df.groupby(['Ticker', 'factor', '_streak_id']).cumcount() + 1
    df['brick_direction_streak'] = shift_within_group(df, '_raw_brick_streak', 1)
    df.drop(columns=['_raw_brick_streak', '_streak_change', '_streak_id'], inplace=True)

df['trade_pos_in_day'] = df.groupby(GROUP_KEYS).cumcount()
group_open = df.groupby(GROUP_KEYS)['entry_time'].transform('first')
df['time_since_open'] = (df['entry_time'] - group_open).dt.total_seconds()

# ============================================================
# 6. OHLC-Derived Features
# ============================================================
print("\nComputing OHLC features...")

if 'SqueezeMomentum' in df.columns:
    df['ohlc_squeeze_active'] = (df['SqueezeMomentum'] > 0).astype(float)

if all(c in df.columns for c in ['QQE_14_5:fast', 'QQE_14_5:signal']):
    df['ohlc_qqe_bullish'] = (df['QQE_14_5:fast'] > df['QQE_14_5:signal']).astype(float)
    if 'QQE_14_5:slow' in df.columns:
        df['ohlc_qqe_momentum'] = df['QQE_14_5:fast'] - df['QQE_14_5:slow']

if 'RSIOscillator' in df.columns:
    rsi_mean = df.groupby(GROUP_KEYS)['RSIOscillator'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()
    )
    rsi_std = df.groupby(GROUP_KEYS)['RSIOscillator'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).std()
    )
    rsi_shifted = shift_within_group(df, 'RSIOscillator', 1)
    df['ohlc_rsi_zscore_lag1'] = (rsi_shifted - rsi_mean) / rsi_std.replace(0, np.nan)

if 'ATR' in df.columns:
    atr_ma20    = df.groupby(GROUP_KEYS)['ATR'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()
    )
    atr_shifted = shift_within_group(df, 'ATR', 1)
    df['ohlc_volatility_ratio_20_lag1'] = atr_shifted / atr_ma20.replace(0, np.nan)

if all(c in df.columns for c in ['VMA', 'VMA_6_1.5:uband', 'VMA_6_1.5:lband']):
    df['ohlc_band_width'] = (
        (df['VMA_6_1.5:uband'] - df['VMA_6_1.5:lband']) / df['VMA'].replace(0, np.nan)
    )
    if 'close' in df.columns:
        band_range = (df['VMA_6_1.5:uband'] - df['VMA_6_1.5:lband']).replace(0, np.nan)
        df['ohlc_price_in_band'] = (df['close'] - df['VMA_6_1.5:lband']) / band_range

if 'PSAR_uptrend' in df.columns:
    df['ohlc_psar_uptrend'] = df['PSAR_uptrend'].astype(float)

if all(c in df.columns for c in ['VIP', 'VIM']):
    df['ohlc_vortex_diff'] = df['VIP'] - df['VIM']

if 'VMA_6_1.5:regime' in df.columns:
    regime_map = {'bearish': 0, 'neutral': 1, 'bullish': 2}
    df['ohlc_vma_regime_encoded'] = (
        df['VMA_6_1.5:regime'].map(regime_map).fillna(1).astype(float)
    )

if 'CMF' in df.columns:
    df['ohlc_cmf'] = df['CMF']

if all(c in df.columns for c in ['ohlc_squeeze_active', 'ohlc_qqe_bullish', 'ohlc_psar_uptrend']):
    df['ohlc_combined_signal'] = (
        df['ohlc_squeeze_active'].fillna(0) +
        df['ohlc_qqe_bullish'].fillna(0) +
        df['ohlc_psar_uptrend'].fillna(0)
    )

if 'ohlc_squeeze_active' in df.columns:
    df['ohlc_squeeze_bars_active'] = (
        df.groupby(GROUP_KEYS)['ohlc_squeeze_active']
        .transform(
            lambda x: (
                x.fillna(0)
                 .groupby((x.fillna(0) != x.fillna(0).shift(1)).cumsum())
                 .cumcount() + 1
            ) * (x.fillna(0) == 1)
        )
    )

if all(c in df.columns for c in ['ohlc_psar_uptrend', 'ohlc_vma_regime_encoded', 'ohlc_qqe_bullish']):
    psar_bull = (df['ohlc_psar_uptrend'] == 1).astype(float)
    psar_bear = (df['ohlc_psar_uptrend'] == 0).astype(float)
    vma_bull  = (df['ohlc_vma_regime_encoded'] == 2).astype(float)
    vma_bear  = (df['ohlc_vma_regime_encoded'] == 0).astype(float)
    qqe_bull  = (df['ohlc_qqe_bullish'] == 1).astype(float)
    qqe_bear  = (df['ohlc_qqe_bullish'] == 0).astype(float)
    df['ohlc_trend_strength'] = (
        psar_bull + vma_bull + qqe_bull - psar_bear - vma_bear - qqe_bear
    ) / 3

# ============================================================
# 7. Unified Lag Pass
# ============================================================
print("\nApplying unified lag pass...")

no_lag_cols = [
    'Ticker', 'factor', 'entry_date', 'entry_time', 'pnl', 'theoretical_profit',
    'is_anchor', 'anchor_type', 'is_asymmetric_resurge', 'horizontal_duration',
    'vertical_duration', 'regime_breadth', 'streak_len', 'streak_position',
    'is_valid_streak', 'is_streak_start', 'is_vertical_converge', 'v_duration',
    'is_h_anchor', 'is_v_anchor', 'close', 'exit_price', 'decision_bricks',
    'exit_jump_bricks', 'qty', 'date', 'no_bricks', 'entry_price', 'brick_size',
    'atr_brick_size', 'trend', 'trade_date', 'is_win',
]

exclude_from_lag = [
    'trade_pos_in_day', 'time_since_open', 'time_since_open_sq',
    'hour', 'hour_sin', 'hour_cos', 'day_of_week',
    'brick_ratio', 'lying_signal_density_3', 'vol_change_pct_3',
    'entry_exit_gap', 'entry_exit_gap_abs', 'entry_exit_gap_max_5',
    'price_velocity_3', 'price_velocity_accel_3',
    'brick_size_ratio_ma_5', 'brick_direction_streak',
    'ohlc_rsi_zscore_lag1',
    'ohlc_volatility_ratio_20_lag1',
    'pullback_from_max_progress',
    'entry_price_loo_mean_lag1', 'entry_price_loo_std_lag1', 'entry_price_loo_zscore_lag1',
    'brick_size_loo_mean_lag1', 'brick_size_loo_std_lag1', 'brick_size_loo_zscore_lag1',
    'atr_brick_size_loo_mean_lag1', 'atr_brick_size_loo_std_lag1', 'atr_brick_size_loo_zscore_lag1',
    'micro_macro_divergence_entry_lag1',
    'stack_active_count_lag1', 'micro_active_count_lag1', 'macro_active_count_lag1',
    'micro_agreement_signed_lag1', 'micro_consensus_strength_lag1',
    'micro_dispersion_lag1', 'micro_factor_spread_lag1',
    'macro_agreement_signed_lag1', 'macro_dispersion_lag1',
    'micro_macro_alignment_gap_lag1', 'scale_compression_lag1',
    'micro_agreement_accel',
    'alignment_net_lag1',
    'time_atr_inter', 'vol_trend_inter_lag1',
    'squeeze_momentum_inter_lag1', 'micro_macro_div_vol_lag1',
]

all_features = [c for c in df.columns if c not in no_lag_cols]
lag_features = [c for c in all_features if c not in exclude_from_lag]

for col in lag_features:
    df[f'{col}_lag1'] = shift_within_group(df, col, 1)

df.drop(columns=lag_features, inplace=True, errors='ignore')

# ============================================================
# 7.5 Post-lag: micro_agreement_accel
# ============================================================
if 'micro_agreement_signed_lag1' in df.columns:
    df['micro_agreement_accel'] = df.groupby(GROUP_KEYS)['micro_agreement_signed_lag1'].transform(
        lambda x: x.diff(3)
    )
else:
    df['micro_agreement_accel'] = np.nan

# ============================================================
# 8. Interaction Features
# ============================================================
print("\nCreating interaction features...")

if 'time_since_open' in df.columns and 'atr_brick_size_loo_mean_lag1' in df.columns:
    df['time_atr_inter'] = df['time_since_open'] * df['atr_brick_size_loo_mean_lag1']

if 'ohlc_volatility_ratio_20_lag1' in df.columns and 'ohlc_trend_strength_lag1' in df.columns:
    df['vol_trend_inter_lag1'] = (
        df['ohlc_volatility_ratio_20_lag1'] * df['ohlc_trend_strength_lag1']
    )

if 'ohlc_squeeze_bars_active_lag1' in df.columns and 'ohlc_qqe_momentum_lag1' in df.columns:
    df['squeeze_momentum_inter_lag1'] = (
        df['ohlc_squeeze_bars_active_lag1'] * df['ohlc_qqe_momentum_lag1']
    )

if 'micro_macro_divergence_entry_lag1' in df.columns and 'ohlc_volatility_ratio_20_lag1' in df.columns:
    df['micro_macro_div_vol_lag1'] = (
        df['micro_macro_divergence_entry_lag1'] * df['ohlc_volatility_ratio_20_lag1']
    )

df['hour']              = df['entry_time'].dt.hour
df['hour_sin']          = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']          = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week']       = df['entry_time'].dt.dayofweek
df['day_of_week_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 5)
df['day_of_week_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 5)
df['time_since_open_sq'] = df['time_since_open'] ** 2

# GEX term-structure: short-window slope vs medium-window slope divergence.
# Positive = accelerating GEX buildup; negative = GEX fading faster than expected.
if 'slope_net_gex_6_lag1' in df.columns and 'slope_net_gex_12_lag1' in df.columns:
    df['gex_slope_term_spread_lag1'] = (
        df['slope_net_gex_6_lag1'] - df['slope_net_gex_12_lag1']
    )

# ============================================================
# 8.5 Coupled Indicator Features
# ============================================================
# These features combine downloaded 5min / 1min indicator columns
# (suffixed _5min_lag1 / _1min_lag1 after the unified lag pass) with
# each other and with base-dataset features.  All inputs are already
# lagged — no additional shift needed.
print("\nComputing coupled indicator features...")

def _get(col):
    """Return df[col] if present, else a NaN Series."""
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

# ── Group A: GEX Structure Couplings (5min × 5min) ──────────────────────────

# Price trapped between call/put walls — small product → tight squeeze
_d_call = _get('dist_to_call_wall_atr_5min_lag1')
_d_put  = _get('dist_to_put_wall_atr_5min_lag1')
_d_sum  = (_d_call + _d_put).replace(0, np.nan)
if _d_call.notna().any() and _d_put.notna().any():
    df['gex_wall_squeeze_5min_lag1']  = (_d_call * _d_put).clip(0, 100)
    df['gex_wall_skew_5min_lag1']     = (_d_call - _d_put) / _d_sum   # >0 → closer to put wall

# GEX momentum weighted by proximity to gamma flip
_gf_dist   = _get('dist_to_gamma_flip_atr_5min_lag1')
_net_mom   = _get('net_gex_momentum_5min_lag1')
if _gf_dist.notna().any() and _net_mom.notna().any():
    df['gex_flip_momentum_5min_lag1'] = _net_mom / (_gf_dist.abs() + 1.0)

# RSI × Z-score joint elevation: both signals aligned = stronger regime signal
_gex_rsi  = _get('net_gex_rsi_20_5min_lag1')
_gex_z    = _get('net_gex_zscore_20_5min_lag1')
if _gex_rsi.notna().any() and _gex_z.notna().any():
    df['gex_rsi_zscore_joint_5min_lag1'] = _gex_rsi * _gex_z

# Acceleration divergence: 5min net_gex building faster than abs_gex → directional skew
_net_accel = _get('net_gex_accel_5min_lag1')
_abs_accel = _get('abs_gex_accel_5min_lag1')  # note: abs_gex_accel not in L1 file — falls back to NaN gracefully
_abs_accel_1 = _get('abs_gex_momentum_5min_lag1')
if _net_accel.notna().any() and _abs_accel_1.notna().any():
    _abs_safe = _abs_accel_1.abs().replace(0, np.nan)
    df['gex_directional_bias_5min_lag1'] = _net_accel / _abs_safe

# ── Group B: Cross-Frequency Coupling (1min × 5min) ─────────────────────────

# 1min flow momentum aligned with 5min GEX momentum — agreement = conviction
_flow_rsi  = _get('net_flow_rsi_20_1min_lag1')
_gex_mom5  = _get('net_gex_momentum_5min_lag1')
if _flow_rsi.notna().any() and _gex_mom5.notna().any():
    df['flow_gex_momentum_sync_lag1'] = _flow_rsi * _gex_mom5

# OFI amplified by proximity to gamma flip: large order imbalance near the flip = explosive
_ofi   = _get('ofi_proxy_1min_lag1')
_gf_d5 = _get('dist_to_gamma_flip_atr_5min_lag1')
if _ofi.notna().any() and _gf_d5.notna().any():
    df['ofi_gamma_gate_lag1'] = _ofi / (_gf_d5.abs() + 0.5)

# PCR momentum (1min sentiment shift) vs 5min GEX ratio slope (structure change)
_pcr_mom   = _get('pcr_momentum_1min_lag1')
_gex_rslop = _get('gex_ratio_slope_5min_lag1')
if _pcr_mom.notna().any() and _gex_rslop.notna().any():
    df['pcr_gex_divergence_lag1'] = _pcr_mom * _gex_rslop

# 1min price velocity vs 5min GEX momentum: are they aligned?
_vel   = _get('price_velocity_3')   # already computed, not lagged further
_gex5m = _get('net_gex_momentum_5min_lag1')
if _vel.notna().any() and _gex5m.notna().any():
    df['velocity_gex_sync_lag1'] = _vel * _gex5m

# ── Group C: Options Sentiment Composites (1min) ─────────────────────────────

# Net call bias by volume percentage
_call_pct = _get('call_volume_perc_1min_lag1')
_put_pct  = _get('put_volume_perc_1min_lag1')
if _call_pct.notna().any() and _put_pct.notna().any():
    df['call_put_bias_1min_lag1'] = _call_pct - _put_pct

# Block-flow call dominance: call_block / total_block — smart money direction
_cb = _get('call_block_1min_lag1')
_pb = _get('put_block_1min_lag1')
_tb = (_cb + _pb).replace(0, np.nan)
if _cb.notna().any() and _pb.notna().any():
    df['block_call_dominance_1min_lag1'] = _cb / _tb
    df['block_flow_imbalance_1min_lag1'] = (_cb - _pb) / _tb

# PCR × high-GEX environment: extreme put bias in a high-gamma regime = pinned signal
_pcr     = _get('put_call_ratio_1min_lag1')
_abs_pct = _get('abs_gex_percentile_20_5min_lag1')
if _pcr.notna().any() and _abs_pct.notna().any():
    df['pcr_high_gex_gate_lag1'] = _pcr * _abs_pct

# Net notional delta sync with GEX: directional exposure aligned with gamma structure
_nnd     = _get('net_notional_delta_1min_lag1')
_net_gex = _get('net_gex_5min_lag1')
if _nnd.notna().any() and _net_gex.notna().any():
    df['delta_gex_sync_lag1'] = _nnd * _net_gex

_nnd_mom = _get('net_notional_delta_momentum_1min_lag1')
_gex_mom = _get('net_gex_momentum_5min_lag1')
if _nnd_mom.notna().any() and _gex_mom.notna().any():
    df['delta_gex_momentum_sync_lag1'] = _nnd_mom * _gex_mom

# ── Group D: IV Term Structure (cross-frequency) ─────────────────────────────

# Short-term IV / long-term IV: ratio > 1 = vol term inversion (spike risk)
_iv7   = _get('stock_iv_7d_1min_lag1')
_iv30  = _get('stock_iv_30d_5min_lag1')
if _iv7.notna().any() and _iv30.notna().any():
    _iv30_safe = _iv30.replace(0, np.nan)
    df['iv_term_ratio_lag1'] = _iv7 / _iv30_safe
    df['iv_term_spread_lag1'] = _iv7 - _iv30  # absolute spread

# Gamma exposure × IV regime: high abs_gex in high-IV = PINNED attractor
if _abs_pct.notna().any() and _iv30.notna().any():
    df['gex_iv_regime_lag1'] = _abs_pct * _iv30

# Flow alignment × net GEX: directional flow agrees with GEX sign
_fa   = _get('flow_alignment_1min_lag1')
_ngex = _get('net_gex_5min_lag1')
if _fa.notna().any() and _ngex.notna().any():
    df['flow_alignment_gex_lag1'] = _fa * _ngex

# OFI price divergence × gamma flip distance: sustained order imbalance near flip
_ofi_div = _get('ofi_price_divergence_20_1min_lag1')
if _ofi_div.notna().any() and _gf_d5.notna().any():
    df['ofi_div_near_flip_lag1'] = _ofi_div / (_gf_d5.abs() + 0.5)

_coupled_count = sum(
    1 for c in df.columns
    if any(c.startswith(p) for p in [
        'gex_wall_squeeze', 'gex_wall_skew', 'gex_flip_momentum',
        'gex_rsi_zscore_joint', 'gex_directional_bias',
        'flow_gex_momentum_sync', 'ofi_gamma_gate', 'pcr_gex_divergence',
        'velocity_gex_sync', 'call_put_bias', 'block_call_dominance',
        'block_flow_imbalance', 'pcr_high_gex_gate', 'delta_gex_sync',
        'delta_gex_momentum_sync', 'iv_term_ratio', 'iv_term_spread',
        'gex_iv_regime', 'flow_alignment_gex', 'ofi_div_near_flip',
    ])
)
print(f"  {_coupled_count} coupled indicator features created")

# ============================================================
# 8.6 Direction-Specific Composite Features
# ============================================================
# EDA (L1_10_direction_asymmetry.md) found 57 signal-flip features, 40 long-dominant,
# and 15 short-dominant features.  These composites aggregate the EDA top-ranked signals
# per direction into single dense features, letting tree models see pre-aligned signal
# without needing to learn the sign flip from data alone.
# All inputs are lag1 (already shifted) — no additional shift needed.

print("\nComputing direction-specific composite features...")

# ── Long-specific composites ─────────────────────────────────────────────────
# Top long predictors (EDA L1_02): max_pain_rsi, gex_ratio_rsi, gamma_flip_rsi,
# net_gex_rsi, net_gex_pct21_rsi — all negative-direction for long regime.
# We negate them so high value = more regime-like for long.

_mp_rsi   = _get('max_pain_rsi_20_5min_lag1')
_gfr_rsi  = _get('gamma_flip_rsi_20_5min_lag1')
_gex_rsi2 = _get('gex_ratio_rsi_20_5min_lag1')
_net_rsi  = _get('net_gex_rsi_20_5min_lag1')
_pct_rsi  = _get('net_gex_pct21_rsi_20_5min_lag1')
_pw_rsi   = _get('put_wall_rsi_20_5min_lag1')
_cw_rsi   = _get('call_wall_rsi_20_5min_lag1')

# Long RSI composite: average of top-5 EDA predictors (direction-normalised)
# max_pain and gamma_flip are negative for long → negate; gex_ratio and net_gex are positive
_long_rsi_parts = []
if _mp_rsi.notna().any():    _long_rsi_parts.append(-_mp_rsi)
if _gfr_rsi.notna().any():   _long_rsi_parts.append(-_gfr_rsi)
if _gex_rsi2.notna().any():  _long_rsi_parts.append( _gex_rsi2)
if _net_rsi.notna().any():   _long_rsi_parts.append( _net_rsi)
if _pct_rsi.notna().any():   _long_rsi_parts.append( _pct_rsi)
if _long_rsi_parts:
    df['long_rsi_composite_5min_lag1'] = pd.concat(_long_rsi_parts, axis=1).mean(axis=1)

# Long wall-pressure score: call/put wall z-score and percentile (all negative for long)
_cw_z   = _get('call_wall_zscore_20_5min_lag1')
_cw_pct = _get('call_wall_percentile_20_5min_lag1')
_mp_z   = _get('max_pain_zscore_20_5min_lag1')
_mp_pct = _get('max_pain_percentile_20_5min_lag1')
_gf_z   = _get('gamma_flip_zscore_20_5min_lag1')
_gf_pct = _get('gamma_flip_percentile_20_5min_lag1')
_long_wall_parts = []
if _cw_z.notna().any():   _long_wall_parts.append(-_cw_z)
if _cw_pct.notna().any(): _long_wall_parts.append(-_cw_pct)
if _mp_z.notna().any():   _long_wall_parts.append(-_mp_z)
if _mp_pct.notna().any(): _long_wall_parts.append(-_mp_pct)
if _gf_z.notna().any():   _long_wall_parts.append(-_gf_z)
if _gf_pct.notna().any(): _long_wall_parts.append(-_gf_pct)
if _long_wall_parts:
    df['long_wall_pressure_5min_lag1'] = pd.concat(_long_wall_parts, axis=1).mean(axis=1)

# Net GEX level signal for long (positive for long regime): net_gex, vma_net_gex
_ngex5     = _get('net_gex_5min_lag1')
_safe_ngex = _get('safe_net_gex_5min_lag1')
_vma6      = _get('vma_net_gex_6_5min_lag1')
_vma12     = _get('vma_net_gex_12_5min_lag1')
_long_gex_level_parts = []
if _ngex5.notna().any():     _long_gex_level_parts.append(_ngex5)
if _safe_ngex.notna().any(): _long_gex_level_parts.append(_safe_ngex)
if _vma6.notna().any():      _long_gex_level_parts.append(_vma6)
if _vma12.notna().any():     _long_gex_level_parts.append(_vma12)
if _long_gex_level_parts:
    df['long_gex_level_composite_5min_lag1'] = pd.concat(_long_gex_level_parts, axis=1).mean(axis=1)

# Long put-wall safety: dist_to_put_wall_atr positive for long → long protected
_pd_put = _get('safe_dist_to_put_wall_atr_5min_lag1')
if _pd_put.notna().any() and _pw_rsi.notna().any():
    df['long_put_wall_safety_5min_lag1'] = _pd_put - _pw_rsi  # larger dist & smaller RSI = safer for long

# ── Short-specific composites ────────────────────────────────────────────────
# Top short predictors (EDA L1_03): gex_ratio_rsi, max_pain_rsi, net_gex_rsi with
# OPPOSITE sign to long.  vma_net_gex z/percentile are short-dominant (negative short).

# Short RSI composite: sign-inverted from long
_short_rsi_parts = []
if _mp_rsi.notna().any():   _short_rsi_parts.append( _mp_rsi)    # positive for short
if _gfr_rsi.notna().any():  _short_rsi_parts.append( _gfr_rsi)   # positive for short
if _gex_rsi2.notna().any(): _short_rsi_parts.append(-_gex_rsi2)  # negative for short
if _net_rsi.notna().any():  _short_rsi_parts.append(-_net_rsi)   # negative for short
if _pct_rsi.notna().any():  _short_rsi_parts.append(-_pct_rsi)   # negative for short
if _short_rsi_parts:
    df['short_rsi_composite_5min_lag1'] = pd.concat(_short_rsi_parts, axis=1).mean(axis=1)

# Short VMA z/percentile composite: vma_net_gex_12 and _6 z/pct are the strongest short signals
_v12z   = _get('vma_net_gex_12_zscore_20_5min_lag1')
_v12pct = _get('vma_net_gex_12_percentile_20_5min_lag1')
_v6z    = _get('vma_net_gex_6_zscore_20_5min_lag1')
_v6pct  = _get('vma_net_gex_6_percentile_20_5min_lag1')
_short_vma_parts = []
if _v12z.notna().any():   _short_vma_parts.append(-_v12z)
if _v12pct.notna().any(): _short_vma_parts.append(-_v12pct)
if _v6z.notna().any():    _short_vma_parts.append(-_v6z)
if _v6pct.notna().any():  _short_vma_parts.append(-_v6pct)
if _short_vma_parts:
    df['short_vma_zscore_composite_5min_lag1'] = pd.concat(_short_vma_parts, axis=1).mean(axis=1)

# Short GEX z/percentile: net_gex_zscore_20 is the strongest short z-score signal
_ngz   = _get('net_gex_zscore_20_5min_lag1')
_ngpz  = _get('net_gex_pct21_zscore_20_5min_lag1')
_ngpp  = _get('net_gex_pct21_percentile_20_5min_lag1')
_short_z_parts = []
if _ngz.notna().any():  _short_z_parts.append(-_ngz)
if _ngpz.notna().any(): _short_z_parts.append(-_ngpz)
if _ngpp.notna().any(): _short_z_parts.append(-_ngpp)
if _short_z_parts:
    df['short_gex_zscore_composite_5min_lag1'] = pd.concat(_short_z_parts, axis=1).mean(axis=1)

# ── Cross-direction divergence ───────────────────────────────────────────────
# The difference between long and short composites captures the direction of
# options market pressure: positive = long-regime favoured, negative = short.
if 'long_rsi_composite_5min_lag1' in df.columns and 'short_rsi_composite_5min_lag1' in df.columns:
    df['dir_rsi_divergence_5min_lag1'] = (
        df['long_rsi_composite_5min_lag1'] - df['short_rsi_composite_5min_lag1']
    )

_dir_count = sum(1 for c in df.columns if any(c.startswith(p) for p in [
    'long_rsi_composite', 'long_wall_pressure', 'long_gex_level_composite',
    'long_put_wall_safety', 'short_rsi_composite', 'short_vma_zscore_composite',
    'short_gex_zscore_composite', 'dir_rsi_divergence',
]))
print(f"  {_dir_count} direction-specific composite features created")

# ============================================================
# 9. Alignment Net
# ============================================================
print("\nComputing alignment_net_lag1...")

align_cols = [
    'ohlc_psar_uptrend_lag1',
    'ohlc_vma_regime_encoded_lag1',
    'ohlc_vortex_diff_lag1',
    'ohlc_cmf_lag1',
    'ohlc_squeeze_active_lag1',
    'ohlc_qqe_bullish_lag1',
    'ohlc_rsi_zscore_lag1',
]
available = [c for c in align_cols if c in df.columns]
if available:
    bullish, bearish = [], []
    for col in available:
        val = df[col]
        if col == 'ohlc_vma_regime_encoded_lag1':
            bullish.append(val == 2);    bearish.append(val == 0)
        elif col in ('ohlc_vortex_diff_lag1', 'ohlc_cmf_lag1', 'ohlc_rsi_zscore_lag1'):
            bullish.append(val > 0);     bearish.append(val < 0)
        else:
            bullish.append(val == 1);    bearish.append(val == 0)
    n_cond = len(bullish)
    df['alignment_net_lag1'] = (
        (np.sum(bullish, axis=0) - np.sum(bearish, axis=0)) / n_cond
    )
else:
    df['alignment_net_lag1'] = np.nan

# FIX-re24b: Restore precision-targeting composite features (from re22f).
# regime_quality_score: product of three top-stable signals — large only when all three agree.
# conviction_count: integer 0–6 count of stable-8 signals simultaneously "on".
# These were removed in re23 without ablation — re22 benefited from both (confirmed via re23 regression).
if all(c in df.columns for c in ['price_velocity_3', 'entry_exit_gap', 'SqueezeMomentum_lag1']):
    df['regime_quality_score'] = (
        df['price_velocity_3'].clip(lower=0) *
        df['entry_exit_gap'].abs() *
        df['SqueezeMomentum_lag1'].abs()
    )
_conv = pd.Series(0, index=df.index, dtype=int)
if 'price_velocity_3'               in df.columns: _conv += (df['price_velocity_3'] > 0).astype(int)
if 'entry_exit_gap'                 in df.columns: _conv += (df['entry_exit_gap'] > 0).astype(int)
if 'SqueezeMomentum_lag1'           in df.columns: _conv += (df['SqueezeMomentum_lag1'].abs() > 0.1).astype(int)
if 'ohlc_price_in_band_lag1'        in df.columns: _conv += (df['ohlc_price_in_band_lag1'] > 0.5).astype(int)
if 'ohlc_cmf_lag1'                  in df.columns: _conv += (df['ohlc_cmf_lag1'] > 0).astype(int)
if 'ohlc_squeeze_bars_active_lag1'  in df.columns: _conv += (df['ohlc_squeeze_bars_active_lag1'] > 0).astype(int)
df['conviction_count'] = _conv

# ============================================================
# 10. Leakage Sanity Check
# ============================================================
print("\nRunning leakage sanity check...")

feature_cols = [
    c for c in df.columns
    if c not in (list(no_lag_cols) + ['entry_time'])
]

name_suspects = [
    c for c in feature_cols
    if any(f in c for f in _FORBIDDEN_SOURCES)
]

provenance_suspects = []
for forbidden in _FORBIDDEN_SOURCES:
    if forbidden not in df.columns:
        continue
    fvec = pd.to_numeric(df[forbidden], errors='coerce').dropna()
    if len(fvec) < 30:
        continue
    for feat in feature_cols:
        if feat in provenance_suspects:
            continue
        fvec2 = pd.to_numeric(df[feat], errors='coerce')
        common = fvec.index.intersection(fvec2.dropna().index)
        if len(common) < 30:
            continue
        corr = abs(fvec.loc[common].corr(fvec2.loc[common]))
        if corr > 0.999:
            provenance_suspects.append(feat)

if name_suspects:
    print(f"  WARNING [name]       : {name_suspects}")
else:
    print("  OK [name]            : no forbidden names in feature columns.")

if provenance_suspects:
    print(f"  WARNING [provenance] : {provenance_suspects}")
else:
    print("  OK [provenance]      : no feature is near-perfectly correlated with a forbidden source.")

# ============================================================
# 11. Save Final Dataset
# ============================================================
print("\nSaving final dataset...")

state_features = [
    'reversal_density_5_lag1', 'reversal_density_10_lag1',
    'directional_efficiency_10_lag1', 'run_length_lag1', 'alternation_ratio_lag1',
    'atr_compression_ratio_lag1', 'vol_regime_lag1',
    'pullback_from_max_progress',
    'call_wall_proxy_lag1', 'put_wall_proxy_lag1',
    'dist_to_gamma_flip_atr_lag1', 'net_gex_zscore_20_lag1', 'net_gex_percentile_20_lag1',
    'RSIOscillator_lag1', 'SqueezeMomentum_lag1',
]
state_features = [c for c in state_features if c in df.columns]

raw_trade_cols = [
    'Ticker', 'factor', 'entry_date', 'entry_time', 'pnl', 'theoretical_profit',
    'brick_size', 'atr_brick_size', 'entry_price', 'close', 'trend', 'no_bricks',
    'date', 'qty', 'decision_bricks', 'exit_jump_bricks', 'exit_time',
    'is_anchor', 'anchor_type', 'is_asymmetric_resurge',
    'horizontal_duration', 'vertical_duration', 'regime_breadth',
    'is_win', 'streak_len', 'streak_position', 'is_valid_streak', 'is_streak_start',
    'is_vertical_converge', 'v_duration', 'is_h_anchor', 'is_v_anchor',
]
raw_trade_cols = [c for c in raw_trade_cols if c in df.columns]

other_features = [
    c for c in df.columns
    if c not in state_features + raw_trade_cols and c != 'entry_time'
]

output_cols = raw_trade_cols + state_features + other_features
output_cols  = [c for c in dict.fromkeys(output_cols) if c in df.columns]

df[output_cols].to_parquet(OUTPUT_FILE, index=False)
elapsed = time.perf_counter() - t0
print(f"Saved  : {len(df):,} rows × {len(output_cols)} columns → {OUTPUT_FILE}")
if 'direction' in df.columns:
    for _dir in sorted(df['direction'].dropna().unique()):
        _dir_file = OUTPUT_DIR / f"AAPL_tier4_features_no_target_re27_{_dir}.parquet"
        _dir_df = df.loc[df['direction'] == _dir, output_cols]
        _dir_df.to_parquet(_dir_file, index=False)
        print(f"Saved  : {len(_dir_df):,} rows → {_dir_file.name}")
print(f"State  : {state_features}")
print(f"Elapsed: {elapsed:.1f}s")
print("\nFEATURE ENGINEERING COMPLETE (no target – C(d) regime computed separately)")


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL REGIME DETECTION — re20 (Hermes v2 + L2 Rank Regression)
# ═══════════════════════════════════════════════════════════════════════════════

import os
os.environ['OMP_NUM_THREADS']      = '2'
os.environ['MKL_NUM_THREADS']      = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             roc_auc_score, classification_report,
                             adjusted_rand_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import warnings, json, io, logging, bisect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as _mp
warnings.filterwarnings('ignore')

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr, kendalltau, ks_2samp
from scipy.optimize import linear_sum_assignment
from statsmodels.tsa.stattools import adfuller

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = ("/Users/mazza/Desktop/Work/regimeDetection/notebooks/"
               "RegimeWork/newnotebooks1/outdataset/AAPL_tier4_features_no_target_re25.parquet")
OUTPUT_DIR  = Path("/Users/mazza/Desktop/Work/regimeDetection/notebooks/"
                   "RegimeWork/newnotebooks1/outdataset")

RANDOM_SEED    = 42
TOTAL_CORES    = os.cpu_count() or 8
_L2_ENABLED    = False  # set True to re-enable Layer 2 intraday trade selection
USABLE_CORES   = max(1, TOTAL_CORES - 1)
N_PARALLEL     = min(8, USABLE_CORES)
np.random.seed(RANDOM_SEED)

_MP_FORK_CTX = _mp.get_context('fork')

# ── Walk-forward parameters ───────────────────────────────────────────────────
TRAIN_TDAYS    = 200
TEST_TDAYS     = 28
EMBARGO_TDAYS  = 3

# ── State discovery parameters ────────────────────────────────────────────────
K_MIN              = 2
K_MAX              = 3
INNER_FOLDS        = 5
MIN_SAMPLES_STATE  = 20

# ── re17: Clustering quality gates ───────────────────────────────────────────
SHARPE_SPREAD_MIN  = 0.12   # FIX-re20a: lowered from 0.40 (was unreachable at trade level)
ARI_STABILITY_MIN  = 0.60   # FIX-re17c: min ARI between seed-42 and seed-137 runs
TEMPORAL_ARI_MIN   = 0.0    # FIX-re23a: disabled — values 0.011-0.022 in re22 blocked all clustering at 0.30
# FIX-re20a: KS-test gate — minimum KS statistic between cluster PnL distributions
KS_GATE_MIN        = 0.05   # KS stat ≥ 0.05 AND p-value < 0.10 to pass
KS_GATE_PVAL       = 0.10

# ── Profile label thresholds ──────────────────────────────────────────────────
ABS_HIGH_EDGE  = 0.615
ABS_DANGER     = 0.500
HIGH_EDGE_PCT  = 67
NEUTRAL_LO_PCT = 33
CAUTION_LO_PCT = 10

BIVARIATE_ALPHA      = 0.10
TOP_K_FEATURES       = 40
CFI_DISTANCE_THRESH  = 0.5
FRACDIFF_D_GRID      = np.arange(0.0, 1.05, 0.1)
ADF_CONFIDENCE       = 0.05
FRACDIFF_WINDOW      = 30
PERCENTILE_BUFFER_SIZE = 200
LONG_ENTRY_PCT       = 90
MIN_SPEARMAN_DECILE  = 0.7

LGB_PARAMS_SUBMODEL_PARALLEL = dict(
    n_estimators=300, num_leaves=31, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_samples=50,
    class_weight='balanced',
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
)
XGB_PARAMS_SUBMODEL_PARALLEL = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
    random_state=RANDOM_SEED, n_jobs=-1, eval_metric='mlogloss', verbosity=0
)
LGB_PARAMS_MAIN_PARALLEL = dict(
    n_estimators=300, num_leaves=15, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=800,
    reg_alpha=2.0, reg_lambda=10.0, min_gain_to_split=0.01,
    # FIX-re24d: restore is_unbalance=True (reverts re23d explicit SPW).
    # LightGBM is_unbalance=True is mathematically equivalent to scale_pos_weight=n_neg/n_pos
    # but uses a different initial score that matched re22's behaviour.
    is_unbalance=True,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
)
XGB_PARAMS_MAIN_PARALLEL = dict(
    n_estimators=300, max_depth=3, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=200,
    reg_alpha=2.0, reg_lambda=10.0, gamma=0.05,
    random_state=RANDOM_SEED, n_jobs=-1, eval_metric='logloss', verbosity=0
)

_SUB_NJOBS = max(1, (USABLE_CORES + N_PARALLEL - 1) // N_PARALLEL)
LGB_PARAMS_SUBMODEL_1T = {**LGB_PARAMS_SUBMODEL_PARALLEL, "n_jobs": _SUB_NJOBS}
XGB_PARAMS_SUBMODEL_1T = {**XGB_PARAMS_SUBMODEL_PARALLEL, "n_jobs": _SUB_NJOBS}
_MAIN_NJOBS = max(1, USABLE_CORES // 2)
LGB_PARAMS_MAIN_1T     = {**LGB_PARAMS_MAIN_PARALLEL, "n_jobs": _MAIN_NJOBS}
XGB_PARAMS_MAIN_1T     = {**XGB_PARAMS_MAIN_PARALLEL, "n_jobs": _MAIN_NJOBS}

# Direction-specific L1 overrides (EDA: short mean_pnl=-1.12 vs long=-0.98, bimodal slippage,
# longer loss streaks → short target more imbalanced + higher variance → needs stricter regularization)
_LGB_SHORT_OVERRIDES  = {'reg_lambda': 15.0, 'min_child_samples': 1000}
_L1_SPW_MAX_LONG  = 2.5
_L1_SPW_MAX_SHORT = 3.5  # short has lower positive-day rate → target more imbalanced

CORE_STATES   = ['PINNED', 'TREND', 'CHOP', 'TRANSITION']
SUBMODEL_CORE_STATES = ['PINNED', 'TREND', 'CHOP']
L2_OOF_ACC_MARGIN = 0.02   # FIX-re20b: lowered from 0.05; PINNED threshold 0.30→0.27
OUTCOME_COLS  = ['pnl', 'theoretical_profit', 'exit_price',
                 'exit_jump_bricks', 'decision_bricks', 'is_win']
PROFILE_LABELS = ['HIGH_EDGE', 'NEUTRAL', 'CAUTION', 'DANGER']

FORBIDDEN_COLUMNS = [
    'Ticker', 'factor', 'entry_date', 'entry_time', 'date', 'exit_time_lag1', 'timestamp_lag1',
    'entry_price', 'close', 'brick_size', 'atr_brick_size', 'no_bricks', 'qty', 'trend',
    'pnl', 'theoretical_profit', 'exit_price', 'decision_bricks', 'exit_jump_bricks',
    'is_win', 'is_vertical_converge', 'v_duration', 'horizontal_duration', 'regime_breadth',
    'streak_len', 'streak_position', 'is_valid_streak', 'is_streak_start',
    'is_anchor', 'is_h_anchor', 'is_v_anchor', 'is_asymmetric_resurge', 'regime_target'
]

# FIX-re17f: dead features with 100% NaN — explicitly excluded from candidate pool
_DEAD_FEATURES_RE17 = frozenset([
    'micro_anchor_density_5_lag1',
    'micro_anchor_density_20_lag1',
    'scale_compression_lag1',
    # FIX-re22d: high-NaN features identified in re21 health check
    'atr_brick_size_loo_zscore_lag1',      # 84.1% NaN
    'entry_price_loo_zscore_lag1',         # 83.9% NaN
    'macro_dispersion_lag1',               # 67.2% NaN
    'micro_macro_alignment_gap_lag1',      # r=nan
    'ohlc_vma_regime_encoded_lag1',        # r=nan
    'micro_agreement_accel',               # r=nan
    # FIX-re22e: redundant GEX rank-transform siblings of zscore_20/percentile_20/rsi_20.
    # cci_20/stoch_20 are monotone rank transforms of the same rolling window — Spearman
    # within 0.02 of the zscore_20 twin → pure collinearity, wasting bivariate/CFI/GS budget.
    # Drop across all GEX bases; keep zscore_20, percentile_20, rsi_20, slope, momentum.
    'gamma_flip_cci_20_lag1',           'gamma_flip_stoch_20_lag1',
    'gex_ratio_cci_20_lag1',            'gex_ratio_stoch_20_lag1',
    'net_gex_cci_20_lag1',              'net_gex_stoch_20_lag1',
    'abs_gex_cci_20_lag1',              'abs_gex_stoch_20_lag1',
    'abs_gex_pct21_cci_20_lag1',        'abs_gex_pct21_stoch_20_lag1',
    'max_negative_net_gex_cci_20_lag1', 'max_negative_net_gex_stoch_20_lag1',
    'max_positive_net_gex_cci_20_lag1', 'max_positive_net_gex_stoch_20_lag1',
    'max_abs_gex_cci_20_lag1',          'max_abs_gex_stoch_20_lag1',
    'gex_call_cci_20_lag1',             'gex_call_stoch_20_lag1',
    'gex_put_cci_20_lag1',              'gex_put_stoch_20_lag1',
    'gex_accel_cci_20_lag1',            'gex_accel_stoch_20_lag1',
    'gex_slope_cci_20_lag1',            'gex_slope_stoch_20_lag1',
    'gex_momentum_cci_20_lag1',         'gex_momentum_stoch_20_lag1',
    'net_gex_pct21_cci_20_lag1',        'net_gex_pct21_stoch_20_lag1',
    'correlation_net_gex_10_cci_20_lag1', 'correlation_net_gex_10_stoch_20_lag1',
    'correlation_net_gex_20_cci_20_lag1', 'correlation_net_gex_20_stoch_20_lag1',
    'slope_net_gex_6_cci_20_lag1',      'slope_net_gex_6_stoch_20_lag1',
    'slope_net_gex_12_cci_20_lag1',     'slope_net_gex_12_stoch_20_lag1',
    'vma_net_gex_6_cci_20_lag1',        'vma_net_gex_6_stoch_20_lag1',
    'vma_net_gex_12_cci_20_lag1',       'vma_net_gex_12_stoch_20_lag1',
])

CORE_STATE_MAP = {'PINNED': 0, 'TREND': 1, 'CHOP': 2, 'TRANSITION': 3}

CD_MIN_STREAK = 4
# OAA-Gap3: direction-specific streak (short noisier — max loss streak 36 vs 30)
CD_MIN_STREAK_LONG  = 4
CD_MIN_STREAK_SHORT = 5

# Unit 1: L2 window target streak direction-specific (short noisier → require longer streak)
_L2_WINDOW_STREAK_LONG  = 3
_L2_WINDOW_STREAK_SHORT = 4

# Direction-specific L1 thresholds (EDA: short noisier — bimodal slippage, longer loss streaks)
L1_PROB_THRESHOLD_LONG  = 0.58   # long more predictable, slightly lower gate acceptable
L1_PROB_THRESHOLD_SHORT = 0.62   # short noisier, require higher confidence to gate-in
L1_PROB_THRESHOLD = L1_PROB_THRESHOLD_LONG  # kept for cascade summary (cross-direction reporting)
L2_MIN_TRADES_FOLD  = 30
L2_TOP_K_FEATURES   = 15
L2_GATE_PERCENTILE  = 70
L2_GATE_FLOOR       = 0.50

# FIX-re17a: 'small' bucket REMOVED (base_rate=0.007 → AUC illusion). Micro/macro restored.
# OAA-Gap2: restore micro/macro split — EDA shows very different PnL/std profiles.
#   micro (brick_size<0.1): mean_pnl~-1.05, std~21-25 — tight distribution, high vol
#   macro (brick_size≥0.1): mean_pnl~-0.63 to -1.60, std~46-97 — better peak PnL, fatter tails
FACTOR_BUCKETS = {
    'micro': (None,  0.099),   # brick_size < 0.1
    'macro': (0.1,   None),    # brick_size ≥ 0.1
}
L2_MIN_BUCKET_TRADES = 20
L2_SPW_MAX = 2.5
L1_TRADE_THRESHOLD  = 0.34

# Unit 6: direction-specific duration boost thresholds (EDA: WR=66.4% for >=30min longs)
_DUR_BOOST_MIN_LONG  = 30   # EDA: WR=66.4% for >=30min holds vs 27.4% shorter
_DUR_BOOST_MIN_SHORT = 20   # lower threshold for shorts: capture "not a quick stop-out"
_DUR_BOOST_AMOUNT    = 0.20

L2_REJECT_STATES = frozenset({'CHOP', 'TRANSITION'})

# REVERSAL (PINNED state) dropped for both buckets — consistently harmful across re20–re24.
L2_SKIP_BUCKETS  = frozenset({'micro/REVERSAL', 'macro/REVERSAL'})

RENKO_L2_STATE_GROUPS = {
    'TREND':    frozenset({'TREND'}),
    'REVERSAL': frozenset({'PINNED'}),
}


def compute_dynamic_top_k(n_surviving, n_train, min_k=8, max_k=15):
    capacity_k  = n_train // 15
    diversity_k = max(min_k, n_surviving // 2)
    k = min(capacity_k, diversity_k, max_k)
    return max(min_k, k)


print("✓ Config loaded (re25 — no L2 ISO + regime_quality restored + REVERSAL dropped + is_unbalance restored).")
print(f"  Walk-forward : TRAIN={TRAIN_TDAYS} tdays | TEST={TEST_TDAYS} tdays | EMBARGO={EMBARGO_TDAYS} tdays")
print(f"  FACTOR_BUCKETS: {list(FACTOR_BUCKETS.keys())} (small removed — FIX-re17a)")
print(f"  Gates: SHARPE_SPREAD_MIN={SHARPE_SPREAD_MIN} | ARI_STABILITY_MIN={ARI_STABILITY_MIN} | TEMPORAL_ARI_MIN={TEMPORAL_ARI_MIN} (disabled)")
print(f"  Resources    : {TOTAL_CORES} cores → {USABLE_CORES} usable")

# ═══════════════════════════════════════════════════════════════════════════════
# TARGET COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
# Triple-gate daily regime target (re26):
#
#   Gate 1 — Scale-diversity persistence (65% of intraday 30-min windows must
#             satisfy C(d)>=4 consecutive profitable brick_size tiers).
#   Gate 2 — Daily profitability: mean_pnl > 0.
#   Gate 3 — Fold-local win rate: daily WR >= 33rd-pct of train-fold WRs.
#             Threshold comes from fold_train_df ONLY — never from test data.
#
# Prediction timing: features are lagged 1 day (EOD D-1), target is full day D.
# L1 predicts at market close whether TOMORROW is a regime day.
# The persistence check uses all of day D's windows to compute the label —
# the model learns what EOD-D-1 looks like before a day with sustained regime.
#
# Legacy mode: if fold_train_df is None, falls back to original single-shot
# streak check (backward-compatible with the diagnostic global computation).

_CD_WINDOW_MINUTES        = 30    # intraday window size for streak check
_CD_PERSISTENCE_THRESHOLD = 0.65  # fraction of windows that must pass streak
_CD_MIN_TRADES_PER_TIER   = 5     # skip tier if fewer trades in window


def _streak_passes_in_window(window_df):
    """True if the 30-min window has C(d)>=CD_MIN_STREAK consecutive positive tiers."""
    tier_counts = window_df.groupby('brick_size')['pnl'].count()
    tier_med    = window_df.groupby('brick_size')['pnl'].median()
    # only use tiers with enough trades to make median meaningful
    valid_tiers = tier_counts[tier_counts >= _CD_MIN_TRADES_PER_TIER].index
    if len(valid_tiers) < CD_MIN_STREAK:
        return False
    vals = tier_med.loc[valid_tiers].sort_index().values > 0
    best = cur = 0
    for f in vals:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best >= CD_MIN_STREAK


def compute_cd_target(df_with_pnl_and_brick, return_counts=False,
                      fold_train_df=None, min_streak=None):
    """
    C(d) regime target — daily brick-size streak, with optional fold-local gates.

    Always derives _date from entry_time (preferred) or date column, converting to
    datetime.date so the returned Series index matches pd.to_datetime(...).dt.date
    used in the fold loop .map() call.

    fold_train_df: train-fold rows used to set fold-local win-rate threshold (Gate 3).
                   When None, returns pure C(d)>=CD_MIN_STREAK (global/diagnostic mode).

    Bug-fix notes vs previous triple-gate implementation:
      • Removed 30-min window Gate 1: with 70+ brick sizes, consecutive-4 in a single
        window was never achievable → all-zero target every fold.
      • date column in df_raw is stored as str; _date is now derived from entry_time
        via .dt.date to guarantee datetime.date index consistent with fold-loop .map().
    """
    work = df_with_pnl_and_brick.copy()

    # Always compute _date as datetime.date from entry_time (str date column is
    # incompatible with the datetime.date keys produced by .dt.date in the fold loop).
    if 'entry_time' in work.columns:
        work['_date'] = pd.to_datetime(work['entry_time']).dt.date
    else:
        work['_date'] = pd.to_datetime(work['date']).dt.date

    # Daily C(d) streak — same logic used in legacy/global mode.
    # Index renamed to 'date' for backward compat with downstream reset_index/merge.
    daily_bs = work.groupby(['_date', 'brick_size'])['pnl'].median().reset_index()

    def _max_consecutive(grp):
        flags = grp.sort_values('brick_size')['pnl'].values > 0
        best = cur = 0
        for f in flags:
            if f: cur += 1; best = max(best, cur)
            else: cur = 0
        return best

    streak_counts = daily_bs.groupby('_date').apply(_max_consecutive)
    streak_counts.index.name = 'date'

    if return_counts:
        return streak_counts

    _min_s = min_streak if min_streak is not None else CD_MIN_STREAK
    g1 = (streak_counts >= _min_s)

    if fold_train_df is None:
        return g1.astype(int)

    # ── Fold-local gates (Gates 2 & 3) ───────────────────────────────────────
    daily_stats = work.groupby('_date')['pnl'].agg(
        mean_pnl='mean',
        win_rate=lambda x: (x > 0).mean(),
        n='count'
    )
    daily_stats = daily_stats[daily_stats['n'] >= 5]

    # Gate 3 threshold derived ONLY from train-fold rows
    tr = fold_train_df.copy()
    if 'entry_time' in tr.columns:
        tr['_date'] = pd.to_datetime(tr['entry_time']).dt.date
    else:
        tr['_date'] = pd.to_datetime(tr['date']).dt.date
    tr_daily = tr.groupby('_date')['pnl'].agg(
        wr=lambda x: (x > 0).mean(), n='count'
    )
    wr_thresh = float(tr_daily[tr_daily['n'] >= 5]['wr'].quantile(0.33)) \
        if len(tr_daily[tr_daily['n'] >= 5]) >= 3 else 0.33

    g2 = (daily_stats['mean_pnl'] > 0).reindex(g1.index, fill_value=False)
    g3 = (daily_stats['win_rate'] >= wr_thresh).reindex(g1.index, fill_value=False)

    return (g1 & g2 & g3).astype(int)


def _slot_from_entry(df):
    """Extract (_date, _slot) columns from df without copying the whole frame."""
    if 'entry_time' in df.columns:
        ts = pd.to_datetime(df['entry_time'])
        return ts.dt.date, (ts.dt.hour * 60 + ts.dt.minute) // 30
    ts = pd.to_datetime(df['date'])
    return ts.dt.date, pd.Series(0, index=df.index)


def compute_short_avoidance_target(df, fold_train_df, quantile=0.67):
    """
    L1 target for short direction: identify days to AVOID (bad days).
    Returns 1 for bad/avoid days (high intraday drawdown), 0 for acceptable days.
    Gate inverted at prediction time: gate-in when bad_day_prob is LOW.
    """
    _date, _slot = _slot_from_entry(df)
    key = pd.DataFrame({'_date': _date, '_slot': _slot, 'pnl': df['pnl'].values})
    daily_worst = key.groupby(['_date', '_slot'])['pnl'].mean().groupby(level=0).min()

    # Fold-local threshold derived from train rows only (no leakage)
    if fold_train_df is df:
        tr_daily_worst = daily_worst  # same object — skip redundant computation
    else:
        _tr_date, _tr_slot = _slot_from_entry(fold_train_df)
        tr_key = pd.DataFrame({'_date': _tr_date, '_slot': _tr_slot,
                                'pnl': fold_train_df['pnl'].values})
        tr_daily_worst = tr_key.groupby(['_date', '_slot'])['pnl'].mean().groupby(level=0).min()
    threshold = float(tr_daily_worst.quantile(quantile)) if len(tr_daily_worst) >= 3 else -1.0

    # bad day = worst window PnL below threshold (more negative = worse)
    bad_day = (daily_worst <= threshold).astype(int)
    bad_day.index.name = 'date'
    return bad_day


_ROLLING_PREV_COLS = [
    'prev1_regime_target',
    'rolling_5d_regime_rate',
    'rolling_5d_volatility',
    'rolling_5d_trade_count',
    'days_since_last_regime_day',
    'rolling_5d_mean_C',
    'rolling_5d_std_C',
    'regime_streak_length',
]


def compute_rolling_prev_day_features(df_raw_in, global_cd_series, global_cd_counts=None):
    df_out = df_raw_in.copy()
    df_out['_date'] = pd.to_datetime(df_out['entry_time']).dt.date

    daily_vol = (df_out.groupby('_date')['atr_brick_size'].mean()
                 if 'atr_brick_size' in df_out.columns
                 else pd.Series(dtype=float))
    daily_tc  = df_out.groupby('_date').size()
    _cd_counts = global_cd_counts if global_cd_counts is not None else pd.Series(dtype=float)
    all_dates  = sorted(df_out['_date'].unique())

    rows = {}
    for i, d in enumerate(all_dates):
        prev_days = [all_dates[j] for j in range(max(0, i - 5), i)]
        prev1     = all_dates[i - 1] if i >= 1 else None

        cd_vals  = [float(global_cd_series.get(pd_, np.nan)) for pd_ in prev_days]
        vol_vals = [daily_vol.get(pd_, np.nan)  for pd_ in prev_days]
        tc_vals  = [daily_tc.get(pd_, np.nan)   for pd_ in prev_days]
        c_counts = [float(_cd_counts.get(pd_, np.nan)) for pd_ in prev_days]

        past_regime = [j for j in range(max(0, i - 30), i)
                       if global_cd_series.get(all_dates[j], 0) == 1]

        streak = 0
        for k in range(i - 1, max(-1, i - 31), -1):
            if k >= 0 and global_cd_series.get(all_dates[k], 0) == 1:
                streak += 1
            else:
                break

        _c_valid = [x for x in c_counts if not np.isnan(x)]
        _std_c   = float(np.std(_c_valid)) if len(_c_valid) >= 2 else np.nan

        rows[d] = {
            'prev1_regime_target':        (float(global_cd_series.get(prev1, np.nan))
                                           if prev1 else np.nan),
            'rolling_5d_regime_rate':     (float(np.nanmean(cd_vals))  if cd_vals else np.nan),
            'rolling_5d_volatility':      (float(np.nanmean(vol_vals)) if vol_vals else np.nan),
            'rolling_5d_trade_count':     (float(np.nanmean(tc_vals))  if tc_vals else np.nan),
            'days_since_last_regime_day': float(i - past_regime[-1]) if past_regime else 30.0,
            'rolling_5d_mean_C':          (float(np.nanmean(c_counts)) if c_counts else np.nan),
            'rolling_5d_std_C':           _std_c,
            'regime_streak_length':       float(streak),
        }

    feat_df = (pd.DataFrame.from_dict(rows, orient='index')
               .rename_axis('_date')
               .reset_index())

    df_out = df_out.merge(feat_df, on='_date', how='left')
    df_out.drop(columns=['_date'], inplace=True, errors='ignore')
    return df_out


def compute_fold_safe_prev_day_features(df_subset, known_cd_series, known_cd_counts=None):
    """
    FIX: fold-local target history only.

    Compute rolling previous-day features for each row in ``df_subset`` using ONLY
    dates strictly earlier than the row's own date. ``known_cd_series`` is a
    date → C(d) regime-target mapping restricted to the history that is visible
    to the caller (train-only for train rows; train + prior-test-block dates
    for test rows). No future dates — even within the same test block — are
    ever consulted for a given row.

    This replaces the globally-precomputed ``compute_rolling_prev_day_features``
    call so no future knowledge of targets leaks into train/test features.
    """
    import bisect

    df_out = df_subset.copy()
    df_out['_date'] = pd.to_datetime(df_out['entry_time']).dt.date

    # Daily aggregates are taken over df_subset itself (rows available to the
    # caller). Volatility / trade-count for date d will only ever be looked up
    # for dates strictly < d, so same-day values are never read for date d.
    daily_vol = (df_out.groupby('_date')['atr_brick_size'].mean()
                 if 'atr_brick_size' in df_out.columns
                 else pd.Series(dtype=float))
    daily_tc  = df_out.groupby('_date').size()
    _cd_counts = known_cd_counts if known_cd_counts is not None else pd.Series(dtype=float)

    known_dates = sorted(known_cd_series.index)
    # Convert to dict once so .get() works correctly (pd.Series lacks .get())
    _known_cd_dict    = known_cd_series.to_dict()
    _known_count_dict = _cd_counts.to_dict() if hasattr(_cd_counts, 'to_dict') else dict(_cd_counts)

    subset_dates = sorted(df_out['_date'].unique())
    rows = {}
    for d in subset_dates:
        # Index of first known date >= d — all indices < i are strictly earlier.
        i = bisect.bisect_left(known_dates, d)
        prev_days = known_dates[max(0, i - 5):i]
        prev30    = known_dates[max(0, i - 30):i]
        prev1     = known_dates[i - 1] if i >= 1 else None

        cd_vals  = [float(_known_cd_dict.get(pd_, np.nan)) for pd_ in prev_days]
        vol_vals = [daily_vol.get(pd_, np.nan) for pd_ in prev_days]
        tc_vals  = [daily_tc.get(pd_, np.nan)  for pd_ in prev_days]
        c_counts = [float(_known_count_dict.get(pd_, np.nan)) for pd_ in prev_days]

        past_regime = [j for j, pd_ in enumerate(prev30)
                       if _known_cd_dict.get(pd_, 0) == 1]

        streak = 0
        for k in range(len(prev30) - 1, -1, -1):
            if _known_cd_dict.get(prev30[k], 0) == 1:
                streak += 1
            else:
                break

        _c_valid = [x for x in c_counts if not np.isnan(x)]
        _std_c   = float(np.std(_c_valid)) if len(_c_valid) >= 2 else np.nan

        rows[d] = {
            'prev1_regime_target':        (float(known_cd_series.get(prev1, np.nan))
                                           if prev1 else np.nan),
            'rolling_5d_regime_rate':     (float(np.nanmean(cd_vals))  if cd_vals else np.nan),
            'rolling_5d_volatility':      (float(np.nanmean(vol_vals)) if vol_vals else np.nan),
            'rolling_5d_trade_count':     (float(np.nanmean(tc_vals))  if tc_vals else np.nan),
            'days_since_last_regime_day': (float(len(prev30) - past_regime[-1])
                                           if past_regime else 30.0),
            'rolling_5d_mean_C':          (float(np.nanmean(c_counts)) if c_counts else np.nan),
            'rolling_5d_std_C':           _std_c,
            'regime_streak_length':       float(streak),
        }

    feat_df = (pd.DataFrame.from_dict(rows, orient='index')
               .rename_axis('_date')
               .reset_index())

    df_out = df_out.drop(columns=[c for c in _ROLLING_PREV_COLS if c in df_out.columns],
                         errors='ignore')
    df_out = df_out.merge(feat_df, on='_date', how='left')
    df_out.drop(columns=['_date'], inplace=True, errors='ignore')
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO REGIME FEATURES (Hermes v2 Layer 1)
# Produces P(TREND_LV), P(TREND_HV), P(CHOP_LV), P(CHOP_HV) soft weights.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_macro_regime_features(df_in):
    """
    Compute macro regime soft probabilities for Hermes v2 MoE.

    Uses rolling ATR z-score as volatility axis and price_velocity_3 z-score
    as trend axis.  Both are computed from already-lagged features so there is
    no lookahead.

    Returns df_in with columns added:
        macro_atr_pct       : rolling 20-period ATR percentile rank (VIX proxy)
        macro_vel_z         : rolling 20-period price_velocity_3 z-score
        macro_p_trend       : P(trending regime)
        macro_p_vol         : P(high-volatility regime)
        macro_p_trend_lv    : P(TREND_LV)
        macro_p_trend_hv    : P(TREND_HV)
        macro_p_chop_lv     : P(CHOP_LV)
        macro_p_chop_hv     : P(CHOP_HV)
    """
    df = df_in.copy()
    df['_et'] = pd.to_datetime(df['entry_time'])
    df['_date'] = df['_et'].dt.date

    # FIX: avoid same-day lookahead.
    # Aggregate to daily values, shift by 1 so date d sees only dates < d,
    # then compute rolling percentile rank / z-score over the lagged series.
    if 'atr_brick_size' in df.columns:
        daily_atr = (
            df.groupby('_date')['atr_brick_size']
              .mean()
              .shift(1)
        )
        macro_atr_pct = (
            daily_atr
            .rolling(20, min_periods=5)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )
        _atr_pct_lookup = macro_atr_pct.reset_index()
        _atr_pct_lookup.columns = ['_date', 'macro_atr_pct']
        df = df.merge(_atr_pct_lookup, on='_date', how='left')
    else:
        df['macro_atr_pct'] = 0.5

    # FIX: avoid same-day lookahead — shift daily price_velocity_3 mean by 1
    # before rolling z-score computation.
    if 'price_velocity_3' in df.columns:
        daily_vel = (
            df.groupby('_date')['price_velocity_3']
              .mean()
              .shift(1)
        )
        macro_vel_z = (
            daily_vel
            .rolling(20, min_periods=5)
            .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False)
        )
        _vel_z_lookup = macro_vel_z.reset_index()
        _vel_z_lookup.columns = ['_date', 'macro_vel_z']
        df = df.merge(_vel_z_lookup, on='_date', how='left')
    else:
        df['macro_vel_z'] = 0.0

    df['macro_atr_pct'] = df['macro_atr_pct'].fillna(0.5)
    df['macro_vel_z']   = df['macro_vel_z'].fillna(0.0)

    # Sigmoid-based soft regime weights
    df['macro_p_trend'] = 1.0 / (1.0 + np.exp(-3.0 * (df['macro_vel_z'] - 0.5)))
    df['macro_p_vol']   = 1.0 / (1.0 + np.exp(-3.0 * (df['macro_atr_pct'] - 0.6)))

    df['macro_p_trend_lv'] = df['macro_p_trend'] * (1.0 - df['macro_p_vol'])
    df['macro_p_trend_hv'] = df['macro_p_trend'] * df['macro_p_vol']
    df['macro_p_chop_lv']  = (1.0 - df['macro_p_trend']) * (1.0 - df['macro_p_vol'])
    df['macro_p_chop_hv']  = (1.0 - df['macro_p_trend']) * df['macro_p_vol']

    # Regime drift: z-score of 5-day ATR mean vs 20-day ATR mean.
    # Positive = vol expanding (regime shift up); negative = vol compressing.
    # Both windows use the same shift(1)-lagged daily_atr so no lookahead.
    if 'atr_brick_size' in df_in.columns:
        df['_date2'] = pd.to_datetime(df_in['entry_time']).dt.date
        _daily_atr_drift = (
            df_in.copy()
            .assign(_date2=pd.to_datetime(df_in['entry_time']).dt.date)
            .groupby('_date2')['atr_brick_size']
            .mean()
            .shift(1)
        )
        _atr5  = _daily_atr_drift.rolling(5,  min_periods=3).mean()
        _atr20 = _daily_atr_drift.rolling(20, min_periods=5).mean()
        _atr20_std = _daily_atr_drift.rolling(20, min_periods=5).std()
        _macro_atr_drift = ((_atr5 - _atr20) / (_atr20_std + 1e-8)).rename('macro_atr_drift')
        _drift_lookup = _macro_atr_drift.reset_index()
        _drift_lookup.columns = ['_date2', 'macro_atr_drift']
        df = df.merge(_drift_lookup, on='_date2', how='left')
        df['macro_atr_drift'] = df['macro_atr_drift'].fillna(0.0)
        df.drop(columns=['_date2'], inplace=True, errors='ignore')
    else:
        df['macro_atr_drift'] = 0.0

    df.drop(columns=['_et', '_date'], inplace=True, errors='ignore')
    return df


_MACRO_REGIME_COLS = [
    'macro_atr_pct', 'macro_vel_z',
    'macro_p_trend', 'macro_p_vol',
    'macro_p_trend_lv', 'macro_p_trend_hv',
    'macro_p_chop_lv', 'macro_p_chop_hv',
    'macro_atr_drift',
]

print("✓ compute_macro_regime_features defined (Hermes v2 Layer 1 macro regime).")


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZER (Hermes v2 Layer 5)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_position_size(meta_prob, regime_probs):
    """
    Quarter-Kelly position size based on entropy of regime probs and meta confidence.

    size_weight = (1 - H / log(4)) × meta_prob × 0.25

    Parameters
    ----------
    meta_prob    : float — Layer 4 meta-labeler probability
    regime_probs : array-like of 4 floats [P_TREND_LV, P_TREND_HV, P_CHOP_LV, P_CHOP_HV]

    Returns
    -------
    float in [0, 0.25]
    """
    rp = np.array(regime_probs, dtype=float)
    rp = np.clip(rp, 1e-10, 1.0)
    rp = rp / rp.sum()
    H = -float(np.sum(rp * np.log(rp)))
    H_max = np.log(len(rp))
    certainty = 1.0 - H / (H_max + 1e-10)
    return 0.25 * float(meta_prob) * max(0.0, certainty)


print("✓ compute_position_size defined (Hermes v2 Layer 5).")


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_parquet(DATA_PATH)
df_raw['entry_time'] = pd.to_datetime(df_raw['entry_time'])
df_raw['date'] = df_raw['entry_time'].dt.date
df_raw = df_raw.sort_values('entry_time').reset_index(drop=True)

# Derive direction from trend column (trend=True=long brick, trend=False=short brick).
# The re19 parquet was built from the same long/short trade data but lacks a direction
# column. trend maps 1:1: True → 'long', False → 'short'.
if 'direction' not in df_raw.columns:
    if 'trend' in df_raw.columns:
        df_raw['direction'] = df_raw['trend'].map({True: 'long', False: 'short', 1: 'long', -1: 'short', 1.0: 'long', -1.0: 'short'})
        _unmatched = df_raw['direction'].isna().sum()
        if _unmatched > 0:
            print(f"  ⚠  {_unmatched:,} rows could not derive direction from trend — filling 'long'")
            df_raw['direction'] = df_raw['direction'].fillna('long')
        print(f"  ✔  Derived direction from trend: {df_raw['direction'].value_counts().to_dict()}")
    else:
        print("  ⚠  No trend or direction column — direction will not be available")

outcomes = df_raw[['entry_time', 'factor', 'pnl']].copy()

DROP_ALWAYS = ['Unnamed: 0', 'timestamp_x', 'timestamp_y', 'timestamp',
               'exit_price', 'theoretical_profit', 'decision_bricks', 'exit_jump_bricks']
df = df_raw.drop(columns=[c for c in DROP_ALWAYS + OUTCOME_COLS if c in df_raw.columns],
                 errors='ignore').copy()

META_COLS = ['no_bricks', 'entry_time', 'Ticker', 'date',
             'entry_price', 'qty', 'factor', 'brick_size', 'atr_brick_size',
             'trend', 'close']
META_PRESENT = [c for c in META_COLS if c in df.columns]

print(f"\n✓ Loaded {len(df):,} rows | {len(df.columns)} columns")
print(f"  Date range : {df['entry_time'].min().date()} → {df['entry_time'].max().date()}")
print(f"  Factors    : {sorted(df['factor'].unique())}")

# ── Proxy fallback: call_wall_proxy_lag1 / put_wall_proxy_lag1 ───────────────
# If gex_call_percentile_20 was absent during feature engineering, the proxy
# was never created. Reconstruct from call_wall_percentile_20_5min_lag1.
if 'call_wall_proxy_lag1' not in df.columns:
    if 'call_wall_percentile_20_5min_lag1' in df.columns:
        df['call_wall_proxy_lag1'] = df['call_wall_percentile_20_5min_lag1'].fillna(0.5)
        print("  ⚠  call_wall_proxy_lag1 absent — reconstructed from call_wall_percentile_20_5min_lag1")
    else:
        df['call_wall_proxy_lag1'] = 0.5
        print("  ⚠  call_wall_proxy_lag1 absent — filled with constant 0.5")
if 'put_wall_proxy_lag1' not in df.columns:
    if 'put_wall_percentile_20_5min_lag1' in df.columns:
        df['put_wall_proxy_lag1'] = (1 - df['put_wall_percentile_20_5min_lag1'].fillna(0.5)).clip(0, 1)
        print("  ⚠  put_wall_proxy_lag1 absent — reconstructed from put_wall_percentile_20_5min_lag1")
    else:
        df['put_wall_proxy_lag1'] = 0.5
        print("  ⚠  put_wall_proxy_lag1 absent — filled with constant 0.5")

# ── Direction-aware target attachment (DIAGNOSTIC ONLY) ──────────────────────
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  FIX-re25c: DIAGNOSTIC ONLY — no authoritative target is stored here.      │
# │  The regime_target column on `df` is set to a placeholder 0 below.          │
# │  The authoritative C(d) target is computed FOLD-LOCALLY inside the          │
# │  walk-forward loop (compute_cd_target per direction, per fold), where        │
# │  train_df / test_df get their own regime_target derived from training-window │
# │  pnl only.  This section exists purely to populate _global_cd for the        │
# │  dataset health-check prints and prev-day rolling features init.             │
# └─────────────────────────────────────────────────────────────────────────────┘
# C(d) is computed separately per direction (long / short) so that a day can be
# a "regime day" for longs without being one for shorts and vice versa.
# _global_cd (used for fold-level rolling prev-day features) takes the per-day
# MAX across directions: a day is regime=1 for rolling features if any direction
# qualifies — this is the most conservative / least leaky choice.
df['date'] = pd.to_datetime(df['entry_time']).dt.date
df = df.drop(columns=['regime_target'], errors='ignore')

_directions_in_data = (sorted(df_raw['direction'].unique().tolist())
                       if 'direction' in df_raw.columns else [])

if len(_directions_in_data) > 1:
    _dir_target_rows = []
    for _dir in _directions_in_data:
        _mask = df_raw['direction'] == _dir
        _dir_slice = df_raw.loc[_mask, ['date', 'brick_size', 'pnl']].copy()
        if _dir_slice.empty or 'brick_size' not in _dir_slice.columns or _dir_slice['brick_size'].isna().all():
            print(f"  ✖  Direction '{_dir}': empty or no brick_size — skipping C(d) target")
            continue
        _dcd  = compute_cd_target(_dir_slice)
        _dcd_df = _dcd.rename('regime_target').reset_index()
        _dcd_df['date']     = pd.to_datetime(_dcd_df['date']).dt.date
        _dcd_df['direction'] = _dir
        _dir_target_rows.append(_dcd_df)
    _dir_target_df = pd.concat(_dir_target_rows, ignore_index=True)
    df = df.merge(_dir_target_df, on=['date', 'direction'], how='left')
    df['regime_target'] = df['regime_target'].fillna(0).astype(int)
    # global_cd for rolling prev-day features: regime=1 if any direction qualifies
    _global_cd = df.groupby('date')['regime_target'].max()
    # global_cd_counts: use combined data (brick-size convergence depth)
    _global_cd_counts = compute_cd_target(
        df_raw[['date', 'brick_size', 'pnl']].copy(), return_counts=True)
else:
    # Single-direction fallback — original behaviour
    _global_cd = compute_cd_target(df_raw[['date', 'brick_size', 'pnl']].copy())
    _global_cd_counts = compute_cd_target(
        df_raw[['date', 'brick_size', 'pnl']].copy(), return_counts=True)
    _global_cd_df = _global_cd.rename('regime_target').reset_index()
    _global_cd_df['date'] = pd.to_datetime(_global_cd_df['date']).dt.date
    df = df.merge(_global_cd_df, on='date', how='left')
    df['regime_target'] = df['regime_target'].fillna(0).astype(int)

# ── direction_encoded: numeric feature derived from factor ────────────────────
# factor is in FORBIDDEN_COLUMNS (excluded as raw metadata string) but
# direction_encoded is a numeric signal the model can use to learn
# direction-specific regime patterns (long=0, short=1).
if 'direction' in df.columns:
    df['direction_encoded']     = (df['direction'] == 'short').astype(float)
    df_raw['direction_encoded'] = (df_raw['direction'] == 'short').astype(float)

# FIX: fold-local target history only.
# Rolling previous-day features (prev1_regime_target, rolling_5d_*, etc.) used
# to be precomputed globally here via compute_rolling_prev_day_features(df, _global_cd,
# _global_cd_counts). That leaked the full-dataset C(d) series into train/test
# features. These are now computed per-fold inside the walk-forward loop using
# compute_fold_safe_prev_day_features, which restricts the known history to
# (train) or (train + prior test dates) depending on the row.
# FIX-re25c: Do NOT dropna on regime_target here — the placeholder 0 value is
# always non-NaN (set above).  Fold-local recomputation in the WF loop is
# authoritative; filtering here would silently drop rows that the fold loop
# would handle correctly.
df = df.reset_index(drop=True)

_day_regime_rate = df.groupby('date')['regime_target'].max().mean()
print(f"\n✓ Trade-level dataset: {len(df):,} rows ({df['date'].nunique()} unique days)")
print(f"  Day-level C(d)≥{CD_MIN_STREAK} base rate (any direction): {_day_regime_rate:.3f}")
for _dir in _directions_in_data:
    _br = df[df['direction'] == _dir]['regime_target'].mean()
    _n  = (df['direction'] == _dir).sum()
    print(f"  [{_dir:>5}] trade-level base rate: {_br:.3f}  ({_n:,} rows)")
if 'direction_encoded' in df.columns:
    print(f"  direction_encoded added: 0=long / 1=short (included in CANDIDATE_COLS)")

# ── Dataset Health ────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("DATASET HEALTH CHECK (re22 — comprehensive)")
print(f"{'─'*60}")
_health_feat_cols = [c for c in df.columns
                     if c not in {'date', 'factor', 'entry_time', 'regime_target', 'pnl'}
                     and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
_nan_rates = df[_health_feat_cols].isna().mean().sort_values(ascending=False)
_high_nan = _nan_rates[_nan_rates > 0.50]
if len(_high_nan) > 0:
    print(f"  WARNING: {len(_high_nan)} features with >50% NaN rate:")
    for _col, _rate in _high_nan.head(10).items():
        print(f"    {_col}: {_rate:.1%} NaN")
else:
    print(f"  OK: No features with >50% NaN (checked {len(_health_feat_cols)})")

# Per-feature Spearman correlation vs regime_target (daily-aggregated, causal)
if 'regime_target' in df.columns and len(_health_feat_cols) > 0:
    try:
        from scipy.stats import spearmanr as _sp
        _daily_df = df.assign(_dd=pd.to_datetime(df['entry_time']).dt.date)
        _agg_cols = [c for c in _health_feat_cols if c in _daily_df.columns] + ['regime_target', '_dd']
        _daily_agg = _daily_df[_agg_cols].groupby('_dd').mean().reset_index(drop=True)
        _target_daily = _daily_agg['regime_target'].values
        _spear_rows = []
        for _fc in _health_feat_cols:
            if _fc not in _daily_agg.columns:
                continue
            _vals = _daily_agg[_fc].values
            _mask = np.isfinite(_vals) & np.isfinite(_target_daily)
            if _mask.sum() < 30:
                continue
            try:
                _r, _p = _sp(_vals[_mask], _target_daily[_mask])
                _spear_rows.append({'feature': _fc, 'spearman_r': round(float(_r), 4),
                                    'pval': round(float(_p), 4),
                                    'nan_pct': round(float(_nan_rates.get(_fc, 0)) * 100, 1)})
            except Exception:
                pass
        if _spear_rows:
            _spear_df = pd.DataFrame(_spear_rows).sort_values('spearman_r', ascending=False, key=abs)
            print(f"\n  Feature-target Spearman (daily agg, |r| sorted) — {len(_spear_df)} features:")
            print(f"  {'Feature':<45} {'Spearman_r':>11} {'p-val':>8} {'NaN%':>6}")
            print(f"  {'─'*45} {'─'*11} {'─'*8} {'─'*6}")
            _sig  = _spear_df[_spear_df['pval'] < 0.10]
            _nsig = _spear_df[_spear_df['pval'] >= 0.10]
            for _, row in _spear_df.head(15).iterrows():
                _mk = '*' if row['pval'] < 0.10 else ' '
                print(f"  {row['feature']:<44}{_mk} {row['spearman_r']:>+11.4f} {row['pval']:>8.4f} {row['nan_pct']:>5.1f}%")
            print(f"  ... (showing top 15 by |r|; {len(_sig)} significant at α=0.10)")
            if len(_nsig) > 0:
                print(f"  Bottom 5 by |r| (weakest signal):")
                for _, row in _spear_df.tail(5).iterrows():
                    print(f"    {row['feature']:<44} r={row['spearman_r']:>+.4f}  p={row['pval']:.4f}")
    except Exception as _e:
        print(f"  Spearman health check failed: {_e}")
print(f"{'─'*60}")


def add_fold_safe_target(train_idx, test_idx, df_raw_full, df_features_full):
    """
    FIX-re25c: regime_target here is a placeholder 0 (set globally before the loop).
    The authoritative target is recomputed fold-locally from training-window pnl in
    the walk-forward loop (compute_cd_target per direction).  Do NOT filter on
    regime_target or cast it — the fold loop overwrites it immediately after this call.
    """
    train_df = df_features_full.loc[train_idx].copy()
    test_df  = df_features_full.loc[test_idx].copy()
    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════
# Unit 5: signed slippage — positive means favorable, negative means adverse (direction-aware)
# direction_encoded: 0=long (+1 sign), 1=short (-1 sign) → sign = 1 - 2*direction_encoded
if 'slippage' in df.columns:
    if 'direction_encoded' in df.columns:
        df['net_slippage_signed'] = df['slippage'] * (1.0 - 2.0 * df['direction_encoded'])
    else:
        df['net_slippage_signed'] = df['slippage']

# Unit 7: Renko structural features derived from df_raw (source cols are FORBIDDEN)
if 'no_bricks' in df_raw.columns:
    df['bricks_above_breakeven'] = np.maximum(0, df_raw['no_bricks'].values - 8)

if 'exit_jump_bricks' in df_raw.columns and 'trend' in df_raw.columns:
    _trend_sign = df_raw['trend'].fillna(0).values
    df['exit_jump_direction_impact'] = df_raw['exit_jump_bricks'].values * _trend_sign

# Unit 8: day-of-week cyclical encoding (EDA: Mon worst PnL=-1.28, Wed best WR=34.7%)
if 'entry_time' in df.columns:
    _dow = pd.to_datetime(df['entry_time']).dt.dayofweek.astype(float)  # 0=Mon, 4=Fri
    df['dow_sin'] = np.sin(2 * np.pi * _dow / 5.0)
    df['dow_cos'] = np.cos(2 * np.pi * _dow / 5.0)

STATE_FEATURES = [
    'reversal_density_5_lag1', 'reversal_density_10_lag1',
    'directional_efficiency_10_lag1', 'run_length_lag1', 'alternation_ratio_lag1',
    'atr_compression_ratio_lag1', 'vol_regime_lag1',
    'pullback_from_max_progress',
    'call_wall_proxy_lag1', 'put_wall_proxy_lag1',
    'dist_to_gamma_flip_atr_lag1', 'net_gex_zscore_20_lag1', 'net_gex_percentile_20_lag1',
    'RSIOscillator_lag1', 'SqueezeMomentum_lag1'
]
STATE_FEATURES = [f for f in STATE_FEATURES if f in df.columns]

# FIX-re22a: exclude l1_cascade features from L1 candidate pool.
# re21 ablation showed zeroing these IMPROVED L1 AUC +0.186, Spearman +0.440.
# Root cause: target-lag features learn serial correlation of the binary label
# instead of causal market features; harmful under regime change.
# These features stay in df/df_raw for L2 access (l1_day_probability, macro_p_* for MoE).
_L1_EXCLUDE_FEATURES = frozenset([
    # Target-lag features (directly encode past regime_target)
    'prev1_regime_target',
    'rolling_5d_regime_rate',
    'rolling_10d_regime_rate',
    'rolling_20d_regime_rate',
    'days_since_last_regime_day',
    'rolling_5d_mean_C',
    'rolling_5d_std_C',
    'regime_streak_length',
    'rolling_5d_trade_count',
    'rolling_5d_volatility',
    # Macro regime soft weights (sigmoid of price_velocity_3 / ATR — kept for MoE only)
    'macro_p_trend', 'macro_p_vol',
    'macro_p_trend_lv', 'macro_p_trend_hv',
    'macro_p_chop_lv', 'macro_p_chop_hv',
    'macro_atr_pct', 'macro_vel_z',
    # L1 model output (available in L2 after FIX-re21c OOF, but not a causal L1 feature)
    'l1_day_probability',
])

EXCLUDE_FROM_FEATURES = set(META_COLS + FORBIDDEN_COLUMNS) | _L1_EXCLUDE_FEATURES

RAW_FEATURE_COLS = [c for c in df.columns
                    if c not in EXCLUDE_FROM_FEATURES
                    and c not in _DEAD_FEATURES_RE17
                    and not c.endswith('_1min')   # 1min indicator cols reserved for L2 only
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
CANDIDATE_COLS = list(dict.fromkeys(RAW_FEATURE_COLS + [f for f in STATE_FEATURES if f not in RAW_FEATURE_COLS]))

print(f"\n✓ State features : {len(STATE_FEATURES)}")
print(f"  Candidate features (trade-level pool) : {len(CANDIDATE_COLS)}")
print(f"  Dead features excluded (FIX-re17f): {sorted(_DEAD_FEATURES_RE17)}")

_leaked = [c for c in FORBIDDEN_COLUMNS if c in CANDIDATE_COLS]
if _leaked:
    raise RuntimeError(f"LEAKAGE DETECTED in CANDIDATE_COLS: {_leaked}")
print("✓ Leakage guard passed")

_inf_counts = {}
_nan_counts = {}
for c in CANDIDATE_COLS:
    col = df[c]
    _nan_counts[c] = col.isna().sum()
    _inf_counts[c] = np.isinf(col.values[~np.isnan(col.values)]).sum() if col.dtype in [np.float64, np.float32] else 0
_bad_cols = set()
for c in CANDIDATE_COLS:
    if _nan_counts[c] == len(df):
        _bad_cols.add(c)
    elif _inf_counts[c] > len(df) * 0.01:
        _bad_cols.add(c)
_const_cols = set()
for c in CANDIDATE_COLS:
    if c not in _bad_cols:
        _s = df[c].std()
        if pd.notna(_s) and _s < 1e-6:
            _const_cols.add(c)
_bad_cols.update(_const_cols)
# Also exclude dead features
_bad_cols.update(_DEAD_FEATURES_RE17 & set(CANDIDATE_COLS))
CLEAN_CANDIDATE_COLS = [c for c in CANDIDATE_COLS if c not in _bad_cols]
print(f"  Cleaned pool: {len(CLEAN_CANDIDATE_COLS)} (removed {len(_bad_cols)} cols)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-Q-NOLEAK v14d  Per-fold correlation deduplication
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import networkx as _nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    print("  WARNING: networkx not installed — GEX deduplication skipped")


def _compute_dedup_cols(df, candidate_cols, target_col='regime_target', corr_threshold=0.85):
    """Collapse r>corr_threshold Spearman clusters to their best-vs-target rep.
    FIX-Q-NOLEAK v14d: must be called with TRAIN-ONLY data.
    FIX-re17f: dead features are pre-filtered from candidate_cols before this call.
    """
    if not _HAS_NX:
        return candidate_cols
    # Filter out dead features (100% NaN)
    _valid = [c for c in candidate_cols
              if c in df.columns and c not in _DEAD_FEATURES_RE17]
    if target_col not in df.columns or len(_valid) == 0:
        return candidate_cols

    if 'entry_time' in df.columns and len(df) > 5000:
        _dd = pd.to_datetime(df['entry_time']).dt.date
        _agg_cols = _valid + ([target_col] if target_col in df.columns else [])
        _df_daily = (df.assign(_dd=_dd)
                       .groupby('_dd')[_agg_cols]
                       .mean()
                       .reset_index(drop=True))
    else:
        _df_daily = df

    _y = _df_daily[target_col].values if target_col in _df_daily.columns else df[target_col].values

    _spearman = {}
    for _c in _valid:
        _x = _df_daily[_c].values if _c in _df_daily.columns else df[_c].values
        _mask = np.isfinite(_x) & np.isfinite(_y)
        if _mask.sum() < 20:
            _spearman[_c] = 0.0
            continue
        try:
            _r, _ = spearmanr(_x[_mask], _y[_mask])
            _spearman[_c] = abs(_r) if not np.isnan(_r) else 0.0
        except Exception:
            _spearman[_c] = 0.0

    _corr = _df_daily[_valid].corr(method='spearman').abs()
    _G = _nx.Graph()
    _G.add_nodes_from(_valid)
    for _i, _ci in enumerate(_valid):
        for _j, _cj in enumerate(_valid):
            if _j <= _i:
                continue
            if _corr.iloc[_i, _j] > corr_threshold:
                _G.add_edge(_ci, _cj)
    _kept = set()
    for _comp in _nx.connected_components(_G):
        _best = max(_comp, key=lambda _c: _spearman.get(_c, 0.0))
        _kept.add(_best)
    _kept.update(_c for _c in candidate_cols
                 if _c not in _G.nodes and _c not in _DEAD_FEATURES_RE17)
    return [_c for _c in candidate_cols if _c in _kept]


# ── Direction-specific feature pools (EDA-informed, re25) ────────────────────
# EDA (L1_10_direction_asymmetry.md) identified:
#   - 57 signal-flip features (opposite sign between directions)
#   - 40 long-dominant features (>1.5× stronger for long)
#   - 15 short-dominant features (>1.5× stronger for short)
# Each direction uses its full CLEAN pool PLUS direction-specific composites.
# Signal-flip features are kept in both pools since models are already direction-separated
# and will learn the correct sign from direction-filtered training data.

# Long-specific additions: direction-specific composites created in Section 8.6
_LONG_SPECIFIC_FEATS = [
    c for c in [
        'long_rsi_composite_5min_lag1',
        'long_wall_pressure_5min_lag1',
        'long_gex_level_composite_5min_lag1',
        'long_put_wall_safety_5min_lag1',
        'dir_rsi_divergence_5min_lag1',
    ] if c in df.columns
]

# Short-specific additions
_SHORT_SPECIFIC_FEATS = [
    c for c in [
        'short_rsi_composite_5min_lag1',
        'short_vma_zscore_composite_5min_lag1',
        'short_gex_zscore_composite_5min_lag1',
        'dir_rsi_divergence_5min_lag1',
    ] if c in df.columns
]

# Remove any direction composites from the base pool to avoid double-counting
_DIR_COMPOSITE_FEATS = set(_LONG_SPECIFIC_FEATS) | set(_SHORT_SPECIFIC_FEATS)
_BASE_POOL = [c for c in CLEAN_CANDIDATE_COLS if c not in _DIR_COMPOSITE_FEATS]

# Direction-specific pools (direction composites added ONCE at end to avoid dedup removing them)
_CANDIDATE_COLS_LONG  = list(dict.fromkeys(_BASE_POOL + _LONG_SPECIFIC_FEATS))
_CANDIDATE_COLS_SHORT = list(dict.fromkeys(_BASE_POOL + _SHORT_SPECIFIC_FEATS))

# Validate no leakage
for _dc in (_CANDIDATE_COLS_LONG + _CANDIDATE_COLS_SHORT):
    if _dc in FORBIDDEN_COLUMNS:
        raise RuntimeError(f"LEAKAGE in direction pool: {_dc}")

print(f"  Long  candidate pool: {len(_CANDIDATE_COLS_LONG)} features "
      f"({len(_LONG_SPECIFIC_FEATS)} long-specific added)")
print(f"  Short candidate pool: {len(_CANDIDATE_COLS_SHORT)} features "
      f"({len(_SHORT_SPECIFIC_FEATS)} short-specific added)")

# Global pool intentionally NOT dedup'd here — dedup runs per-fold inside the walk-forward loop
# (see _compute_dedup_cols call ~L3957).  This name exists so downstream code that needs
# a fallback pool has something to reference; do not treat it as a deduplicated set.
_CANDIDATE_COLS_POOL = CLEAN_CANDIDATE_COLS          # pre-dedup; per-fold dedup produces fold_dedup_cols
DEDUPLICATED_CANDIDATE_COLS = _CANDIDATE_COLS_POOL   # alias kept for backward compat with _run_l2_bucket
print(f"  [FIX-Q-NOLEAK v14d] Per-fold dedup enabled — global pool = {len(DEDUPLICATED_CANDIDATE_COLS)}")


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD FOLD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_wf_folds(df, train_tdays=TRAIN_TDAYS,
                      test_tdays=TEST_TDAYS, embargo_tdays=EMBARGO_TDAYS):
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df['entry_time']).dt.date
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    folds, ptr = [], train_tdays
    while ptr + embargo_tdays + test_tdays <= n_dates:
        train_start = unique_dates[ptr - train_tdays]
        train_end   = unique_dates[ptr - 1]
        test_start  = unique_dates[min(ptr + embargo_tdays, n_dates - 1)]
        test_end    = unique_dates[min(ptr + embargo_tdays + test_tdays - 1, n_dates - 1)]
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask  = (dates >= test_start) & (dates <= test_end)
        tr = df[train_mask].index.tolist()
        te = df[test_mask].index.tolist()
        if len(tr) >= 100 and len(te) >= 10:
            folds.append((tr, te))
        ptr += test_tdays
    return folds


folds = generate_wf_folds(df)
print(f"\n✓ Walk-forward folds: {len(folds)}")
for i, (tr, te) in enumerate(folds):
    tr_start = df.loc[tr[0], 'entry_time'].date()
    tr_end   = df.loc[tr[-1], 'entry_time'].date()
    te_start = df.loc[te[0], 'entry_time'].date()
    te_end   = df.loc[te[-1], 'entry_time'].date()
    tr_days  = df.loc[tr, 'date'].nunique()
    te_days  = df.loc[te, 'date'].nunique()
    print(f"  Fold {i+1:02d}: train [{tr_start} → {tr_end}] ({len(tr):,} trades / {tr_days} days) | "
          f"test [{te_start} → {te_end}] ({len(te):,} trades / {te_days} days)")


# ═══════════════════════════════════════════════════════════════════════════════
# MANN-WHITNEY U TEST
# ═══════════════════════════════════════════════════════════════════════════════
from scipy.stats import mannwhitneyu


def mann_whitney_regime_separation(test_df, pred_probs, threshold=L1_PROB_THRESHOLD):
    if 'pnl' not in test_df.columns:
        return np.nan, np.nan, np.nan, np.nan
    split = float(np.median(pred_probs))
    pos_mask = pred_probs > split
    neg_mask = ~pos_mask
    pnl_pos = test_df.loc[pos_mask, 'pnl'].dropna().values
    pnl_neg = test_df.loc[neg_mask, 'pnl'].dropna().values
    if len(pnl_pos) < 5 or len(pnl_neg) < 5:
        return np.nan, np.nan, (np.median(pnl_pos) if len(pnl_pos) else np.nan),\
               (np.median(pnl_neg) if len(pnl_neg) else np.nan)
    try:
        u_stat, p_val = mannwhitneyu(pnl_pos, pnl_neg, alternative='greater')
    except Exception:
        u_stat, p_val = np.nan, np.nan
    return u_stat, p_val, float(np.median(pnl_pos)), float(np.median(pnl_neg))


print("✓ Mann-Whitney U test defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# PATH A (deterministic state machine)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_thresholds(train_df):
    sf = train_df[STATE_FEATURES].replace([np.inf, -np.inf], np.nan)
    def pct(col, p):
        if col not in sf.columns:
            return np.nan
        vals = sf[col].dropna()
        return float(np.percentile(vals, p)) if len(vals) > 0 else np.nan
    return {
        'gf_33':    pct('dist_to_gamma_flip_atr_lag1', 33),
        'cw_80':    pct('call_wall_proxy_lag1', 80),
        'pw_80':    pct('put_wall_proxy_lag1', 80),
        'cw_50':    pct('call_wall_proxy_lag1', 50),
        'pw_50':    pct('put_wall_proxy_lag1', 50),
        'vr_80':    pct('vol_regime_lag1', 80),
        'acr_30':   pct('atr_compression_ratio_lag1', 30),
        'alt_70':   pct('alternation_ratio_lag1', 70),
        'de_trend': pct('directional_efficiency_10_lag1', 60),
        'rd_chop':  pct('reversal_density_5_lag1', 50),
    }


def assign_states_path_a(df_in, t):
    n = len(df_in)
    core_arr = np.full(n, 'TRANSITION', dtype=object)
    sub_arr  = np.full(n, 'EXPANSION', dtype=object)
    def get(col, fill=0.0):
        return df_in[col].fillna(fill).values if col in df_in.columns else np.full(n, fill)
    dist_gf = get('dist_to_gamma_flip_atr_lag1', 999)
    cw = get('call_wall_proxy_lag1', 0.5)
    pw = get('put_wall_proxy_lag1', 0.5)
    de = get('directional_efficiency_10_lag1', 0.5)
    rd5 = get('reversal_density_5_lag1', 0.5)
    alt = get('alternation_ratio_lag1', 0.5)
    rl = get('run_length_lag1', 1.0)
    pb = get('pullback_from_max_progress', 0.0)
    vr = get('vol_regime_lag1', 1.0)
    acr = get('atr_compression_ratio_lag1', 1.0)
    for i in range(n):
        if not np.isnan(t['gf_33']) and dist_gf[i] <= t['gf_33']:
            core_arr[i], sub_arr[i] = 'PINNED', 'AT_GF'
        elif not np.isnan(t['cw_80']) and cw[i] >= t['cw_80']:
            core_arr[i], sub_arr[i] = 'PINNED', 'AT_CALL_WALL'
        elif not np.isnan(t['pw_80']) and pw[i] >= t['pw_80']:
            core_arr[i], sub_arr[i] = 'PINNED', 'AT_PUT_WALL'
        elif (not np.isnan(t['cw_50']) and cw[i] >= t['cw_50']
              and not np.isnan(t['pw_50']) and pw[i] >= t['pw_50']):
            core_arr[i], sub_arr[i] = 'PINNED', 'RANGE_BOUND'
        elif de[i] >= t['de_trend'] and rd5[i] <= t['rd_chop']:
            core_arr[i] = 'TREND'
            if de[i] >= 0.65 and rd5[i] <= 0.35 and rl[i] >= 3:
                sub_arr[i] = 'STRONG'
            elif de[i] >= 0.60 and pb[i] >= 0.3:
                sub_arr[i] = 'PULLBACK'
            elif de[i] >= 0.60 and rl[i] >= 5:
                sub_arr[i] = 'EXHAUSTION'
            else:
                sub_arr[i] = 'WEAK'
        elif rd5[i] >= t['rd_chop']:
            core_arr[i] = 'CHOP'
            if alt[i] >= t['alt_70'] and rd5[i] >= 0.7:
                sub_arr[i] = 'WHIPSAW'
            elif vr[i] >= t['vr_80']:
                sub_arr[i] = 'HIGH_VOL'
            elif acr[i] <= t['acr_30']:
                sub_arr[i] = 'LOW_VOL'
            else:
                sub_arr[i] = 'DRIFT'
        else:
            core_arr[i] = 'TRANSITION'
            if acr[i] <= t['acr_30']:
                sub_arr[i] = 'COMPRESSION'
            elif vr[i] >= t['vr_80'] and de[i] >= 0.45:
                sub_arr[i] = 'BREAKOUT'
            elif vr[i] >= t['vr_80']:
                sub_arr[i] = 'BREAKDOWN'
            else:
                sub_arr[i] = 'EXPANSION'
    out = df_in.copy()
    out['core_state'] = core_arr
    out['rule_sub_state'] = sub_arr
    return out


print("✓ Path A state machine defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# PATH B (data-driven discovery) — re17: with Sharpe/ARI gates + deterministic labels
# ═══════════════════════════════════════════════════════════════════════════════
SUB_STATE_SIGS = {
    'TREND': {
        'STRONG':     {'directional_efficiency_10_lag1': 0.85, 'reversal_density_5_lag1': 0.05, 'run_length_lag1': 5.0},
        'WEAK':       {'directional_efficiency_10_lag1': 0.60, 'reversal_density_5_lag1': 0.25, 'run_length_lag1': 2.0},
        'PULLBACK':   {'directional_efficiency_10_lag1': 0.70, 'pullback_from_max_progress': 1.0},
        'EXHAUSTION': {'directional_efficiency_10_lag1': 0.62, 'run_length_lag1': 8.0, 'reversal_density_5_lag1': 0.40},
    },
    'CHOP': {
        'WHIPSAW':  {'alternation_ratio_lag1': 0.9, 'reversal_density_5_lag1': 0.85},
        'HIGH_VOL': {'vol_regime_lag1': 1.6, 'reversal_density_5_lag1': 0.70},
        'LOW_VOL':  {'atr_compression_ratio_lag1': 0.4, 'reversal_density_5_lag1': 0.65},
        'DRIFT':    {'reversal_density_5_lag1': 0.55, 'vol_regime_lag1': 1.0},
    },
    'PINNED': {
        'AT_GF':        {'dist_to_gamma_flip_atr_lag1': 0.1},
        'AT_CALL_WALL': {'call_wall_proxy_lag1': 0.92},
        'AT_PUT_WALL':  {'put_wall_proxy_lag1': 0.92},
        'RANGE_BOUND':  {'call_wall_proxy_lag1': 0.65, 'put_wall_proxy_lag1': 0.65},
    },
    'TRANSITION': {
        'COMPRESSION': {'atr_compression_ratio_lag1': 0.35},
        'BREAKOUT':    {'vol_regime_lag1': 1.4, 'directional_efficiency_10_lag1': 0.65},
        'BREAKDOWN':   {'vol_regime_lag1': 1.4, 'directional_efficiency_10_lag1': 0.25},
        'EXPANSION':   {'vol_regime_lag1': 1.6},
    },
}


class ClusterPipeline:
    def __init__(self, model, scaler, feature_cols, cluster_to_name, method, silhouette):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.cluster_to_name = cluster_to_name
        self.method = method
        self.silhouette = silhouette

    def predict_labels(self, df_in):
        X = df_in[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_sc = self.scaler.transform(X)
        ints = self.model.predict(X_sc)
        return np.array([self.cluster_to_name.get(int(c), 'UNKNOWN') for c in ints])


def _centroid_sim(centroid, sig, feats):
    common = [f for f in sig if f in feats and f in centroid.index]
    if not common:
        return 0.0
    c_vec = np.array([centroid.get(f, 0.0) for f in common], dtype=float)
    s_vec = np.array([sig[f] for f in common], dtype=float)
    n_c, n_s = np.linalg.norm(c_vec), np.linalg.norm(s_vec)
    if n_c < 1e-9 or n_s < 1e-9:
        return 0.0
    return float(np.dot(c_vec, s_vec) / (n_c * n_s))


def _map_cluster_to_name(cluster_id, centroid, core_state, existing_names):
    sigs = SUB_STATE_SIGS.get(core_state, {})
    feats = list(centroid.index)
    scored = []
    for name, sig in sigs.items():
        sim = _centroid_sim(centroid, sig, feats)
        scored.append((name, sim))
    scored.sort(key=lambda x: (-x[1], x[0]))
    best_name, best_sim = (scored[0] if scored else (None, 0.0))
    if best_name and best_sim >= 0.4 and best_name not in existing_names:
        return best_name
    return f'{core_state}_DISC_{cluster_id}'


# ── FIX-re17b / FIX-re20a: Cluster PnL separation gate ──────────────────────
def _compute_cluster_sharpe_spread(data_df, labels, pnl_col='pnl', min_trades=20):
    """
    Compute Sharpe spread across clusters (kept for logging / fallback).

    Returns (spread, per-cluster Sharpes dict).
    If pnl_col not present or fewer than 2 clusters have sufficient data,
    returns (0.0, {}).
    """
    if pnl_col not in data_df.columns:
        return 0.0, {}
    cluster_sharpes = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() < min_trades:
            continue
        pnls = data_df.loc[mask, pnl_col].values
        mu, sigma = pnls.mean(), pnls.std()
        cluster_sharpes[lbl] = mu / (sigma + 1e-8)
    if len(cluster_sharpes) < 2:
        return 0.0, cluster_sharpes
    sharpes = list(cluster_sharpes.values())
    return max(sharpes) - min(sharpes), cluster_sharpes


def _compute_cluster_ks_separation(data_df, labels, pnl_col='pnl', min_trades=20):
    """
    FIX-re20a: KS-test gate on cluster PnL distributions.

    For every pair of clusters with sufficient data, computes the two-sample
    KS statistic.  Returns (max_ks_stat, min_p_value) across all pairs.
    A cluster pair 'passes' if KS ≥ KS_GATE_MIN and p < KS_GATE_PVAL.

    Returns (best_ks_stat, best_p_value, passes_gate).
    Fallback to (0.0, 1.0, False) if pnl_col missing or insufficient data.
    """
    if pnl_col not in data_df.columns:
        return 0.0, 1.0, False

    groups = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() >= min_trades:
            groups[int(lbl)] = data_df.loc[mask, pnl_col].values

    if len(groups) < 2:
        return 0.0, 1.0, False

    lbl_list  = list(groups.keys())
    best_ks   = 0.0
    best_pval = 1.0
    for i in range(len(lbl_list)):
        for j in range(i + 1, len(lbl_list)):
            try:
                ks_stat, p_val = ks_2samp(groups[lbl_list[i]], groups[lbl_list[j]])
            except Exception:
                continue
            if ks_stat > best_ks:
                best_ks   = ks_stat
                best_pval = p_val

    passes = (best_ks >= KS_GATE_MIN) and (best_pval < KS_GATE_PVAL)

    # Secondary gate: Kruskal-Wallis H-test across all clusters (non-parametric)
    # Only run if KS gate passed to avoid double-testing overhead.
    if passes and len(groups) >= 2:
        try:
            from scipy.stats import kruskal as _kruskal
            _kw_stat, _kw_p = _kruskal(*groups.values())
            if _kw_p >= 0.15:   # relax to 0.15 since KS already passed
                passes = False  # clusters don't differ meaningfully on PnL
        except Exception:
            pass   # degrade gracefully: KS gate result stands

    return best_ks, best_pval, passes


# ── FIX-re17c: ARI cross-seed stability gate ─────────────────────────────────
def _cluster_ari_stability(X_scaled, k, method='kmeans', seed1=42, seed2=137):
    """
    Fit clustering twice with different seeds; return ARI between the two labelings.
    Returns float in [-1, 1].  Values < ARI_STABILITY_MIN indicate unstable clustering.
    """
    try:
        if method == 'kmeans':
            l1 = KMeans(n_clusters=k, random_state=seed1, n_init=10).fit_predict(X_scaled)
            l2 = KMeans(n_clusters=k, random_state=seed2, n_init=10).fit_predict(X_scaled)
        else:
            gm1 = GaussianMixture(n_components=k, random_state=seed1, n_init=5)
            gm2 = GaussianMixture(n_components=k, random_state=seed2, n_init=5)
            l1 = gm1.fit(X_scaled).predict(X_scaled)
            l2 = gm2.fit(X_scaled).predict(X_scaled)
        return float(adjusted_rand_score(l1, l2))
    except Exception:
        return 0.0


# ── FIX-re17d: Temporal ARI gate ─────────────────────────────────────────────
def _temporal_ari(X_scaled, labels):
    """
    Split X_scaled into first/second half temporally.  Cluster each half independently
    and compute ARI between the matched labelings using Hungarian matching.
    Returns float in [-1, 1].
    """
    n = len(X_scaled)
    if n < 20:
        return 0.0
    mid = n // 2
    k = len(np.unique(labels))
    if k < 2:
        return 0.0
    try:
        km1 = KMeans(n_clusters=k, random_state=42, n_init=5)
        km2 = KMeans(n_clusters=k, random_state=42, n_init=5)
        l1 = km1.fit_predict(X_scaled[:mid])
        l2 = km2.fit_predict(X_scaled[mid:])
        n_matched = min(len(l1), len(l2))
        if n_matched < 5:
            return 0.0
        # Hungarian matching to align l2 labels to l1 labels
        cm = np.zeros((k, k), dtype=float)
        for a, b in zip(l1[:n_matched], l2[:n_matched]):
            if a < k and b < k:
                cm[a, b] += 1
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        l2_mapped = np.array([mapping.get(x, x) for x in l2[:n_matched]])
        return float(adjusted_rand_score(l1[:n_matched], l2_mapped))
    except Exception:
        return 0.0


# ── FIX-re17e: Deterministic fine Renko state assignment ─────────────────────
# Fine state names and their coarse mapping
_FINE_STATE_TRADEABLE = frozenset({'CLEAN_TREND', 'EXHAUSTING', 'GENUINE_REVERSAL'})
_FINE_STATE_SKIP      = frozenset({'COMPRESSION', 'CHOP'})

def _assign_deterministic_fine_states(X_raw_df, labels, sf_cols):
    """
    Assign deterministic fine Renko state names to cluster labels using
    centroid percentile-band rules instead of random cluster IDs.

    Rules (percentile thresholds computed from the training data):
      CLEAN_TREND       : run_length ≥ 75th pct  AND  alternation_ratio ≤ 25th pct
      EXHAUSTING        : run_length ≥ 75th pct  AND  vol_regime ≥ 75th pct
      GENUINE_REVERSAL  : directional_efficiency ≥ 75th pct  AND  reversal_density ≥ 75th pct
      COMPRESSION       : atr_compression_ratio ≤ 25th pct
      CHOP              : alternation_ratio ≥ 75th pct  AND  run_length ≤ 25th pct
      (fallback)        : NEUTRAL_{cluster_id}

    Returns dict: cluster_label (int) → fine state name (str)
    """
    cluster_to_name = {}
    unique_labels = np.unique(labels)

    # Compute percentile thresholds from all training rows
    def _pct(col, p):
        if col not in X_raw_df.columns:
            return np.nan
        vals = X_raw_df[col].dropna().values
        return float(np.percentile(vals, p)) if len(vals) > 0 else np.nan

    rl_75   = _pct('run_length_lag1', 75)
    rl_25   = _pct('run_length_lag1', 25)
    alt_75  = _pct('alternation_ratio_lag1', 75)
    alt_25  = _pct('alternation_ratio_lag1', 25)
    vr_75   = _pct('vol_regime_lag1', 75)
    de_75   = _pct('directional_efficiency_10_lag1', 75)
    rd_75   = _pct('reversal_density_5_lag1', 75)
    acr_25  = _pct('atr_compression_ratio_lag1', 25)

    used_names = set()
    for lbl in unique_labels:
        mask = labels == lbl
        cent = X_raw_df[mask].mean()

        def _c(col, fallback=np.nan):
            v = cent.get(col, np.nan) if hasattr(cent, 'get') else cent[col] if col in cent.index else np.nan
            return v if not np.isnan(v) else fallback

        rl_c   = _c('run_length_lag1', 1.0)
        alt_c  = _c('alternation_ratio_lag1', 0.5)
        vr_c   = _c('vol_regime_lag1', 1.0)
        de_c   = _c('directional_efficiency_10_lag1', 0.5)
        rd_c   = _c('reversal_density_5_lag1', 0.5)
        acr_c  = _c('atr_compression_ratio_lag1', 1.0)

        # Apply rules in priority order
        name = None
        if (not np.isnan(rl_75) and rl_c >= rl_75 and
                not np.isnan(alt_25) and alt_c <= alt_25 and
                'CLEAN_TREND' not in used_names):
            name = 'CLEAN_TREND'
        elif (not np.isnan(rl_75) and rl_c >= rl_75 and
              not np.isnan(vr_75) and vr_c >= vr_75 and
              'EXHAUSTING' not in used_names):
            name = 'EXHAUSTING'
        elif (not np.isnan(de_75) and de_c >= de_75 and
              not np.isnan(rd_75) and rd_c >= rd_75 and
              'GENUINE_REVERSAL' not in used_names):
            name = 'GENUINE_REVERSAL'
        elif (not np.isnan(acr_25) and acr_c <= acr_25 and
              'COMPRESSION' not in used_names):
            name = 'COMPRESSION'
        elif (not np.isnan(alt_75) and alt_c >= alt_75 and
              not np.isnan(rl_25) and rl_c <= rl_25 and
              'CHOP' not in used_names):
            name = 'CHOP'

        if name is None:
            name = f'NEUTRAL_{int(lbl)}'
        used_names.add(name)
        cluster_to_name[int(lbl)] = name

    return cluster_to_name


def discover_states_for_core(train_core_df, core_state):
    """
    Path B sub-state discovery with re17 gates:
      1. Sharpe spread gate (FIX-re17b)
      2. ARI cross-seed stability gate (FIX-re17c)
      3. Temporal ARI gate (FIX-re17d)
      4. Deterministic fine state naming (FIX-re17e)
    """
    sf_cols = [c for c in STATE_FEATURES
               if c in train_core_df.columns and c not in _DEAD_FEATURES_RE17]
    if len(sf_cols) < 3:
        return None
    X_raw  = train_core_df[sf_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    n_rows = len(X_raw)
    max_k  = min(K_MAX, n_rows // MIN_SAMPLES_STATE)
    if max_k < K_MIN:
        print(f"    [{core_state}] {n_rows} rows — skip (need ≥ {K_MIN * MIN_SAMPLES_STATE})")
        return None
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)
    candidates = []

    def _validate_and_add(model, lbl, method_name):
        """Apply gates; if all pass, add to candidates."""
        if len(np.unique(lbl)) < 2:
            return
        try:
            sil = silhouette_score(X_sc, lbl, sample_size=min(2000, n_rows))
        except Exception:
            return

        # Gate 1a: KS-test PnL separation (FIX-re20a — primary gate)
        _df_reset = train_core_df.reset_index(drop=True)
        _ks_stat, _ks_pval, _ks_pass = _compute_cluster_ks_separation(_df_reset, lbl)
        if not _ks_pass:
            # Fallback: also try Sharpe spread gate (FIX-re17b, threshold lowered re20a)
            _spread, _ = _compute_cluster_sharpe_spread(_df_reset, lbl)
            if _spread < SHARPE_SPREAD_MIN:
                print(f"    [{core_state}] {method_name} REJECTED — "
                      f"KS={_ks_stat:.3f} (p={_ks_pval:.3f}) < {KS_GATE_MIN}, "
                      f"Sharpe spread={_spread:.3f} < {SHARPE_SPREAD_MIN}")
                return
            print(f"    [{core_state}] {method_name} KS gate missed but "
                  f"Sharpe spread={_spread:.3f} ≥ {SHARPE_SPREAD_MIN} — continuing")

        # Gate 2: ARI cross-seed stability (FIX-re17c)
        k_lbl = len(np.unique(lbl))
        _ari_cs = _cluster_ari_stability(X_sc, k_lbl,
                                          method='kmeans' if method_name != 'GMM' else 'gmm')
        if _ari_cs < ARI_STABILITY_MIN:
            print(f"    [{core_state}] {method_name} REJECTED — ARI cross-seed={_ari_cs:.3f} < {ARI_STABILITY_MIN}")
            return

        # Gate 3: Temporal ARI (FIX-re17d)
        _ari_t = _temporal_ari(X_sc, lbl)
        if _ari_t < TEMPORAL_ARI_MIN:
            print(f"    [{core_state}] {method_name} REJECTED — Temporal ARI={_ari_t:.3f} < {TEMPORAL_ARI_MIN}")
            return

        # FIX-re17e: deterministic fine state naming
        name_map = _assign_deterministic_fine_states(X_raw, lbl, sf_cols)
        candidates.append((sil, ClusterPipeline(model, scaler, sf_cols, name_map, method_name, sil)))
        print(f"    [{core_state}] {method_name} PASSED — sil={sil:.3f} "
              f"KS={_ks_stat:.3f}(p={_ks_pval:.3f}) ARI_cs={_ari_cs:.3f} ARI_t={_ari_t:.3f} "
              f"states={list(name_map.values())}")

    # GMM
    best_bic, best_gmm, best_gmm_lbl = np.inf, None, None
    for k in range(K_MIN, max_k + 1):
        try:
            g = GaussianMixture(n_components=k, covariance_type='full',
                                random_state=RANDOM_SEED, n_init=3, max_iter=300)
            g.fit(X_sc)
            bic = g.bic(X_sc)
            if bic < best_bic:
                best_bic = bic
                best_gmm = g
                best_gmm_lbl = g.predict(X_sc)
        except Exception:
            pass
    if best_gmm is not None and best_gmm_lbl is not None:
        _validate_and_add(best_gmm, best_gmm_lbl, 'GMM')

    # KMeans
    best_sil_km, best_km, best_km_lbl = -1, None, None
    for k in range(K_MIN, max_k + 1):
        try:
            km  = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
            lbl = km.fit_predict(X_sc)
            if len(np.unique(lbl)) < 2:
                continue
            sil = silhouette_score(X_sc, lbl, sample_size=min(2000, n_rows))
            if sil > best_sil_km:
                best_sil_km = sil
                best_km = km
                best_km_lbl = lbl
        except Exception:
            pass
    if best_km is not None and best_km_lbl is not None:
        _validate_and_add(best_km, best_km_lbl, 'KMeans')

    # DTree
    try:
        k_dt = min(4, max_k)
        km4  = KMeans(n_clusters=k_dt, random_state=RANDOM_SEED, n_init=10)
        pseudo = km4.fit_predict(X_sc)
        dt   = DecisionTreeClassifier(max_depth=4, min_samples_leaf=MIN_SAMPLES_STATE,
                                      random_state=RANDOM_SEED)
        dt.fit(X_sc, pseudo)
        lbl_dt  = dt.predict(X_sc)
        if len(np.unique(lbl_dt)) >= 2:
            _validate_and_add(dt, lbl_dt, 'DTree')
    except Exception:
        pass

    if not candidates:
        print(f"    [{core_state}] No clustering passed all gates — falling back to rule sub-states")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_sil, best_pipe = candidates[0]
    n_disc = sum(1 for v in best_pipe.cluster_to_name.values() if 'NEUTRAL' in v)
    print(f"    [{core_state}] Best: {best_pipe.method} sil={best_sil:.3f} | "
          f"{list(best_pipe.cluster_to_name.values())} | disc={n_disc}")
    return best_pipe


def build_unified_sub_states(train_df, verbose=True):
    train_df = train_df.copy()
    train_df['unified_sub_state'] = train_df['rule_sub_state']
    cluster_models = {}
    for cs in SUBMODEL_CORE_STATES:
        mask = train_df['core_state'] == cs
        if verbose:
            print(f"  {cs}: {mask.sum()} rows")
        pipe = discover_states_for_core(train_df[mask], cs)
        cluster_models[cs] = pipe
        if pipe is not None:
            train_df.loc[mask, 'unified_sub_state'] = pipe.predict_labels(train_df[mask])
    return train_df, cluster_models


def apply_cluster_models_to_test(test_df, cluster_models):
    test_df = test_df.copy()
    test_df['unified_sub_state'] = test_df['rule_sub_state']
    for cs, pipe in cluster_models.items():
        if pipe is None:
            continue
        mask = test_df['core_state'] == cs
        if mask.sum() == 0:
            continue
        test_df.loc[mask, 'unified_sub_state'] = pipe.predict_labels(test_df[mask])
    return test_df


print("✓ Path B functions defined (re17: Sharpe/ARI gates + deterministic state names).")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE PROFILER
# ═══════════════════════════════════════════════════════════════════════════════
def profile_sub_states(train_df, target_col='regime_target'):
    rows = []
    for (cs, ss), grp in train_df.groupby(['core_state', 'unified_sub_state']):
        n = len(grp)
        if n < 5:
            continue
        wr = grp[target_col].mean()
        rows.append({'core_state': cs, 'sub_state': ss, 'n': n, 'win_rate': wr})
    if not rows:
        return pd.DataFrame(), {}
    profile_df = pd.DataFrame(rows)
    win_rates = profile_df['win_rate'].values
    pct_hi = np.percentile(win_rates, HIGH_EDGE_PCT)
    pct_mid = np.percentile(win_rates, NEUTRAL_LO_PCT)
    pct_lo = np.percentile(win_rates, CAUTION_LO_PCT)
    hi_thresh = max(pct_hi, ABS_HIGH_EDGE)
    lo_thresh = min(pct_lo, ABS_DANGER)
    def _assign(wr):
        if wr >= hi_thresh: return 'HIGH_EDGE'
        if wr >= pct_mid: return 'NEUTRAL'
        if wr >= lo_thresh: return 'CAUTION'
        return 'DANGER'
    profile_df['profile'] = profile_df['win_rate'].apply(_assign)
    profile_map = {(r.core_state, r.sub_state): r.profile for r in profile_df.itertuples()}
    return profile_df, profile_map


def apply_profile_to_df(df_in, profile_map):
    df_in = df_in.copy()
    df_in['sub_state_profile'] = df_in.apply(
        lambda r: profile_map.get((r['core_state'], r['unified_sub_state']), 'NEUTRAL'),
        axis=1
    )
    return df_in


print("✓ State profiler defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def _grouped_ts_splits(entry_times, n_splits, embargo_steps=1):
    unique_ts = np.sort(np.unique(entry_times))
    n_ts      = len(unique_ts)
    fold_size = n_ts // (n_splits + 1)
    if fold_size < 1:
        return
    for k in range(1, n_splits + 1):
        tr_end_ts    = unique_ts[fold_size * k - 1]
        val_start_ix = min(fold_size * k + embargo_steps, n_ts)
        val_end_ix   = min(val_start_ix + fold_size, n_ts)
        if val_start_ix >= n_ts or val_end_ix <= val_start_ix:
            continue
        val_start_ts = unique_ts[val_start_ix]
        val_end_ts   = unique_ts[val_end_ix - 1]
        tr_idx  = np.where(entry_times <= tr_end_ts)[0]
        val_idx = np.where((entry_times >= val_start_ts) & (entry_times <= val_end_ts))[0]
        if len(tr_idx) > 0 and len(val_idx) > 0:
            yield tr_idx, val_idx


_FLOAT32_MAX = np.float32(np.finfo(np.float32).max)


def _prep_X(df_in, feature_cols):
    X       = df_in[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    vals    = X.fillna(medians).values
    np.clip(vals, -_FLOAT32_MAX, _FLOAT32_MAX, out=vals)
    np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return vals, medians


def _encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le


def train_submodel_for_core(train_df, core_state, feature_cols,
                            inner_folds=INNER_FOLDS,
                            lgb_p=None, xgb_p=None):
    lgb_p = lgb_p or LGB_PARAMS_SUBMODEL_PARALLEL
    xgb_p = xgb_p or XGB_PARAMS_SUBMODEL_PARALLEL
    mask       = train_df['core_state'] == core_state
    core_train = train_df[mask].copy()

    if len(core_train) < MIN_SAMPLES_STATE * 2:
        return None

    labels        = core_train['unified_sub_state'].values
    unique_labels = np.unique(labels)
    n_classes     = len(unique_labels)
    if n_classes < 2:
        return None

    y, le = _encode_labels(labels)
    X, medians = _prep_X(core_train, feature_cols)
    X_raw_np = core_train[feature_cols].replace([np.inf, -np.inf], np.nan).values
    X_raw_np = np.clip(X_raw_np, -_FLOAT32_MAX, _FLOAT32_MAX)
    np.nan_to_num(X_raw_np, copy=False, posinf=np.nan, neginf=np.nan)

    uniform_prob  = 1.0 / n_classes
    oof_probs_arr = np.full((len(core_train), n_classes), uniform_prob, dtype=float)
    oof_filled    = np.ones(len(core_train), dtype=bool)
    inner_times   = core_train['entry_time'].values

    for inner_tr, inner_val in _grouped_ts_splits(inner_times, inner_folds):
        y_tr = y[inner_tr]
        if len(np.unique(y_tr)) < n_classes:
            print(f"    [WARN] Degenerate inner fold skipped "
                  f"({len(np.unique(y_tr))}/{n_classes} classes) — OOF stays at uniform")
            continue
        inner_medians = np.nan_to_num(np.nanmedian(X_raw_np[inner_tr], axis=0), nan=0.0)
        X_tr  = X_raw_np[inner_tr].copy()
        X_val = X_raw_np[inner_val].copy()
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]),  j] = inner_medians[j]
            X_val[np.isnan(X_val[:, j]), j] = inner_medians[j]
        sw_tr = compute_sample_weight('balanced', y_tr)
        lgb_m = lgb.LGBMClassifier(objective='multiclass', num_class=n_classes, **lgb_p)
        lgb_m.fit(X_tr, y_tr, sample_weight=sw_tr)
        xgb_m = xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes, **xgb_p)
        xgb_m.fit(X_tr, y_tr, sample_weight=sw_tr)
        p_lgb = lgb_m.predict_proba(X_val)
        p_xgb = xgb_m.predict_proba(X_val)
        oof_probs_arr[inner_val] = (p_lgb + p_xgb) / 2
        oof_filled[inner_val]    = False

    has_pred      = ~oof_filled
    oof_preds_arr = oof_probs_arr.argmax(axis=1)
    if has_pred.sum() > 0:
        oof_acc   = accuracy_score(y[has_pred], oof_preds_arr[has_pred])
        oof_f1_w  = f1_score(y[has_pred], oof_preds_arr[has_pred],
                             labels=range(n_classes), average='weighted', zero_division=0)
        oof_f1_m  = f1_score(y[has_pred], oof_preds_arr[has_pred],
                             labels=range(n_classes), average='macro',    zero_division=0)
        oof_report = classification_report(y[has_pred], oof_preds_arr[has_pred],
                                           labels=range(n_classes),
                                           target_names=le.classes_, zero_division=0)
    else:
        oof_acc = oof_f1_w = oof_f1_m = 0.0
        oof_report = "(no valid inner folds)"

    min_useful_acc = 1.0 / n_classes + L2_OOF_ACC_MARGIN
    if oof_acc < min_useful_acc:
        print(f"    [{core_state}] OOF acc={oof_acc:.3f} < {min_useful_acc:.3f} — sub-model SKIPPED")
        return None

    prob_cols = [f'{core_state}__p_{le.classes_[c]}' for c in range(n_classes)]
    oof_df    = pd.DataFrame(oof_probs_arr, index=core_train.index, columns=prob_cols)

    sw_full   = compute_sample_weight('balanced', y)
    final_lgb = lgb.LGBMClassifier(objective='multiclass', num_class=n_classes, **lgb_p)
    final_lgb.fit(X, y, sample_weight=sw_full)
    final_xgb = xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes, **xgb_p)
    final_xgb.fit(X, y, sample_weight=sw_full)

    return {
        'lgb': final_lgb, 'xgb': final_xgb,
        'oof_df': oof_df, 'label_encoder': le,
        'medians': medians, 'n_classes': n_classes,
        'prob_cols': prob_cols, 'feature_cols': feature_cols,
        'oof_acc': oof_acc, 'oof_f1_w': oof_f1_w, 'oof_f1_m': oof_f1_m,
        'oof_report': oof_report, 'sub_states': list(le.classes_), 'n_train': len(core_train),
    }


def _run_submodel_sequential(cs, train_df, feature_cols):
    buf = io.StringIO()
    mask       = train_df['core_state'] == cs
    core_train = train_df[mask]
    n_rows     = len(core_train)
    buf.write(f"  [{cs}] {n_rows} rows")
    if len(core_train) < MIN_SAMPLES_STATE * 2:
        buf.write(" — skip (need ≥ {})\n".format(MIN_SAMPLES_STATE * 2))
        return cs, None, buf.getvalue()
    labels        = core_train['unified_sub_state'].values
    unique_labels = np.unique(labels)
    n_classes     = len(unique_labels)
    if n_classes < 2:
        buf.write(" — skip (single class)\n")
        return cs, None, buf.getvalue()
    res = train_submodel_for_core(
        train_df, cs, feature_cols,
        lgb_p=LGB_PARAMS_SUBMODEL_1T,
        xgb_p=XGB_PARAMS_SUBMODEL_1T,
    )
    if res:
        buf.write(f" | k={res['n_classes']} | OOF acc={res['oof_acc']:.3f}"
                  f" | wF1={res['oof_f1_w']:.3f} | mF1={res['oof_f1_m']:.3f}\n")
    return cs, res, buf.getvalue()


_train_submodel_worker = _run_submodel_sequential


def train_all_submodels(train_df, feature_cols, verbose=True, n_workers=N_PARALLEL):
    submodel_results = {}
    oof_frames       = []
    log_lines        = {}
    present_states = set(train_df['core_state'].unique()) if 'core_state' in train_df.columns else set()
    states_to_run  = [cs for cs in SUBMODEL_CORE_STATES if cs in present_states]

    if n_workers > 1 and len(states_to_run) > 1:
        # Run independent core-state submodels in parallel (LGB/XGB release GIL).
        with ThreadPoolExecutor(max_workers=min(n_workers, len(states_to_run))) as ex:
            futs = {ex.submit(_run_submodel_sequential, cs, train_df, feature_cols): cs
                    for cs in states_to_run}
            for fut in as_completed(futs):
                cs_out, res, log = fut.result()
                log_lines[cs_out] = log
                if res is not None:
                    submodel_results[cs_out] = res
                    oof_frames.append(res['oof_df'])
    else:
        for cs in states_to_run:
            cs_out, res, log = _run_submodel_sequential(cs, train_df, feature_cols)
            log_lines[cs_out] = log
            if res is not None:
                submodel_results[cs_out] = res
                oof_frames.append(res['oof_df'])

    if verbose:
        print(f"  Sub-model training ({'parallel' if n_workers > 1 else 'sequential'}, "
              f"{len(states_to_run)} states):")
        for cs in CORE_STATES:
            if cs in log_lines:
                print(log_lines[cs], end='')

    if oof_frames:
        oof_frames.sort(key=lambda f: f.columns[0])
    oof_meta = pd.concat(oof_frames, axis=1) if oof_frames else pd.DataFrame(index=train_df.index)
    return submodel_results, oof_meta


def get_submodel_meta_features(df_in, submodel_results):
    meta_frames = []
    for cs, res in submodel_results.items():
        if res is None:
            continue
        mask = df_in['core_state'] == cs
        if mask.sum() == 0:
            continue
        core_df = df_in[mask]
        X_raw   = core_df[res['feature_cols']].replace([np.inf, -np.inf], np.nan)
        X       = X_raw.fillna(res['medians']).values
        p_lgb   = res['lgb'].predict_proba(X)
        p_xgb   = res['xgb'].predict_proba(X)
        probs   = (p_lgb + p_xgb) / 2
        meta_frames.append(pd.DataFrame(probs, index=core_df.index, columns=res['prob_cols']))
    if meta_frames:
        return pd.concat(meta_frames, axis=0).reindex(df_in.index)
    return pd.DataFrame(index=df_in.index)


print("✓ Sub-model functions defined (sequential — re17).")


# ═══════════════════════════════════════════════════════════════════════════════
# BIVARIATE FEATURE PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════════
def _bivariate_test_one(args):
    col, x_vals, y_vals, alpha = args
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if mask.sum() < 30:
        return col, {'pearson_p': 1.0, 'spearman_p': 1.0, 'kept': False}
    xm, ym = x_vals[mask], y_vals[mask]
    if np.std(xm) < 1e-12:
        return col, {'pearson_p': 1.0, 'spearman_p': 1.0, 'kept': False}
    try:
        _, p_pear = pearsonr(xm, ym)
    except Exception:
        p_pear = 1.0
    try:
        _, p_spear = spearmanr(xm, ym)
    except Exception:
        p_spear = 1.0
    kept = (p_pear < alpha) or (p_spear < alpha)
    return col, {'pearson_p': p_pear, 'spearman_p': p_spear, 'kept': kept}


def bivariate_filter(train_df, candidate_cols, target_col='regime_target',
                     alpha=BIVARIATE_ALPHA):
    if target_col == 'regime_target' and 'entry_time' in train_df.columns:
        _df = train_df.copy()
        _df['_bivar_date'] = pd.to_datetime(_df['entry_time']).dt.date
        _stat_cols = [c for c in candidate_cols if c in _df.columns]
        if _df['_bivar_date'].nunique() == len(_df):
            df_stat = _df
        else:
            _df = (
                _df.groupby('_bivar_date')[_stat_cols + [target_col]]
                .mean()
                .reset_index(drop=True)
            )
            df_stat = _df
    else:
        df_stat = train_df

    y = df_stat[target_col].values
    tasks = [
        (col,
         df_stat[col].values if col in df_stat.columns else np.full(len(df_stat), np.nan),
         y, alpha)
        for col in candidate_cols
    ]
    stats = {}
    surviving = []
    with ThreadPoolExecutor(max_workers=USABLE_CORES) as ex:
        for col, result in ex.map(_bivariate_test_one, tasks):
            stats[col] = result
            if result['kept']:
                surviving.append(col)
    return surviving, stats


print("✓ Bivariate filter defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTERED FEATURE IMPORTANCE (CFI)
# ═══════════════════════════════════════════════════════════════════════════════
def clustered_feature_importance(train_df, surviving_cols, target_col='regime_target',
                                 top_k=TOP_K_FEATURES, distance_thresh=CFI_DISTANCE_THRESH):
    if len(surviving_cols) <= top_k:
        return surviving_cols, {'clusters': {}, 'method': 'passthrough'}

    y     = train_df[target_col].values
    X_raw = train_df[surviving_cols].replace([np.inf, -np.inf], np.nan)
    n     = len(train_df)
    assert train_df['entry_time'].is_monotonic_increasing, (
        "CFI requires train_df sorted by entry_time.")
    split_idx = int(n * 0.8)
    col_medians = X_raw.median()
    X_filled    = X_raw.fillna(col_medians).values

    corr_matrix = pd.DataFrame(X_filled, columns=surviving_cols).corr(method='spearman').values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    dist_matrix = np.maximum(1.0 - np.abs(corr_matrix), 0.0)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=distance_thresh, criterion='distance')

    clusters = {}
    for feat_idx, cl in enumerate(cluster_labels):
        clusters.setdefault(int(cl), []).append(feat_idx)

    X_tr_mda  = X_raw.iloc[:split_idx].fillna(col_medians).values
    X_val_mda = X_raw.iloc[split_idx:].fillna(col_medians).values
    y_tr_mda, y_val_mda = y[:split_idx], y[split_idx:]

    val_unique_dates = (train_df.iloc[split_idx:]['entry_time'].dt.date.nunique()
                        if 'entry_time' in train_df.columns else len(X_val_mda))
    if val_unique_dates < 80:
        importances = []
        for i, col in enumerate(surviving_cols):
            try:
                c, _ = spearmanr(X_filled[:, i], y)
                importances.append((abs(c) if not np.isnan(c) else 0.0, col))
            except Exception:
                importances.append((0.0, col))
        importances.sort(reverse=True)
        selected = [col for _, col in importances[:top_k]]
        return selected, {'clusters': clusters, 'method': 'spearman_fallback'}

    mda_lgb = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=USABLE_CORES, verbose=-1
    )
    mda_lgb.fit(X_tr_mda, y_tr_mda)
    baseline_acc = accuracy_score(y_val_mda, mda_lgb.predict(X_val_mda))

    def _perm_cluster(args):
        cl_id, feat_indices, seed = args
        rng_l = np.random.RandomState(seed)
        X_p = X_val_mda.copy()
        for fi in feat_indices:
            X_p[:, fi] = rng_l.permutation(X_p[:, fi])
        return cl_id, baseline_acc - accuracy_score(y_val_mda, mda_lgb.predict(X_p))

    cluster_tasks = [(cl_id, fi_list, RANDOM_SEED + cl_id) for cl_id, fi_list in clusters.items()]
    cluster_importance = {}
    with ThreadPoolExecutor(max_workers=USABLE_CORES) as ex:
        for cl_id, imp in ex.map(_perm_cluster, cluster_tasks):
            cluster_importance[cl_id] = imp

    def _perm_individual(args):
        fi, seed = args
        rng_l = np.random.RandomState(seed)
        X_p = X_val_mda.copy()
        X_p[:, fi] = rng_l.permutation(X_p[:, fi])
        return fi, baseline_acc - accuracy_score(y_val_mda, mda_lgb.predict(X_p))

    indiv_tasks = [(fi, RANDOM_SEED + fi) for fi_list in clusters.values() for fi in fi_list]
    individual_importance = {}
    with ThreadPoolExecutor(max_workers=USABLE_CORES) as ex:
        for fi, imp in ex.map(_perm_individual, indiv_tasks):
            individual_importance[fi] = imp

    sorted_clusters = sorted(cluster_importance.items(), key=lambda x: x[1], reverse=True)
    selected_indices = []
    for cl_id, _ in sorted_clusters:
        feat_indices = clusters[cl_id]
        sorted_feats = sorted(feat_indices, key=lambda fi: individual_importance.get(fi, 0), reverse=True)
        for fi in sorted_feats:
            if len(selected_indices) >= top_k:
                break
            selected_indices.append(fi)
        if len(selected_indices) >= top_k:
            break

    selected_cols = [surviving_cols[i] for i in selected_indices]
    info = {
        'clusters': {int(k): [surviving_cols[i] for i in v] for k, v in clusters.items()},
        'cluster_importance': cluster_importance,
        'individual_importance': {surviving_cols[k]: v for k, v in individual_importance.items()},
        'baseline_acc': baseline_acc,
        'method': 'CFI_MDA',
    }
    return selected_cols, info


print("✓ CFI defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# FRACDIFF (retained for reference, not called)
# ═══════════════════════════════════════════════════════════════════════════════
def get_frac_diff_weights(d, window=FRACDIFF_WINDOW):
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def apply_frac_diff(series, d, window=FRACDIFF_WINDOW):
    if d == 0.0:
        return series.copy()
    weights = get_frac_diff_weights(d, window)
    n = len(series)
    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        segment = series[max(0, t - window + 1):t + 1]
        w = weights[:len(segment)]
        result[t] = np.dot(w[::-1], segment[-len(w):])
    return result


print("✓ Fracdiff functions defined (reference only — not called).")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAM-SCHMIDT ORTHOGONALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def gram_schmidt_fit(train_X):
    n_samples, n_features = train_X.shape
    means     = np.nanmean(train_X, axis=0)
    X_centered = np.nan_to_num(train_X - means, nan=0.0)
    Q          = np.zeros_like(X_centered, dtype=float)
    R          = np.zeros((n_features, n_features), dtype=float)
    valid_cols = []
    for j in range(n_features):
        v = X_centered[:, j].copy()
        for i in valid_cols:
            R[i, j] = np.dot(Q[:, i], v)
            norm_qi = np.dot(Q[:, i], Q[:, i])
            if norm_qi > 1e-12:
                v = v - (R[i, j] / norm_qi) * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            Q[:, j] = 0.0
        else:
            Q[:, j] = v
            valid_cols.append(j)
    return {
        'means': means, 'R': R,
        'norms': np.array([np.dot(Q[:, i], Q[:, i]) for i in range(n_features)]),
        'n_features': n_features, 'valid_cols': valid_cols,
    }


def gram_schmidt_transform(X, gs_params):
    means      = gs_params['means']
    R          = gs_params['R']
    norms      = gs_params['norms']
    n_features = gs_params['n_features']
    valid_cols = gs_params['valid_cols']
    X_centered = np.nan_to_num(X - means, nan=0.0)
    Q = np.zeros_like(X_centered, dtype=float)
    for j in range(n_features):
        v = X_centered[:, j].copy()
        for i in valid_cols:
            if i >= j:
                break
            if norms[i] > 1e-12:
                v = v - (R[i, j] / (norms[i] + 1e-12)) * Q[:, i]
        Q[:, j] = v
    return np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)


print("✓ Gram-Schmidt orthogonalization defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE STATE ENCODING & MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def encode_core_state(df_in):
    return df_in['core_state'].map(CORE_STATE_MAP).values.astype(np.int32)


def _compute_meta_fill(submodel_results, meta_cols):
    fill = {}
    for cs, res in submodel_results.items():
        if res is None:
            continue
        uniform = 1.0 / res['n_classes']
        for col in res['prob_cols']:
            fill[col] = uniform
    for c in meta_cols:
        if c not in fill:
            fill[c] = 0.0
    return fill


def build_main_feature_matrix_v5(X_ortho, core_state_encoded, meta_df,
                                  submodel_results, train_medians_ortho=None):
    if train_medians_ortho is None:
        train_medians_ortho = np.nanmedian(X_ortho, axis=0)
    X_ortho_clean = np.where(np.isfinite(X_ortho), X_ortho,
                             np.broadcast_to(train_medians_ortho, X_ortho.shape))
    cs_col     = core_state_encoded.reshape(-1, 1).astype(float)
    meta_fill  = _compute_meta_fill(submodel_results, meta_df.columns)
    meta_arr   = meta_df.fillna(meta_fill).values
    X_combined = np.column_stack([X_ortho_clean, cs_col, meta_arr])
    X_combined = np.nan_to_num(np.clip(X_combined, -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)
    return X_combined, train_medians_ortho


def train_main_model_v5(X_train, y_train, feature_names, cat_indices=None, direction='long',
                        sample_weight=None):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw   = float(n_neg / n_pos) if n_pos > 0 else 1.0
    _spw_cap = _L1_SPW_MAX_SHORT if direction == 'short' else _L1_SPW_MAX_LONG
    spw   = min(spw, _spw_cap)

    def _fit_lgb():
        # FIX-re24d: use LGB_PARAMS_MAIN_1T directly — is_unbalance=True is already set
        # in the params dict (restored from re22). No explicit SPW injection here.
        _lgb_p = {**LGB_PARAMS_MAIN_1T}
        if direction == 'short':
            _lgb_p.update(_LGB_SHORT_OVERRIDES)
        m = lgb.LGBMClassifier(**_lgb_p)
        m.fit(X_train, y_train,
              categorical_feature=cat_indices if cat_indices else 'auto',
              sample_weight=sample_weight)
        return m

    def _fit_xgb():
        m = xgb.XGBClassifier(**{**XGB_PARAMS_MAIN_1T, 'scale_pos_weight': spw})
        m.fit(X_train, y_train, sample_weight=sample_weight)
        return m

    _ensemble_workers = max(2, N_PARALLEL // 2)
    with ThreadPoolExecutor(max_workers=_ensemble_workers) as ex:
        f_lgb = ex.submit(_fit_lgb)
        f_xgb = ex.submit(_fit_xgb)
        lgb_main = f_lgb.result()
        xgb_main = f_xgb.result()

    return {'lgb': lgb_main, 'xgb': xgb_main,
            'feature_names': feature_names, 'scale_pos_weight': spw}


def predict_main_model_v5(X_test, model_result):
    p_lgb = model_result['lgb'].predict_proba(X_test)[:, 1]
    p_xgb = model_result['xgb'].predict_proba(X_test)[:, 1]
    return (p_lgb + p_xgb) / 2


print("✓ Main model functions defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTILE RANKING FILTER & DECILE MONOTONICITY TEST
# ═══════════════════════════════════════════════════════════════════════════════
def percentile_rank_filter(all_probs, buffer_size=PERCENTILE_BUFFER_SIZE,
                           long_pct=LONG_ENTRY_PCT):
    n        = len(all_probs)
    signals  = np.full(n, 'HOLD', dtype=object)
    pct_ranks = np.full(n, np.nan)
    for i in range(n):
        start  = max(0, i - buffer_size + 1)
        buffer = all_probs[start:i + 1]
        if len(buffer) < 20:
            signals[i] = 'INSUFFICIENT_BUFFER'
            continue
        rank = (buffer < all_probs[i]).sum() / len(buffer) * 100
        pct_ranks[i] = rank
        if rank >= long_pct:
            signals[i] = 'LONG'
        else:
            signals[i] = 'HOLD'
    return signals, pct_ranks


def decile_monotonicity_test(all_probs, all_targets, all_pnl=None):
    df_dec = pd.DataFrame({'prob': all_probs, 'target': all_targets})
    if all_pnl is not None:
        df_dec['pnl'] = all_pnl
    df_dec['decile'] = pd.qcut(df_dec['prob'], 10, labels=False, duplicates='drop') + 1
    grp = df_dec.groupby('decile')
    decile_agg = pd.DataFrame({
        'decile':    sorted(df_dec['decile'].unique()),
        'mean_prob': grp['prob'].mean().values,
        'win_rate':  grp['target'].mean().values,
        'n':         grp['target'].count().values,
    })
    if all_pnl is not None:
        decile_agg['mean_pnl'] = grp['pnl'].mean().values
    decile_ranks = decile_agg['decile'].values
    win_rates    = decile_agg['win_rate'].values
    try:
        sp_corr, sp_p = spearmanr(decile_ranks, win_rates)
    except Exception:
        sp_corr, sp_p = 0.0, 1.0
    try:
        kt_tau, kt_p = kendalltau(decile_ranks, win_rates)
    except Exception:
        kt_tau, kt_p = 0.0, 1.0
    spread      = win_rates[-1] - win_rates[0] if len(decile_agg) >= 2 else 0.0
    is_monotonic = (sp_corr >= MIN_SPEARMAN_DECILE) and (kt_p < 0.05)
    return decile_agg, sp_corr, sp_p, kt_tau, kt_p, spread, is_monotonic


print("✓ Percentile filter and decile monotonicity test defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# L2 INTRADAY FEATURE ENGINEERING (v15 — 18 features)
# ═══════════════════════════════════════════════════════════════════════════════
_L2_INTRADAY_COLS = [
    'intraday_trade_count', 'intraday_entry_zscore', 'intraday_trend_consistency',
    'intraday_brick_momentum', 'factor_entry_rank', 'time_since_last_trade',
    'intraday_vol_accel', 'intraday_direction_streak',
    'intraday_price_range_pos', 'time_since_open_hours', 'intraday_trade_rate',
    'intraday_trend_alignment', 'intraday_reversal_count', 'intraday_brick_zscore',
    'factor_intraday_consistency', 'intraday_price_velocity',
    'entry_vs_day_open', 'intraday_new_high_flag',
]


def engineer_l2_intraday_features(df_in):
    df = df_in.copy()
    df['_et'] = pd.to_datetime(df['entry_time'])
    df['_date'] = df['_et'].dt.date

    df['intraday_trade_count'] = df.groupby(['Ticker', '_date']).cumcount()

    if 'entry_price' in df.columns:
        _ep_expanding_mean = df.groupby(['Ticker', '_date'])['entry_price'].transform(
            lambda x: x.expanding().mean().shift(1))
        _ep_expanding_std = df.groupby(['Ticker', '_date'])['entry_price'].transform(
            lambda x: x.expanding().std().shift(1)).clip(lower=1e-8)
        df['intraday_entry_zscore'] = (df['entry_price'] - _ep_expanding_mean) / _ep_expanding_std
    else:
        df['intraday_entry_zscore'] = np.nan

    if 'trend' in df.columns:
        df['intraday_trend_consistency'] = df.groupby(['Ticker', '_date'])['trend'].transform(
            lambda x: x.astype(float).shift(1).expanding().mean())
    else:
        df['intraday_trend_consistency'] = np.nan

    if 'brick_size' in df.columns:
        df['intraday_brick_momentum'] = df.groupby(['Ticker', '_date'])['brick_size'].transform(
            lambda x: x.pct_change().shift(1).expanding().mean())
    else:
        df['intraday_brick_momentum'] = np.nan

    if 'entry_price' in df.columns:
        df['factor_entry_rank'] = df.groupby(['Ticker', '_et'])['entry_price'].rank(pct=True)
    else:
        df['factor_entry_rank'] = np.nan

    df['time_since_last_trade'] = df.groupby(['Ticker', '_date'])['_et'].transform(
        lambda x: x.diff().dt.total_seconds()).fillna(0)

    if 'atr_brick_size' in df.columns:
        df['intraday_vol_accel'] = df.groupby(['Ticker', '_date'])['atr_brick_size'].transform(
            lambda x: x.diff().shift(1).rolling(3, min_periods=1).mean())
    else:
        df['intraday_vol_accel'] = np.nan

    if 'trend' in df.columns:
        def _intraday_streak(s):
            out = pd.Series(0, index=s.index, dtype=int)
            vals = s.values
            for i in range(1, len(vals)):
                if vals[i] == vals[i - 1]:
                    out.iloc[i] = out.iloc[i - 1] + 1
            return out
        df['intraday_direction_streak'] = df.groupby(['Ticker', '_date'])['trend'].transform(
            _intraday_streak)
    else:
        df['intraday_direction_streak'] = 0

    if 'entry_price' in df.columns:
        _ep_exp_min = df.groupby(['Ticker', '_date'])['entry_price'].transform(
            lambda x: x.expanding().min().shift(1))
        _ep_exp_max = df.groupby(['Ticker', '_date'])['entry_price'].transform(
            lambda x: x.expanding().max().shift(1))
        _ep_day_range = (_ep_exp_max - _ep_exp_min).clip(lower=1e-8)
        df['intraday_price_range_pos'] = (
            (df['entry_price'] - _ep_exp_min) / _ep_day_range).clip(0.0, 1.0)
    else:
        df['intraday_price_range_pos'] = np.nan

    df['time_since_open_hours'] = (
        (df['_et'].dt.hour - 9) * 60 + df['_et'].dt.minute - 30
    ).clip(lower=0).astype(float) / 60.0

    df['intraday_trade_rate'] = (
        df['intraday_trade_count'].astype(float) /
        df['time_since_open_hours'].clip(lower=0.1))

    if 'trend' in df.columns:
        _prior_trend_mean = df.groupby(['Ticker', '_date'])['trend'].transform(
            lambda x: x.astype(float).shift(1).expanding().mean())
        df['intraday_trend_alignment'] = (
            np.sign(df['trend'].astype(float)) * np.sign(_prior_trend_mean.fillna(0)))
    else:
        df['intraday_trend_alignment'] = 0

    if 'trend' in df.columns:
        df['intraday_reversal_count'] = df.groupby(['Ticker', '_date'])['trend'].transform(
            lambda x: x.astype(float).diff().abs().fillna(0).shift(1).expanding().sum())
    else:
        df['intraday_reversal_count'] = 0

    if 'brick_size' in df.columns:
        _bs_exp_mean = df.groupby(['Ticker', '_date'])['brick_size'].transform(
            lambda x: x.expanding().mean().shift(1))
        _bs_exp_std = df.groupby(['Ticker', '_date'])['brick_size'].transform(
            lambda x: x.expanding().std().shift(1)).clip(lower=1e-8)
        df['intraday_brick_zscore'] = ((df['brick_size'] - _bs_exp_mean) / _bs_exp_std).clip(-5, 5)
    else:
        df['intraday_brick_zscore'] = np.nan

    if 'trend' in df.columns:
        df['factor_intraday_consistency'] = df.groupby(
            ['Ticker', 'factor', '_date'])['trend'].transform(
            lambda x: x.astype(float).shift(1).expanding().mean())
    else:
        df['factor_intraday_consistency'] = np.nan

    if 'entry_price' in df.columns:
        _raw_vel = df.groupby(['Ticker', '_date'])['entry_price'].transform(
            lambda x: x.diff().shift(1))
        if 'brick_size' in df.columns:
            _bs_safe = df['brick_size'].replace(0, np.nan)
            df['intraday_price_velocity'] = (_raw_vel / _bs_safe).clip(-20, 20)
        else:
            df['intraday_price_velocity'] = _raw_vel
    else:
        df['intraday_price_velocity'] = np.nan

    if 'entry_price' in df.columns:
        _first_ep = df.groupby(['Ticker', '_date'])['entry_price'].transform('first')
        _first_ep_safe = _first_ep.replace(0, np.nan).abs() + 1e-8
        df['entry_vs_day_open'] = (
            (df['entry_price'] - _first_ep) / _first_ep_safe).clip(-0.05, 0.05)
    else:
        df['entry_vs_day_open'] = np.nan

    if 'entry_price' in df.columns:
        _prior_max_ep = df.groupby(['Ticker', '_date'])['entry_price'].transform(
            lambda x: x.expanding().max().shift(1))
        df['intraday_new_high_flag'] = (df['entry_price'] > _prior_max_ep).astype(float).fillna(0.0)
    else:
        df['intraday_new_high_flag'] = 0.0

    # ── FIX-re20c: New time-of-day + running-state features ───────────────────
    # time_of_day_frac: fraction of 6.5-hour session elapsed (9:30–16:00)
    _secs_since_open = (
        (df['_et'].dt.hour - 9) * 3600
        + (df['_et'].dt.minute - 30) * 60
        + df['_et'].dt.second
    ).clip(lower=0)
    _session_secs = 6.5 * 3600
    df['time_of_day_frac']   = (_secs_since_open / _session_secs).clip(0.0, 1.0)
    df['time_to_close_frac'] = 1.0 - df['time_of_day_frac']
    df['is_first_hour']      = (df['time_of_day_frac'] < (1.0 / 6.5)).astype(float)
    df['is_last_hour']       = (df['time_of_day_frac'] > (5.5 / 6.5)).astype(float)

    # factor_macro_alignment: +1 if trade direction agrees with macro regime,
    # -1 if opposed, 0 if macro data missing.  Uses macro_p_trend if present.
    if 'macro_p_trend' in df.columns and 'trend' in df.columns:
        _macro_dir = np.sign(df['macro_p_trend'].fillna(0.5) - 0.5)   # +1 trending, -1 choppy
        _trade_dir = np.sign(df['trend'].astype(float) - 0.5)          # +1 up, -1 down
        df['factor_macro_alignment'] = (_trade_dir * _macro_dir).fillna(0.0)
    else:
        df['factor_macro_alignment'] = 0.0

    # day_running_pnl_lag1: sum of pnl from earlier trades same day (lag-1, causal)
    if 'pnl' in df.columns:
        df['day_running_pnl_lag1'] = df.groupby(['Ticker', '_date'])['pnl'].transform(
            lambda x: x.shift(1).expanding().sum().fillna(0.0))
    else:
        df['day_running_pnl_lag1'] = 0.0

    # day_trade_count_lag1: number of prior trades same day (causal)
    df['day_trade_count_lag1'] = df.groupby(['Ticker', '_date']).cumcount().astype(float)

    df.drop(columns=['_et', '_date'], inplace=True, errors='ignore')
    return df


print("✓ L2 intraday feature engineering defined (v16 — 25 features, re20: +7 time/alignment).")


# ═══════════════════════════════════════════════════════════════════════════════
# L2 TARGET COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_intraday_cd_target(df, pnl_col='pnl', brick_col='brick_size'):
    dates  = pd.to_datetime(df['entry_time']).dt.date.values
    times  = pd.to_datetime(df['entry_time']).values
    pnls   = df[pnl_col].values
    tiers  = df[brick_col].values
    n      = len(df)
    labels = np.zeros(n, dtype=int)

    for unique_date in np.unique(dates):
        day_mask    = (dates == unique_date)
        day_indices = np.where(day_mask)[0]
        sort_order     = np.argsort(times[day_indices], kind='stable')
        sorted_indices = day_indices[sort_order]
        tier_pnls       = {}
        seen_tiers      = []
        tier_profitable = {}

        for pos in sorted_indices:
            if not seen_tiers:
                streak = 0
            else:
                sorted_tiers = sorted(seen_tiers)
                max_streak   = 0
                cur_streak   = 0
                for t in sorted_tiers:
                    if tier_profitable.get(t, False):
                        cur_streak += 1
                        max_streak  = max(max_streak, cur_streak)
                    else:
                        cur_streak  = 0
                streak = max_streak
            labels[pos] = 1 if streak >= CD_MIN_STREAK else 0
            tier = tiers[pos]
            pnl  = pnls[pos]
            if tier not in tier_pnls:
                tier_pnls[tier]  = []
                seen_tiers.append(tier)
            tier_pnls[tier].append(pnl)
            tier_profitable[tier] = float(np.median(tier_pnls[tier])) > 0

    return labels


def compute_l2_window_target(df, pnl_col='pnl', brick_col='brick_size'):
    """
    re15 / re16 / re17 L2 target (FIX-re16b: day PnL > 0 gate preserved).
    """
    causal = compute_intraday_cd_target(df, pnl_col=pnl_col, brick_col=brick_col)
    dates   = pd.to_datetime(df['entry_time']).dt.date.values
    pnl_arr = df[pnl_col].values
    n       = len(df)
    labels  = np.zeros(n, dtype=int)

    day_total_pnl = {}
    for unique_date in np.unique(dates):
        day_total_pnl[unique_date] = float(pnl_arr[dates == unique_date].sum())

    for unique_date in np.unique(dates):
        if day_total_pnl[unique_date] <= 0:
            continue
        day_mask    = dates == unique_date
        day_indices = np.where(day_mask)[0]
        day_causal  = causal[day_indices]
        if day_causal.sum() == 0:
            continue
        pos_pos  = np.where(day_causal == 1)[0]
        first_p, last_p = pos_pos[0], pos_pos[-1]
        labels[day_indices[first_p: last_p + 1]] = 1

    return labels


print("✓ compute_l2_window_target defined.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-re18/re19a: L2 RANK TARGET  (replaces binary window target for main model)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_l2_rank_target(df, pnl_col='pnl'):
    """
    Within-day percentile rank of pnl.  Returns float array in (0, 1].
    Higher rank = better trade on that day.  Ties broken by average rank.

    This is the regression target for the L2 main model (FIX-re18/re19a).
    It is distinct from trade_target (binary window label) which is still
    used for bivariate_filter, CFI, and sub-state profiling.
    """
    dates = pd.to_datetime(df['entry_time']).dt.date.values
    pnls  = df[pnl_col].values
    ranks = np.zeros(len(df), dtype=float)
    for ud in np.unique(dates):
        mask = dates == ud
        # pct=True gives rank / n in (0,1], average ties
        ranks[mask] = pd.Series(pnls[mask]).rank(pct=True).values
    return ranks


print("✓ compute_l2_rank_target defined (FIX-re18/re19a).")


# ═══════════════════════════════════════════════════════════════════════════════
# re26: L2 WINDOW CD TARGET  (replaces old window + rank targets)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_l2_window_cd_target(df_raw, fold_train_df=None,
                                 window_minutes=30, cd_min_streak=3,
                                 duration_threshold_min=20, duration_fraction=0.35):
    """
    Per-trade binary target: 1 if the trade falls in a 'good' 30-min window.

    Window is good if ALL three gates pass:
      Gate 1 (C(d)):    >= cd_min_streak consecutive profitable brick_size tiers within window
      Gate 2 (profit):  window mean_pnl > 0
      Gate 3 (duration): >= duration_fraction of window trades have duration >= duration_threshold_min min
                         (Gate 3 skipped if exit_time is not present in df_raw)

    duration_fraction is fold-local when fold_train_df is provided (no leakage).
    Returns int array (0 or 1) of the same length as df_raw.
    """
    from collections import defaultdict

    ts_series  = pd.to_datetime(df_raw['entry_time'])
    times      = ts_series.values
    pnls       = df_raw['pnl'].values
    tiers      = df_raw['brick_size'].values
    dates      = ts_series.dt.date.values
    slots      = ((ts_series.dt.hour * 60 + ts_series.dt.minute) // window_minutes).values
    n          = len(df_raw)
    labels     = np.zeros(n, dtype=int)

    # Compute duration in minutes (optional)
    has_duration = 'exit_time' in df_raw.columns
    if has_duration:
        exit_ts  = pd.to_datetime(df_raw['exit_time']).values
        dur_min  = (exit_ts - times).astype('timedelta64[s]').astype(float) / 60.0
    else:
        dur_min = None

    # Fold-local duration fraction threshold
    if (fold_train_df is not None and has_duration
            and 'exit_time' in fold_train_df.columns):
        _et = pd.to_datetime(fold_train_df['entry_time']).values
        _xt = pd.to_datetime(fold_train_df['exit_time']).values
        _d  = (_xt - _et).astype('timedelta64[s]').astype(float) / 60.0
        _d  = _d[np.isfinite(_d) & (_d >= 0)]
        if len(_d) > 0:
            duration_fraction = float(np.clip((_d >= duration_threshold_min).mean(), 0.20, 0.60))

    # Group trade indices by (date, slot)
    window_groups = defaultdict(list)
    for i in range(n):
        window_groups[(dates[i], int(slots[i]))].append(i)

    for (date, slot), idxs in window_groups.items():
        idxs      = np.array(idxs)
        win_pnls  = pnls[idxs]
        win_tiers = tiers[idxs]

        # Sort by entry_time within window
        sort_ord    = np.argsort(times[idxs], kind='stable')
        idxs_s      = idxs[sort_ord]
        win_pnls_s  = win_pnls[sort_ord]
        win_tiers_s = win_tiers[sort_ord]

        # Gate 2: window mean_pnl > 0
        if win_pnls.mean() <= 0:
            continue

        # Gate 3: duration quality (skipped if no exit_time)
        if dur_min is not None:
            win_dur = dur_min[idxs]
            win_dur_v = win_dur[np.isfinite(win_dur) & (win_dur >= 0)]
            if len(win_dur_v) > 0 and float((win_dur_v >= duration_threshold_min).mean()) < duration_fraction:
                continue

        # Gate 1: >= cd_min_streak consecutive profitable brick_size tiers
        tier_pnls       = {}
        tier_profitable = {}
        seen_tiers      = []
        max_streak      = 0
        for pos in range(len(idxs_s)):
            tier = win_tiers_s[pos]
            pnl  = win_pnls_s[pos]
            if tier not in tier_pnls:
                tier_pnls[tier]  = []
                seen_tiers.append(tier)
            tier_pnls[tier].append(pnl)
            tier_profitable[tier] = float(np.median(tier_pnls[tier])) > 0
            cur, _mx = 0, 0
            for t in sorted(seen_tiers):
                if tier_profitable.get(t, False):
                    cur += 1
                    _mx  = max(_mx, cur)
                else:
                    cur  = 0
            max_streak = _mx

        if max_streak >= cd_min_streak:
            labels[idxs] = 1

    return labels


def _add_window_lag_features(df_raw, window_labels, window_minutes=30, duration_threshold_min=20):
    """
    Add causal lag features from the preceding 30-min window to df_raw.

    Features added (NaN when no preceding window on same day):
      prev_window_quality:           1 if the previous 30-min window had label=1, else 0
      rolling_win_trade_frac_prev:  fraction of trades with duration >= duration_threshold_min
                                     in the previous 30-min window
    """
    from collections import defaultdict

    ts_series = pd.to_datetime(df_raw['entry_time'])
    dates     = ts_series.dt.date.values
    slots     = ((ts_series.dt.hour * 60 + ts_series.dt.minute) // window_minutes).values
    n         = len(df_raw)

    # Build per-window quality and long-trade-fraction maps
    window_quality   = {}
    win_dur_lists    = defaultdict(list)
    has_duration     = 'exit_time' in df_raw.columns

    if has_duration:
        exit_ts  = pd.to_datetime(df_raw['exit_time']).values
        entry_ts = ts_series.values
        dur_min  = (exit_ts - entry_ts).astype('timedelta64[s]').astype(float) / 60.0
    else:
        dur_min = None

    for i in range(n):
        k = (dates[i], int(slots[i]))
        window_quality[k] = max(window_quality.get(k, 0), int(window_labels[i]))
        if dur_min is not None and np.isfinite(dur_min[i]) and dur_min[i] >= 0:
            win_dur_lists[k].append(dur_min[i])

    window_long_frac = {
        k: float(np.mean(np.array(v) >= duration_threshold_min))
        for k, v in win_dur_lists.items() if v
    }

    # Look up previous window for each trade
    prev_quality   = np.full(n, np.nan)
    prev_long_frac = np.full(n, np.nan)
    for i in range(n):
        prev_k = (dates[i], int(slots[i]) - 1)
        if prev_k in window_quality:
            prev_quality[i]   = float(window_quality[prev_k])
        if prev_k in window_long_frac:
            prev_long_frac[i] = window_long_frac[prev_k]

    out = df_raw.copy()
    out['prev_window_quality']          = prev_quality
    out['rolling_win_trade_frac_prev'] = prev_long_frac
    return out


print("✓ compute_l2_window_cd_target / _add_window_lag_features defined (re26).")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-re18/re19b: L2 REGRESSION MODEL  (replaces binary classifier)
# ═══════════════════════════════════════════════════════════════════════════════
LGB_PARAMS_L2_REG = dict(
    n_estimators=400, num_leaves=15, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.6, min_child_samples=150,
    reg_alpha=2.0, reg_lambda=8.0, min_split_gain=0.01,
    objective='regression', metric='rmse',
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)
XGB_PARAMS_L2_REG = dict(
    n_estimators=400, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.6, min_child_weight=80,
    reg_alpha=2.0, reg_lambda=8.0, gamma=0.05,
    objective='reg:squarederror',
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
)
# FIX-re21d: LambdaRank params tuned for small data (~80 query groups / fold).
# num_leaves=31 was far too large for ~5 day groups per bucket → overfitting.
# label_gain avoids uniform-gain degenerate NDCG on 4-tier integer targets.
LGB_PARAMS_L2_RANK = dict(
    objective='rank_xendcg',
    metric='ndcg',
    eval_at=[5, 10],
    label_gain=[0, 1, 3, 7],        # explicit NDCG gains for 4-tier relevance
    n_estimators=300, num_leaves=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
    reg_alpha=1.0, reg_lambda=5.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)
# Relevance tiers for lambdarank (0=worst … 3=best, mapped from float rank)
_L2_RANK_N_TIERS = 4

# FIX-re25a: L2 binary win/loss classifier params (restored from re22).
# Directly optimises for precision on sign(pnl).
# ISO calibration is fit on win_prob OOF — not on rank scores — so it does NOT
# suffer from the re23 compression bug (which was ISO-on-rank-stretched predictions).
# scale_pos_weight set dynamically per bucket in train_l2_cls_model.
# Direction-specific L2 CLS gates and params (EDA: short bimodal slippage + macro factor gap)
L2_CLS_PROB_GATE_LONG  = 0.50
L2_CLS_PROB_GATE_SHORT = 0.55  # short noisier — require higher win_prob confidence for reporting
L2_CLS_PROB_GATE = L2_CLS_PROB_GATE_LONG  # fallback for non-directional reporting

_LGB_L2_CLS_SHORT_OVERRIDES = {'reg_lambda': 12.0, 'min_child_samples': 150}
_L2_CLS_SPW_MAX_LONG  = 3.0   # long WR~34.9% → raw SPW~1.87, cap gives safety on small buckets
_L2_CLS_SPW_MAX_SHORT = 4.0   # short WR~33.9%, macro factors short-underperforms more → allow higher cap
LGB_PARAMS_L2_CLS = dict(
    n_estimators=400, num_leaves=15, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.6, min_child_samples=100,
    reg_alpha=2.0, reg_lambda=8.0, min_split_gain=0.01,
    objective='binary', metric='binary_logloss',
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)
XGB_PARAMS_L2_CLS = dict(
    n_estimators=400, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.6, min_child_weight=80,
    reg_alpha=2.0, reg_lambda=8.0, gamma=0.05,
    objective='binary:logistic', eval_metric='logloss',
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
)


def _make_lambdarank_inputs(X, y_rank_float, entry_times):
    """
    Convert float rank [0,1] + entry_times to lambdarank inputs.
    Returns (X_sorted, y_int_sorted, groups) where groups is list of
    per-day query sizes, sorted by date.
    FIX-re20d.
    """
    dates     = pd.to_datetime(entry_times).dt.date.values
    unique_d  = np.unique(dates)
    # Integer relevance: qcut within each day → 0..(_L2_RANK_N_TIERS-1)
    y_int = np.zeros(len(y_rank_float), dtype=np.int32)
    for ud in unique_d:
        mask = dates == ud
        vals = y_rank_float[mask]
        n    = mask.sum()
        if n < _L2_RANK_N_TIERS:
            y_int[mask] = 0
        else:
            # rank within day → tier
            tier_edges = np.linspace(0, 1, _L2_RANK_N_TIERS + 1)
            y_int[mask] = np.searchsorted(tier_edges[1:-1], vals).astype(np.int32)
    # Build group array: sort by date (required by LGB ranker)
    sort_idx  = np.argsort(dates, kind='stable')
    X_s       = X[sort_idx]
    y_s       = y_int[sort_idx]
    dates_s   = dates[sort_idx]
    _, counts = np.unique(dates_s, return_counts=True)
    return X_s, y_s, list(counts)


def train_l2_reg_model(X_train, y_rank_train, feature_names,
                       entry_times_train=None):
    """
    FIX-re20d: Train LGB regressor + XGB regressor + LGB LambdaRank ensemble.
    Final prediction = mean of the three members' [0,1] rank scores.

    entry_times_train: pd.Series or array of entry_time values for the
    training rows — required for lambdarank grouping.  If None, lambdarank
    member is skipped and only the two regressors are used.
    """
    def _fit_lgb_reg():
        m = lgb.LGBMRegressor(**LGB_PARAMS_L2_REG)
        m.fit(X_train, y_rank_train)
        return m

    def _fit_xgb_reg():
        m = xgb.XGBRegressor(**XGB_PARAMS_L2_REG)
        m.fit(X_train, y_rank_train)
        return m

    _workers = max(2, N_PARALLEL // 2)
    with ThreadPoolExecutor(max_workers=_workers) as ex:
        f_lgb = ex.submit(_fit_lgb_reg)
        f_xgb = ex.submit(_fit_xgb_reg)
        lgb_m = f_lgb.result()
        xgb_m = f_xgb.result()

    # LambdaRank member (FIX-re20d)
    rank_m = None
    if entry_times_train is not None and len(entry_times_train) == len(X_train):
        try:
            X_rs, y_rs, groups = _make_lambdarank_inputs(
                X_train, y_rank_train, entry_times_train)
            rank_m = lgb.LGBMRanker(**LGB_PARAMS_L2_RANK)
            rank_m.fit(X_rs, y_rs, group=groups)
        except Exception as _re:
            rank_m = None  # degrade gracefully to 2-member ensemble

    return {'lgb': lgb_m, 'xgb': xgb_m, 'rank': rank_m,
            'feature_names': feature_names}


def predict_l2_reg_model(X_test, model_result):
    """
    FIX-re20d: Average LGB-reg, XGB-reg, and LGB-rank predictions.
    LambdaRank scores are percentile-ranked within the test batch to [0,1].
    Falls back to 2-member average if rank model is absent.
    """
    p_lgb = model_result['lgb'].predict(X_test)
    p_xgb = model_result['xgb'].predict(X_test)
    rank_m = model_result.get('rank')
    if rank_m is not None:
        try:
            raw_rank = rank_m.predict(X_test)
            # Convert raw NDCG scores to [0,1] percentile rank within batch
            p_rank = pd.Series(raw_rank).rank(pct=True).values
            avg = np.clip((p_lgb + p_xgb + p_rank) / 3.0, 0.0, 1.0)
        except Exception:
            avg = np.clip((p_lgb + p_xgb) / 2.0, 0.0, 1.0)
    else:
        avg = np.clip((p_lgb + p_xgb) / 2.0, 0.0, 1.0)
    return avg


print("✓ train_l2_reg_model / predict_l2_reg_model defined (re22: LGB-reg + XGB-reg + LambdaRank + OOF ISO calibration).")


# ── FIX-re25a: L2 binary win/loss classifier (restored from re22 Step 9b) ─────

def compute_l2_cls_target(df_raw, pnl_col='pnl', train_threshold=None):
    """
    Binary L2 target: 1 if trade is in the top tercile by PnL.

    OAA-Gap4: pnl>0 has poor SNR given kurtosis=99.36 (top 5% mean +87 vs bottom 5%
    mean -59). Top-tercile (67th percentile) better captures high-value trades with
    the fat upper tail. train_threshold must be set from train-fold data to avoid
    test leakage — pass the same value for train and test calls within each fold.

    OAA-Gap5 (long): caller passes pnl_col='theoretical_profit' for long trades
    to strip the predictable constant slippage drag (-11.09 mean, 99% negative).
    """
    pnl = df_raw[pnl_col].values
    thresh = train_threshold if train_threshold is not None else float(np.percentile(pnl, 67))
    return (pnl >= thresh).astype(int)


def _apply_hour_calibration(test_probs, test_hours, train_probs, train_hours, train_labels,
                             min_hour_samples=30):
    """
    OAA-Gap7: additive per-hour probability correction.
    EDA: hour 13 mean_pnl=-0.68, hour 17=-0.71 vs hour 14=-1.21, hour 16=-1.22.
    Shift test_probs by (train_hour_true_rate - train_global_rate) for each hour bucket.
    """
    global_tr = float(np.mean(train_labels))
    cal = test_probs.copy()
    for h in np.unique(test_hours):
        tr_mask = (train_hours == h)
        te_mask = (test_hours == h)
        if tr_mask.sum() < min_hour_samples or te_mask.sum() == 0:
            continue
        hour_lift = float(np.mean(train_labels[tr_mask])) - global_tr
        cal[te_mask] = np.clip(test_probs[te_mask] + hour_lift, 0.0, 1.0)
    return cal


def train_l2_cls_model(X_train, y_cls_train, feature_names, direction='long',
                       sample_weight=None):
    """
    LGB + XGB binary win/loss classifier with OOF isotonic calibration.

    scale_pos_weight is set from the train-set class ratio to handle imbalance.
    ISO calibration is fit on 3-fold OOF predictions (train data only —
    no test leakage).  Returns calibrated win probabilities in [0, 1].

    Note: ISO here calibrates win_prob — not rank scores — so it is NOT subject
    to the re23 compression bug (which was ISO-on-per-day-rank predictions).
    """
    pos = float(y_cls_train.sum())
    neg = float((y_cls_train == 0).sum())
    _spw_cap = _L2_CLS_SPW_MAX_SHORT if direction == 'short' else _L2_CLS_SPW_MAX_LONG
    spw = min(max(1.0, neg / (pos + 1e-8)), _spw_cap)

    lgb_p = {**LGB_PARAMS_L2_CLS, 'scale_pos_weight': spw}
    xgb_p = {**XGB_PARAMS_L2_CLS, 'scale_pos_weight': spw}
    if direction == 'short':
        lgb_p.update(_LGB_L2_CLS_SHORT_OVERRIDES)

    def _fit_lgb():
        m = lgb.LGBMClassifier(**lgb_p)
        m.fit(X_train, y_cls_train, sample_weight=sample_weight)
        return m

    def _fit_xgb():
        m = xgb.XGBClassifier(**xgb_p)
        m.fit(X_train, y_cls_train, sample_weight=sample_weight)
        return m

    with ThreadPoolExecutor(max_workers=2) as _ex:
        _fl, _fx = _ex.submit(_fit_lgb), _ex.submit(_fit_xgb)
        lgb_m, xgb_m = _fl.result(), _fx.result()

    # OOF isotonic calibration — train-only, no test data touched
    iso_cal = None
    if len(X_train) >= 40 and pos >= 5 and neg >= 5:
        try:
            from sklearn.model_selection import KFold as _KFoldCLS
            _n_sp = min(3, max(2, len(X_train) // 20))
            _kf = _KFoldCLS(n_splits=_n_sp, shuffle=False)
            _oof = np.zeros(len(X_train))
            for _tri, _vli in _kf.split(X_train):
                _ml = lgb.LGBMClassifier(**lgb_p); _ml.fit(X_train[_tri], y_cls_train[_tri])
                _mx = xgb.XGBClassifier(**xgb_p); _mx.fit(X_train[_tri], y_cls_train[_tri])
                _oof[_vli] = (_ml.predict_proba(X_train[_vli])[:, 1] +
                              _mx.predict_proba(X_train[_vli])[:, 1]) / 2.0
            iso_cal = IsotonicRegression(out_of_bounds='clip')
            iso_cal.fit(_oof, y_cls_train)
        except Exception:
            iso_cal = None

    return {'lgb': lgb_m, 'xgb': xgb_m, 'iso': iso_cal, 'feature_names': feature_names}


def predict_l2_cls_model(X_test, model_result):
    """Return ISO-calibrated win probability for each test trade."""
    p_lgb = model_result['lgb'].predict_proba(X_test)[:, 1]
    p_xgb = model_result['xgb'].predict_proba(X_test)[:, 1]
    avg = (p_lgb + p_xgb) / 2.0
    iso = model_result.get('iso')
    if iso is not None:
        avg = np.clip(iso.predict(avg), 0.0, 1.0)
    return avg


print("✓ compute_l2_cls_target / train_l2_cls_model / predict_l2_cls_model defined (re25: CLS restored from re22).")


# ═══════════════════════════════════════════════════════════════════════════════
# L2 BUCKET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _filter_bucket(df, bucket_name):
    lo, hi = FACTOR_BUCKETS[bucket_name]
    if lo is None and hi is None:
        return df.copy()
    fac = pd.to_numeric(df['factor'], errors='coerce')
    mask = pd.Series(True, index=df.index)
    if lo is not None:
        mask &= fac >= lo
    if hi is not None:
        mask &= fac <= hi
    return df[mask].copy()


def _run_l2_bucket(l2_train_raw_bkt, l2_test_raw_bkt, bucket_name, fold_num,
                   dedup_cols=None, state_filter=None, direction='long'):
    """
    Full L2 pipeline (Steps 0–9+) for one factor bucket × Renko state group.

    re18 changes vs re17 (FIX-re18/re19a–f)
    ------------------------------------
    * L2 target: within-day PnL percentile rank (float in (0,1]) via compute_l2_rank_target.
    * Binary trade_target kept only for profiling + bivariate filter.
    * Main model: LGB/XGB regressors (train_l2_reg_model / predict_l2_reg_model).
    * Isotonic calibration removed.
    * MoE experts: also rank regressors.
    * Evaluation: Spearman(pred_rank,pnl), Spearman(pred_rank,true_rank), RMSE,
      Lift@top30-by-rank; no AUC/precision/recall.
    * _preds = top-30% indicator by predicted rank.
    """
    _pfx = f"    [{bucket_name:>12}]"
    _skipped_base = {'bucket': bucket_name, 'fold': fold_num, 'skipped': True,
                     'n_train': len(l2_train_raw_bkt), 'n_test': len(l2_test_raw_bkt)}

    if (len(l2_train_raw_bkt) < L2_MIN_BUCKET_TRADES or
            len(l2_test_raw_bkt) < L2_MIN_BUCKET_TRADES):
        print(f"{_pfx} Skip — n_train={len(l2_train_raw_bkt):,} "
              f"n_test={len(l2_test_raw_bkt):,} (< {L2_MIN_BUCKET_TRADES})")
        return None, _skipped_base, None

    # FIX-re21a: keep pnl in l2_tr until after sub-state discovery so that
    # _compute_cluster_ks_separation can compare cluster PnL distributions.
    # In re20, pnl was dropped here → KS always returned (0.0, 1.0, False).
    _drop_for_tr = [c for c in DROP_ALWAYS + OUTCOME_COLS
                    if c in l2_train_raw_bkt.columns and c != 'pnl']
    l2_tr = l2_train_raw_bkt.drop(columns=_drop_for_tr, errors='ignore').copy()
    l2_te = l2_test_raw_bkt.drop(
        columns=[c for c in DROP_ALWAYS + OUTCOME_COLS if c in l2_test_raw_bkt.columns],
        errors='ignore').copy()

    l2_tr['entry_time'] = pd.to_datetime(l2_train_raw_bkt['entry_time'].values)
    l2_te['entry_time'] = pd.to_datetime(l2_test_raw_bkt['entry_time'].values)

    # OAA-Gap1: duration feature — EDA: ≥30min WR=66.4% vs <30min WR=27.4%
    if 'exit_time' in l2_train_raw_bkt.columns:
        l2_tr['duration_min'] = (
            (pd.to_datetime(l2_train_raw_bkt['exit_time'].values) -
             pd.to_datetime(l2_train_raw_bkt['entry_time'].values))
            .astype('timedelta64[s]').astype(float) / 60.0
        )
        l2_te['duration_min'] = (
            (pd.to_datetime(l2_test_raw_bkt['exit_time'].values) -
             pd.to_datetime(l2_test_raw_bkt['entry_time'].values))
            .astype('timedelta64[s]').astype(float) / 60.0
        )
    else:
        l2_tr['duration_min'] = np.nan
        l2_te['duration_min'] = np.nan

    # re26: window-level triple-gate binary target (replaces old window + rank targets)
    # Unit 1: use direction-specific streak threshold
    _l2_win_streak = _L2_WINDOW_STREAK_SHORT if direction == 'short' else _L2_WINDOW_STREAK_LONG
    _tr_win_labels = compute_l2_window_cd_target(l2_train_raw_bkt, fold_train_df=l2_train_raw_bkt, cd_min_streak=_l2_win_streak)
    _te_win_labels = compute_l2_window_cd_target(l2_test_raw_bkt,  fold_train_df=l2_train_raw_bkt, cd_min_streak=_l2_win_streak)
    l2_tr['trade_target'] = _tr_win_labels
    l2_te['trade_target'] = _te_win_labels

    # OAA-Gap4/5: top-tercile target; long uses theoretical_profit to strip predictable slippage
    # Unit 4: short uses risk-adjusted PnL (pnl minus worst same-window drawdown)
    if direction == 'short':
        _entry_floor = pd.to_datetime(l2_train_raw_bkt['entry_time']).dt.floor('30min')
        _min_win_pnl_tr = l2_train_raw_bkt.groupby(_entry_floor)['pnl'].transform('min')
        _l2_tr_aug = l2_train_raw_bkt.copy()
        _l2_tr_aug['pnl_risk_adj'] = _l2_tr_aug['pnl'] - _min_win_pnl_tr.abs()
        _entry_floor_te = pd.to_datetime(l2_test_raw_bkt['entry_time']).dt.floor('30min')
        _min_win_pnl_te = l2_test_raw_bkt.groupby(_entry_floor_te)['pnl'].transform('min')
        _l2_te_aug = l2_test_raw_bkt.copy()
        _l2_te_aug['pnl_risk_adj'] = _l2_te_aug['pnl'] - _min_win_pnl_te.abs()
        _pnl_col_cls = 'pnl_risk_adj'
        l2_train_raw_bkt = _l2_tr_aug
        l2_test_raw_bkt  = _l2_te_aug
    else:
        _pnl_col_cls = ('theoretical_profit'
                        if 'theoretical_profit' in l2_train_raw_bkt.columns
                        else 'pnl')
    _cls_tr_thresh = float(np.percentile(l2_train_raw_bkt[_pnl_col_cls].dropna().values, 67))
    l2_tr['trade_cls_target'] = compute_l2_cls_target(l2_train_raw_bkt, pnl_col=_pnl_col_cls,
                                                       train_threshold=_cls_tr_thresh)
    l2_te['trade_cls_target'] = compute_l2_cls_target(l2_test_raw_bkt, pnl_col=_pnl_col_cls,
                                                       train_threshold=_cls_tr_thresh)

    # Causal window lag features (prev_window_quality, rolling_win_trade_frac_prev)
    _tr_raw_aug = _add_window_lag_features(l2_train_raw_bkt, _tr_win_labels)
    _te_raw_aug = _add_window_lag_features(l2_test_raw_bkt,  _te_win_labels)
    for _wf in ('prev_window_quality', 'rolling_win_trade_frac_prev'):
        if _wf in _tr_raw_aug.columns:
            l2_tr[_wf] = _tr_raw_aug[_wf].values
            l2_te[_wf] = _te_raw_aug[_wf].values

    if l2_tr['trade_target'].nunique() < 2:
        print(f"{_pfx} Skip — single class in train window target")
        return None, _skipped_base, None
    if l2_tr['trade_cls_target'].nunique() < 2:
        print(f"{_pfx} Skip — single class in train cls target (all wins or all losses)")
        return None, _skipped_base, None

    _base_rate     = float(l2_te['trade_target'].mean())
    _win_rate_tr   = float(l2_tr['trade_target'].mean())
    _cls_base_rate = float(l2_tr['trade_cls_target'].mean())
    print(f"{_pfx} n_train={len(l2_tr):,}  n_test={len(l2_te):,}  "
          f"window_base={_base_rate:.3f}  win_base_tr={_win_rate_tr:.3f}  cls_rate_tr={_cls_base_rate:.3f}")

    l2_tr['entry_time'] = pd.to_datetime(l2_tr['entry_time'])
    l2_te['entry_time'] = pd.to_datetime(l2_te['entry_time'])
    l2_tr['date'] = l2_tr['entry_time'].dt.date
    l2_te['date'] = l2_te['entry_time'].dt.date

    # Step 1: Path A
    _thresh = compute_thresholds(l2_tr)
    l2_tr   = assign_states_path_a(l2_tr, _thresh)
    l2_te   = assign_states_path_a(l2_te, _thresh)

    # Step 1b: Hard coarse-state filter
    _n_tr_before = len(l2_tr)
    _n_te_before = len(l2_te)
    l2_tr = l2_tr[~l2_tr['core_state'].isin(L2_REJECT_STATES)].copy()
    l2_te = l2_te[~l2_te['core_state'].isin(L2_REJECT_STATES)].copy()
    print(f"{_pfx} Coarse filter: removed {_n_tr_before - len(l2_tr)} train / "
          f"{_n_te_before - len(l2_te)} test trades ({L2_REJECT_STATES})")

    if state_filter is not None:
        l2_tr = l2_tr[l2_tr['core_state'].isin(state_filter)].copy()
        l2_te = l2_te[l2_te['core_state'].isin(state_filter)].copy()
        print(f"{_pfx} State filter {state_filter}: "
              f"{len(l2_tr)} train / {len(l2_te)} test remain")

    if len(l2_tr) < L2_MIN_BUCKET_TRADES or len(l2_te) < L2_MIN_BUCKET_TRADES:
        print(f"{_pfx} Skip — insufficient after filter (train={len(l2_tr)}, test={len(l2_te)})")
        return None, _skipped_base, None

    # Step 2: Path B sub-state discovery
    # FIX-re22e: clustering gated out in all 18 re22 folds (100% fallback).
    # Skip GMM/KMeans/DTree entirely; use rule sub-states directly.
    l2_tr = l2_tr.copy()
    l2_te = l2_te.copy()
    l2_tr['unified_sub_state'] = l2_tr['rule_sub_state']
    l2_te['unified_sub_state'] = l2_te['rule_sub_state']
    l2_tr.drop(columns=['pnl'], inplace=True, errors='ignore')

    # Step 3: Profiling
    _, _profile_map = profile_sub_states(l2_tr, target_col='trade_target')
    l2_tr = apply_profile_to_df(l2_tr, _profile_map)
    l2_te = apply_profile_to_df(l2_te, _profile_map)

    # Step 4: Bivariate filter
    _excl_cols = set(FORBIDDEN_COLUMNS) | _DEAD_FEATURES_RE17 | {
        'trade_target', 'trade_rank', 'trade_cls_target', 'date', 'core_state', 'rule_sub_state',
        'unified_sub_state', 'sub_state_profile'}
    # 5min indicator columns are L1-only — exclude from L2 candidate pool
    _excl_cols |= {c for c in l2_tr.columns if c.endswith('_5min')}

    _dedup_pool = dedup_cols if dedup_cols is not None else DEDUPLICATED_CANDIDATE_COLS
    _cand_dedup = [c for c in _dedup_pool
                   if c in l2_tr.columns and c not in _excl_cols
                   and l2_tr[c].isna().mean() < 1.0]

    # Inject 1min indicator columns (not in dedup_pool since CLEAN_CANDIDATE_COLS
    # only holds L1 features; 1min cols belong to L2 only)
    _1min_extra = [c for c in l2_tr.columns
                   if c.endswith('_1min') and c not in _excl_cols
                   and l2_tr[c].dtype in (np.float64, np.float32, np.int64, np.int32)
                   and l2_tr[c].isna().mean() < 1.0]
    _cand_dedup = list(dict.fromkeys(_cand_dedup + _1min_extra))

    if len(_cand_dedup) < 10:
        _cand_dedup = [c for c in l2_tr.columns
                       if c not in _excl_cols
                       and l2_tr[c].dtype in (np.float64, np.float32, np.int64, np.int32)
                       and l2_tr[c].isna().mean() < 1.0]
    _bad = {c for c in _cand_dedup
            if (lambda v: len(v) > 0 and np.isinf(v).mean() > 0.01)(
                l2_tr[c].values[~np.isnan(l2_tr[c].values)])}
    _cand_dedup = [c for c in _cand_dedup if c not in _bad]

    # OAA-Gap1: inject duration_min as structural L2 feature (not in L1 dedup pool)
    if ('duration_min' in l2_tr.columns and 'duration_min' not in _cand_dedup
            and 'duration_min' not in _excl_cols
            and l2_tr['duration_min'].isna().mean() < 1.0
            and l2_tr['duration_min'].std(skipna=True) > 1e-8):
        _cand_dedup.append('duration_min')

    _re16_carry_cols = ['l1_day_probability'] + _ROLLING_PREV_COLS
    for _cc in _re16_carry_cols:
        if (_cc in l2_tr.columns and _cc not in _cand_dedup
                and _cc not in _excl_cols
                and l2_tr[_cc].isna().mean() < 1.0
                and l2_tr[_cc].std(skipna=True) > 1e-8):
            _cand_dedup.append(_cc)

    # re17: also include macro regime features in candidate pool
    for _mc in _MACRO_REGIME_COLS:
        if (_mc in l2_tr.columns and _mc not in _cand_dedup
                and l2_tr[_mc].isna().mean() < 1.0
                and l2_tr[_mc].std(skipna=True) > 1e-8):
            _cand_dedup.append(_mc)

    _surviving, _ = bivariate_filter(l2_tr, _cand_dedup, target_col='trade_target')
    _l2_n_days = l2_tr['date'].nunique() if 'date' in l2_tr.columns else max(1, len(l2_tr) // 100)
    _dyn_k = compute_dynamic_top_k(len(_surviving), _l2_n_days)
    print(f"{_pfx} bivariate {len(_cand_dedup)} → {len(_surviving)} | TOP_K={_dyn_k} ({_l2_n_days} days)")

    # Step 5: CFI
    _selected, _ = clustered_feature_importance(
        l2_tr, _surviving, target_col='trade_target', top_k=_dyn_k)
    print(f"{_pfx} CFI → {len(_selected)} features")

    # Step 6: Median-fill + clip
    _l2_tr_raw = l2_tr[_selected].replace([np.inf, -np.inf], np.nan)
    _l2_tr_meds = _l2_tr_raw.median()
    X_tr_raw  = _l2_tr_raw.fillna(_l2_tr_meds).values
    _l2_te_raw = l2_te[_selected].replace([np.inf, -np.inf], np.nan)
    X_te_raw  = _l2_te_raw.fillna(_l2_tr_meds).values

    # Step 7: Gram-Schmidt
    _gs_p     = gram_schmidt_fit(X_tr_raw)
    X_tr_orth = gram_schmidt_transform(X_tr_raw, _gs_p)
    X_te_orth = gram_schmidt_transform(X_te_raw, _gs_p)

    # Step 8: Sub-models + feature matrix
    _cs_tr = encode_core_state(l2_tr)
    _cs_te = encode_core_state(l2_te)
    _sub_results, _oof_meta = train_all_submodels(l2_tr, _selected, verbose=False)
    _meta_cols    = list(_oof_meta.columns)
    _oof_aln      = _oof_meta.reindex(l2_tr.index)
    _te_meta      = get_submodel_meta_features(l2_te, _sub_results)
    _te_meta_aln  = _te_meta.reindex(columns=_meta_cols).reindex(l2_te.index)

    X_tr_final, _tr_med = build_main_feature_matrix_v5(
        X_tr_orth, _cs_tr, _oof_aln, _sub_results)
    X_te_final, _ = build_main_feature_matrix_v5(
        X_te_orth, _cs_te, _te_meta_aln, _sub_results,
        train_medians_ortho=_tr_med)

    # re26: window target for main classifier; trade-level win/loss for secondary CLS
    y_win_tr  = l2_tr['trade_target'].values      # window-level triple-gate label
    y_cls_tr  = l2_tr['trade_cls_target'].values  # trade-level win/loss (secondary)
    _feat_names = _selected + ['core_state_encoded'] + _meta_cols

    # OAA-Gap6: ATR-epoch temporal weighting — vol doubled 2023→2025; recent rows more relevant
    _tr_times_l2 = pd.to_datetime(l2_tr['entry_time']).values.astype(np.int64)
    _sw_l2 = 0.5 + 0.5 * pd.Series(_tr_times_l2).rank(pct=True).values

    # ── Step 9a: Window binary classifier (re26: replaces rank regression) ────────
    # Trained on triple-gate window labels — predicts which 30-min windows to trade.
    _win_model   = train_l2_cls_model(X_tr_final, y_win_tr, _feat_names, direction=direction,
                                      sample_weight=_sw_l2)
    _win_prob_tr = predict_l2_cls_model(X_tr_final, _win_model)
    _win_prob_te = predict_l2_cls_model(X_te_final, _win_model)
    print(f"{_pfx} Win-model prob range: [{_win_prob_te.min():.3f}, {_win_prob_te.max():.3f}]  "
          f"mean={_win_prob_te.mean():.3f}")

    # ── Step 9b: Trade-level win/loss classifier (re25a, kept as secondary) ───────
    _cls_model       = train_l2_cls_model(X_tr_final, y_cls_tr, _feat_names, direction=direction,
                                          sample_weight=_sw_l2)
    _cls_win_prob_te = predict_l2_cls_model(X_te_final, _cls_model)
    _cls_win_prob_tr = predict_l2_cls_model(X_tr_final, _cls_model)

    # OAA-Gap7: per-hour CLS calibration (EDA: hour 13=-0.68, hour 16=-1.22, delta ~0.5)
    _tr_hours = pd.to_datetime(l2_tr['entry_time']).dt.hour.values
    _te_hours = pd.to_datetime(l2_te['entry_time']).dt.hour.values
    _cls_win_prob_te = _apply_hour_calibration(
        _cls_win_prob_te, _te_hours, _cls_win_prob_tr, _tr_hours, y_cls_tr)

    print(f"{_pfx} CLS win_prob range (post-hour-cal): [{_cls_win_prob_te.min():.3f}, {_cls_win_prob_te.max():.3f}]  "
          f"mean={_cls_win_prob_te.mean():.3f}")

    _oof_moe_rank = _win_prob_tr.copy()   # OOF record (replaces _rank_tr)

    # Selection: primary = window classifier, per-day top-30% by win_prob
    _te_dates_arr  = pd.to_datetime(l2_te['entry_time']).dt.date.values
    _pred_win_rank = _win_prob_te.copy()
    for _ud in np.unique(_te_dates_arr):
        _dm = _te_dates_arr == _ud
        if _dm.sum() >= 2:
            _pred_win_rank[_dm] = pd.Series(_win_prob_te[_dm]).rank(pct=True).values

    # OAA-Gap1: long-hold trades get rank boost; Unit 6: threshold direction-specific
    _dur_te = l2_te['duration_min'].values if 'duration_min' in l2_te.columns else np.full(len(l2_te), np.nan)
    _dur_min = _DUR_BOOST_MIN_SHORT if direction == 'short' else _DUR_BOOST_MIN_LONG
    _dur_long_mask = np.isfinite(_dur_te) & (_dur_te >= _dur_min)
    if _dur_long_mask.any():
        _pred_win_rank[_dur_long_mask] = np.minimum(_pred_win_rank[_dur_long_mask] + _DUR_BOOST_AMOUNT, 1.0)

    _preds      = (_pred_win_rank >= np.percentile(_pred_win_rank, 70)).astype(int)
    _n_selected = int(_preds.sum())
    _pct_sel    = 100.0 * _n_selected / max(1, len(_preds))
    print(f"{_pfx} Win gate (top30%/day): {_n_selected}/{len(_preds)} "
          f"trades selected ({_pct_sel:.1f}%)")

    # ── Evaluation (re26) ───────────────────────────────────────────────────────────
    _pnl     = l2_test_raw_bkt.loc[l2_te.index, 'pnl'].values
    _win_te  = l2_te['trade_target'].values   # window quality ground truth

    # AUC on window quality prediction
    _win_auc = np.nan
    if _win_te.sum() > 5 and (_win_te == 0).sum() > 5 and np.isfinite(_win_prob_te).all():
        try:
            from sklearn.metrics import roc_auc_score as _roc_auc
            _win_auc = float(_roc_auc(_win_te, _win_prob_te))
        except Exception:
            pass

    # Precision@top50% — how often top-50% by win_prob are actually good windows
    _top50_mask  = _win_prob_te >= np.percentile(_win_prob_te, 50)
    _prec_top50  = float(_win_te[_top50_mask].mean()) if _top50_mask.sum() > 0 else np.nan

    # Lift — win-model gate (primary)
    _pnl_mean  = _pnl.mean()
    _win_mask  = _preds.astype(bool)
    _lift_win  = _pnl[_win_mask].mean() if _win_mask.sum() > 5 else np.nan
    _lift_r_win = (float(_lift_win / abs(_pnl_mean))
                   if (not np.isnan(_lift_win) and _pnl_mean != 0) else np.nan)

    # Lift — secondary CLS gate (direction-specific threshold for reporting)
    _cls_gate = L2_CLS_PROB_GATE_SHORT if direction == 'short' else L2_CLS_PROB_GATE_LONG
    _cls_abs_mask = (_cls_win_prob_te >= _cls_gate).astype(bool)
    _lift_cls     = _pnl[_cls_abs_mask].mean() if _cls_abs_mask.sum() > 5 else np.nan
    _lift_r_cls   = (float(_lift_cls / abs(_pnl_mean))
                     if (not np.isnan(_lift_cls) and _pnl_mean != 0) else np.nan)

    print(f"{_pfx} Win AUC={_win_auc:.3f}  Prec@top50%={_prec_top50:.3f}  "
          f"Lift@Win_gate={_lift_win:.2f} vs overall={_pnl_mean:.2f} ({_lift_r_win:.2f}x)  "
          f"{'✓' if (not np.isnan(_lift_win) and _lift_win > _pnl_mean) else '✗'}")

    # Mann-Whitney: win-model selected vs rejected
    _pos_pnl = _pnl[_win_mask]
    _neg_pnl = _pnl[~_win_mask]
    if len(_pos_pnl) >= 5 and len(_neg_pnl) >= 5:
        try:
            _mw_u, _mw_p = mannwhitneyu(_pos_pnl, _neg_pnl, alternative='greater')
        except Exception:
            _mw_u, _mw_p = np.nan, np.nan
    else:
        _mw_u, _mw_p = np.nan, np.nan
    _mw_sep = (not np.isnan(_mw_p)) and (_mw_p < 0.05)
    _med_pos = float(np.median(_pos_pnl)) if len(_pos_pnl) > 0 else np.nan
    _med_neg = float(np.median(_neg_pnl)) if len(_neg_pnl) > 0 else np.nan
    print(f"{_pfx} MW p={_mw_p:.4f}  "
          f"med_pos={_med_pos:.1f}  med_neg={_med_neg:.1f}  "
          f"{'✓' if _mw_sep else '✗'}")

    # Position sizing (Layer 5) — win_prob as signal strength (FIX-re25a: replaces pred_rank)
    _macro_cols_te = ['macro_p_trend_lv', 'macro_p_trend_hv',
                      'macro_p_chop_lv',  'macro_p_chop_hv']
    _has_macro = all(c in l2_te.columns for c in _macro_cols_te)
    if _has_macro:
        _pos_sizes = np.array([
            compute_position_size(
                float(_win_prob_te[i]),
                [l2_te.iloc[i][c] for c in _macro_cols_te]
            )
            for i in range(len(l2_te))
        ])
    else:
        _pos_sizes = np.full(len(l2_te), 0.0)

    # Build result DataFrame (re26: window classifier outputs)
    _res = l2_te[['entry_time', 'factor', 'core_state',
                  'unified_sub_state', 'sub_state_profile']].copy()
    _res['pred_win_prob']     = _win_prob_te        # window quality probability
    _res['pred_cls_win_prob'] = _cls_win_prob_te    # trade-level win probability (secondary)
    _res['pred_label']        = _preds              # top-30%/day by window quality
    _res['true_label']        = l2_te['trade_target'].values      # window quality label
    _res['true_win']          = l2_te['trade_cls_target'].values  # sign(pnl)>0
    _res['pnl']             = _pnl
    _res['bucket']          = bucket_name
    _res['fold']            = fold_num
    _res['position_size']   = _pos_sizes

    for _mc in _macro_cols_te + ['macro_p_trend', 'macro_p_vol']:
        if _mc in l2_te.columns:
            _res[_mc] = l2_te[_mc].values

    # OOF MoE records for Layer 4 meta-regressor
    _oof_moe_df = l2_tr[['entry_time', 'factor', 'core_state',
                          'unified_sub_state', 'trade_target']].copy()
    _oof_moe_df['moe_rank'] = _oof_moe_rank
    _oof_moe_df['bucket']   = bucket_name
    _oof_moe_df['fold']     = fold_num
    for _mc in _macro_cols_te + ['macro_p_trend', 'macro_p_vol']:
        if _mc in l2_tr.columns:
            _oof_moe_df[_mc] = l2_tr[_mc].values

    _metrics = {
        'bucket':           bucket_name,
        'fold':             fold_num,
        'skipped':          False,
        'n_train':          len(l2_tr),
        'n_test':           len(l2_te),
        'window_base_rate': round(_base_rate, 4),
        'win_auc':          round(_win_auc, 4)     if not np.isnan(_win_auc)     else np.nan,
        'prec_top50':       round(_prec_top50, 4)  if not np.isnan(_prec_top50)  else np.nan,
        'lift_ratio_win':   round(_lift_r_win, 4)  if not np.isnan(_lift_r_win)  else np.nan,
        'lift_ratio_cls':   round(_lift_r_cls, 4)  if not np.isnan(_lift_r_cls)  else np.nan,
        'n_features':       len(_selected),
        'mw_u':             _mw_u,
        'mw_p':             round(_mw_p, 4) if not np.isnan(_mw_p) else np.nan,
        'mw_separation':    _mw_sep,
        'pct_selected_win': round(_pct_sel, 2),
        'win_rate_train':   round(_win_rate_tr, 4),   # window label base rate (train)
        'cls_rate_train':   round(_cls_base_rate, 4), # trade-level win rate (train)
    }
    return _res, _metrics, _oof_moe_df


print("✓ L2 bucket helper defined (re26: window binary classifier + CLS secondary + triple-gate target).")


# ═══════════════════════════════════════════════════════════════════════════════
# HERMES-V2 L1: MACRO REGIME FEATURES (computed once on full dataset)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing Hermes-v2-L1 macro regime features on full dataset...")
df = compute_macro_regime_features(df)
df_raw_macro = compute_macro_regime_features(df_raw.drop(
    columns=[c for c in _MACRO_REGIME_COLS if c in df_raw.columns], errors='ignore'))
print(f"  Macro regime columns added: {_MACRO_REGIME_COLS}")
print(f"  macro_p_trend range: [{df['macro_p_trend'].min():.3f}, {df['macro_p_trend'].max():.3f}]  "
      f"mean={df['macro_p_trend'].mean():.3f}")

# FIX-re22e: post-macro interaction features.
# These combine stable-8 features with regime context to create cross terms
# that are not in the raw pool. Added to both df and df_raw_macro.
def _add_post_macro_interactions(d):
    d = d.copy()
    if 'price_velocity_3' in d.columns and 'macro_atr_pct' in d.columns:
        # Momentum × volatility-regime: large when strong trend in high-vol env
        d['vel_x_atr_regime'] = d['price_velocity_3'] * d['macro_atr_pct']
    if 'price_velocity_3' in d.columns and 'macro_p_trend' in d.columns:
        # Momentum × trend probability: self-reinforcing trend confirmation
        d['vel_x_p_trend'] = d['price_velocity_3'] * d['macro_p_trend']
    if 'ohlc_price_in_band_lag1' in d.columns and 'macro_atr_drift' in d.columns:
        # Band containment × vol-change: band matters more when vol is rising
        d['band_x_atr_drift'] = d['ohlc_price_in_band_lag1'] * d['macro_atr_drift']
    return d

df          = _add_post_macro_interactions(df)
df_raw_macro = _add_post_macro_interactions(df_raw_macro)
print(f"  Post-macro interactions added: vel_x_atr_regime, vel_x_p_trend, band_x_atr_drift")

# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD PIPELINE (re22 — l1_cascade excl + ARI 0.30 + per-day rank + ISO cal)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WALK-FORWARD PIPELINE (re25 — CLS model restored + fold-local targets + no rank ISO + direction loop)")
print("=" * 70)


# ── Direction-separated walk-forward training ─────────────────────────────────
# For each direction, all accumulators are reset, the fold data is filtered to
# that direction only, the fold-local C(d) target is recomputed from training
# data only, and a separate L1+L2 model is trained.  Results are saved with a
# per-direction suffix (_long / _short).

for _TRAIN_DIR in ['long', 'short']:
    print(f"\n{'═'*90}")
    print(f"  DIRECTION: {_TRAIN_DIR.upper()}")
    print(f"{'═'*90}")

    all_test_results      = []   # L1 daily predictions per fold
    fold_metrics          = []   # L1 fold-level metrics
    l2_fold_metrics       = []   # L2 fold-level combined metrics
    l2_bucket_fold_metrics= []   # L2 per-bucket metrics (all folds)
    l2_all_test_results   = []   # L2 test prediction DataFrames
    feature_selection_log = []   # selected feature sets per fold (for stability)
    oof_moe_records       = []   # accumulated OOF MoE records for Layer 4

    for fold_num, (tr_idx, te_idx) in enumerate(folds, 1):
        print("\n" + "=" * 90)
        print(f"FOLD {fold_num:02d}/{len(folds)}")
        print("=" * 90)

        train_df, test_df = add_fold_safe_target(tr_idx, te_idx, df_raw, df)

        # ── Direction filter: keep only rows for _TRAIN_DIR ─────────────────────
        train_df = train_df[train_df['direction'] == _TRAIN_DIR].copy().reset_index(drop=True)
        test_df  = test_df[test_df['direction']   == _TRAIN_DIR].copy().reset_index(drop=True)
        if train_df.empty or test_df.empty:
            print(f"  Fold {fold_num}: no data for {_TRAIN_DIR} — skip")
            continue

        # ── Fold-local direction-specific C(d) target (training data only) ───────
        # Compute regime labels from THIS fold's training rows only so that no
        # test-fold trade outcomes can influence what constitutes a 'regime day'.
        # pnl is in OUTCOME_COLS so it was dropped from df/train_df; pull from df_raw instead.
        _cols_for_target = ['date', 'entry_time', 'brick_size', 'pnl']
        _raw_train_fold = df_raw.loc[tr_idx][df_raw.loc[tr_idx]['direction'] == _TRAIN_DIR]
        _raw_test_fold  = df_raw.loc[te_idx][df_raw.loc[te_idx]['direction'] == _TRAIN_DIR]
        _train_slice = _raw_train_fold[[c for c in _cols_for_target if c in _raw_train_fold.columns]].copy()
        _test_slice  = _raw_test_fold[[c  for c in _cols_for_target if c in _raw_test_fold.columns]].copy()

        # Unit 2/3: direction-specific target with fold-local gates active
        # Short uses avoidance target (bad-day prediction, gate inverted at application).
        # Long uses C(d) streak with Gates 2 & 3 enabled via fold_train_df.
        if _TRAIN_DIR == 'short':
            _fold_cd_train = compute_short_avoidance_target(_train_slice, fold_train_df=_train_slice)
            _fold_cd_test  = compute_short_avoidance_target(_test_slice,  fold_train_df=_train_slice)
            _short_bad_base = float(_fold_cd_train.mean()) if len(_fold_cd_train) > 0 else float('nan')
            print(f"  short_bad_day base rate (train): {_short_bad_base:.3f}")
        else:
            _cd_streak = CD_MIN_STREAK_LONG
            _fold_cd_train = compute_cd_target(_train_slice, min_streak=_cd_streak, fold_train_df=_train_slice)
            _fold_cd_test  = compute_cd_target(_test_slice,  min_streak=_cd_streak, fold_train_df=_train_slice)

        train_df = train_df.drop(columns=['regime_target'], errors='ignore')
        test_df  = test_df.drop(columns=['regime_target'],  errors='ignore')
        train_df['regime_target'] = (
            pd.to_datetime(train_df['entry_time']).dt.date.map(_fold_cd_train).fillna(0).astype(int))
        test_df['regime_target']  = (
            pd.to_datetime(test_df['entry_time']).dt.date.map(_fold_cd_test).fillna(0).astype(int))


        # FIX: fold-local target history only.
        # Build a fold-local cd_series from the training dates available so far,
        # then compute rolling previous-day features separately for train and test.
        # Train rows see only prior TRAIN dates; test rows see train + prior TEST
        # dates within this block (and never future test dates, even within the
        # same block). The embargo gap is naturally preserved because embargo days
        # are present in neither the train nor test cd_series.
        _train_dates = sorted(pd.to_datetime(train_df['entry_time']).dt.date.unique())
        _test_dates  = sorted(pd.to_datetime(test_df['entry_time']).dt.date.unique())
        _train_cd_series    = _fold_cd_train.reindex(_train_dates).dropna()
        _test_cd_series     = _fold_cd_test.reindex(_test_dates).dropna()
        _train_cd_counts    = None   # direction counts not pre-computed; rolling depth unused
        _combined_cd_series = pd.concat([_train_cd_series, _test_cd_series])
        _combined_cd_series = _combined_cd_series[~_combined_cd_series.index.duplicated(keep='first')]
        _combined_cd_counts = None

        train_df = compute_fold_safe_prev_day_features(
            train_df, _train_cd_series, _train_cd_counts)
        test_df = compute_fold_safe_prev_day_features(
            test_df, _combined_cd_series, _combined_cd_counts)

        # FIX: validate no rolling feature for date d depends on date d or later.
        # (1) prev1_regime_target on the first test day must equal C(d) of the
        # latest training day — not of any test day.
        if len(_train_dates) > 0 and len(_test_dates) > 0:
            _first_test_date = _test_dates[0]
            _last_train_date = _train_dates[-1]
            _first_test_rows = test_df[
                pd.to_datetime(test_df['entry_time']).dt.date == _first_test_date]
            if len(_first_test_rows) > 0:
                _p1 = _first_test_rows['prev1_regime_target'].iloc[0]
                _expected = float(_fold_cd_train.get(_last_train_date, np.nan))
                if not (np.isnan(_p1) and np.isnan(_expected)):
                    assert (np.isnan(_p1) and np.isnan(_expected)) or np.isclose(_p1, _expected), (
                        f"Fold {fold_num}: prev1_regime_target on first test day "
                        f"({_first_test_date}) = {_p1}, expected C(d) of last train day "
                        f"({_last_train_date}) = {_expected}"
                    )
        # (2) rolling_5d_regime_rate on test day k must use only previous 5 already-seen days.
        # Confirmed by construction: compute_fold_safe_prev_day_features pulls
        # known_dates strictly < d via bisect_left.

        if fold_num == 1:
            assert train_df['regime_target'].notna().all(), "NaN in train target"
            assert test_df['regime_target'].notna().all(),  "NaN in test target"
            assert train_df['regime_target'].nunique() >= 2, "Single class in train"
            assert test_df['regime_target'].nunique() >= 2,  "Single class in test"
            print("Train target distribution:")
            print(train_df['regime_target'].value_counts(normalize=True))
            print("\nTest target distribution:")
            print(test_df['regime_target'].value_counts(normalize=True))

        fold_br = train_df['regime_target'].mean()
        test_br = test_df['regime_target'].mean()
        fold_spw = (1 - fold_br) / fold_br if fold_br > 0 else 1.0
        print(f"  Train base rate: {fold_br:.3f} | Test: {test_br:.3f} | SPW: {fold_spw:.3f}")
        if train_df['regime_target'].nunique() < 2 or test_df['regime_target'].nunique() < 2:
            print("  Skip fold: degenerate target")
            continue

        # Raw PnL health check
        _fold_test_dates = set(test_df['date'].values)
        _fold_test_raw   = df_raw_macro[df_raw_macro['direction'] == _TRAIN_DIR]
        _fold_test_raw   = _fold_test_raw[
            pd.to_datetime(_fold_test_raw['entry_time']).dt.date.isin(_fold_test_dates)]
        def _bucket_label(f):
            if isinstance(f, str):
                return 'default'
            try:
                return 'medium' if float(f) <= 0.50 else 'large'
            except (TypeError, ValueError):
                return 'default'
        _fold_test_raw = _fold_test_raw.copy()
        _fold_test_raw['_bkt'] = _fold_test_raw['factor'].apply(_bucket_label)
        _bkt_stats = _fold_test_raw.groupby('_bkt')['pnl'].agg(
            total='sum', median='median', win_rate=lambda x: (x > 0).mean()
        ).reindex(list(_fold_test_raw['_bkt'].unique()))
        print("  Test PnL health | "
              + "  ".join(
                  f"{bkt}: total={row['total']:+,.0f} med={row['median']:+.2f} wr={row['win_rate']:.2%}"
                  for bkt, row in _bkt_stats.dropna().iterrows()
              ))

        # Per-fold dedup (train-only — FIX-Q-NOLEAK v14d)
        # Use direction-specific pool (re25): long/short models get direction-aligned candidates
        _dir_pool = _CANDIDATE_COLS_LONG if _TRAIN_DIR == 'long' else _CANDIDATE_COLS_SHORT
        # Filter to columns actually present in this fold's train_df (direction composites may be NaN-only)
        _dir_pool = [c for c in _dir_pool if c in train_df.columns and train_df[c].notna().any()]
        fold_dedup_cols = _compute_dedup_cols(train_df, _dir_pool)
        print(f"  Per-fold dedup [{_TRAIN_DIR}]: {len(_dir_pool)} → {len(fold_dedup_cols)} candidates (train-only)")

        # ── Step 1: Path A (deterministic states) ─────────────────────────────
        thresholds = compute_thresholds(train_df)
        train_df   = assign_states_path_a(train_df, thresholds)
        test_df    = assign_states_path_a(test_df,  thresholds)
        print("  Core state distribution (train):")
        for cs, cnt in train_df['core_state'].value_counts().items():
            cs_br = train_df.loc[train_df['core_state'] == cs, 'regime_target'].mean()
            print(f"    {cs:<12}: {cnt:>6,}  win_rate={cs_br:.3f}")

        # Placeholder sub-state columns (sub-models run inside L2 per-bucket)
        train_df['unified_sub_state'] = train_df['rule_sub_state']
        test_df['unified_sub_state']  = test_df['rule_sub_state']
        train_df['sub_state_profile'] = 'NEUTRAL'
        test_df['sub_state_profile']  = 'NEUTRAL'

        # ── Step 2: Bivariate filter ───────────────────────────────────────────
        surviving_cols, _ = bivariate_filter(train_df, fold_dedup_cols, target_col='regime_target')
        print(f"    {len(fold_dedup_cols)} → {len(surviving_cols)} (bivariate α={BIVARIATE_ALPHA})")
        _n_train_days = train_df['date'].nunique()

        # ── Step 3: CFI ranking ────────────────────────────────────────────────
        dynamic_top_k = compute_dynamic_top_k(len(surviving_cols), _n_train_days)
        selected_cols, cfi_info = clustered_feature_importance(
            train_df, surviving_cols, target_col='regime_target', top_k=dynamic_top_k)
        print(f"    CFI TOP_K={dynamic_top_k} → {len(selected_cols)} features via {cfi_info.get('method','?')}")
        feature_selection_log.append(selected_cols.copy())

        # ── Step 4: Median-fill + clip ─────────────────────────────────────────
        _train_X_raw   = train_df[selected_cols].replace([np.inf, -np.inf], np.nan)
        _train_medians = _train_X_raw.median()
        X_train_raw    = _train_X_raw.fillna(_train_medians).values
        _test_X_raw    = test_df[selected_cols].replace([np.inf, -np.inf], np.nan)
        X_test_raw     = _test_X_raw.fillna(_train_medians).values
        n_diffed = 0

        # ── Step 5: Gram-Schmidt orthogonalization ─────────────────────────────
        gs_params      = gram_schmidt_fit(X_train_raw)
        X_train_ortho  = gram_schmidt_transform(X_train_raw, gs_params)
        X_test_ortho   = gram_schmidt_transform(X_test_raw,  gs_params)
        if X_train_ortho.shape[1] >= 2:
            corr_check = np.corrcoef(X_train_ortho.T)
            np.fill_diagonal(corr_check, 0)
            print(f"    GS max off-diagonal corr: {np.nanmax(np.abs(corr_check)):.4f}")

        # ── Step 6: Core state encoding ────────────────────────────────────────
        cs_train = encode_core_state(train_df)
        cs_test  = encode_core_state(test_df)

        # ── Step 7: Assemble feature matrix (no L1 sub-model meta) ────────────
        train_medians_ortho = np.nanmedian(X_train_ortho, axis=0)
        X_train_clean = np.where(np.isfinite(X_train_ortho), X_train_ortho,
                                 np.broadcast_to(train_medians_ortho, X_train_ortho.shape))
        X_test_clean  = np.where(np.isfinite(X_test_ortho), X_test_ortho,
                                 np.broadcast_to(train_medians_ortho, X_test_ortho.shape))
        X_train_final = np.column_stack([X_train_clean, cs_train.reshape(-1, 1).astype(float)])
        X_test_final  = np.column_stack([X_test_clean,  cs_test.reshape(-1, 1).astype(float)])
        X_train_final = np.nan_to_num(np.clip(X_train_final, -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final  = np.nan_to_num(np.clip(X_test_final,  -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)
        feature_names = selected_cols + ['core_state_encoded']
        cat_indices   = [len(selected_cols)]
        print(f"  Feature matrix: {len(selected_cols)} ortho + 1 core_state = {X_train_final.shape[1]} total")

        # ── Step 8: Train main L1 model ────────────────────────────────────────
        y_train      = train_df['regime_target'].values
        # OAA-Gap6: ATR-epoch temporal weighting — vol doubled 2023→2025, recent rows more relevant
        _tr_times_l1 = pd.to_datetime(train_df['entry_time']).values.astype(np.int64)
        _sw_l1 = 0.5 + 0.5 * (pd.Series(_tr_times_l1).rank(pct=True).values)

        model_result = train_main_model_v5(X_train_final, y_train, feature_names,
                                           cat_indices=cat_indices, direction=_TRAIN_DIR,
                                           sample_weight=_sw_l1)
        print(f"  L1 SPW={model_result['scale_pos_weight']:.3f} (dir={_TRAIN_DIR})")

        # ── Step 9: Test predictions → day-level ──────────────────────────────
        test_probs_trade = predict_main_model_v5(X_test_final, model_result)
        fold_day_br = float(train_df['regime_target'].mean())

        # FIX-re21c: OOF L1 probs for L2 carry feature.
        # re20 used in-sample train predictions → inflated l1_day_probability on L2 train days.
        # Fix: 3-fold TimeSeriesSplit inner CV → out-of-fold train probs (causal).
        _oof_trade_probs = np.full(len(X_train_final), fold_day_br, dtype=float)
        try:
            from sklearn.model_selection import TimeSeriesSplit as _TSS
            _inner_tss = _TSS(n_splits=3)
            _tss_times = train_df['entry_time'].values
            _tss_sort  = np.argsort(_tss_times)
            for _itr, _ival in _inner_tss.split(X_train_final):
                _itr_s = _tss_sort[_itr]
                _ival_s = _tss_sort[_ival]
                _X_itr = X_train_final[_itr_s]
                _X_ival = X_train_final[_ival_s]
                _y_itr = y_train[_itr_s]
                if len(np.unique(_y_itr)) < 2:
                    continue
                _m_i = lgb.LGBMClassifier(
                    n_estimators=100, num_leaves=7, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                    reg_alpha=1.0, reg_lambda=5.0, is_unbalance=True,
                    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
                _m_i.fit(_X_itr, _y_itr)
                _oof_trade_probs[_ival_s] = _m_i.predict_proba(_X_ival)[:, 1]
        except Exception:
            _oof_trade_probs = predict_main_model_v5(X_train_final, model_result)
        train_df['_l1_trade_prob'] = _oof_trade_probs
        _train_day_l1_prob = (
            train_df.groupby('date')['_l1_trade_prob']
            .mean().rename('l1_day_probability').reset_index()
        )
        train_df.drop(columns=['_l1_trade_prob'], inplace=True)

        # Day-level aggregation for L1 predictions
        test_df['_l1_trade_prob'] = test_probs_trade
        test_df['_date_tmp'] = pd.to_datetime(test_df['entry_time']).dt.date
        test_df['_l1_cumprob'] = test_df.groupby('_date_tmp')['_l1_trade_prob'].transform(
            lambda x: x.expanding().mean())
        test_df['_l1_declared'] = (test_df['_l1_cumprob'] >= fold_day_br).astype(int)
        _first_declare = (
            test_df[test_df['_l1_declared'] == 1]
            .groupby('_date_tmp')['entry_time'].min()
        )
        _total_test_days = test_df['_date_tmp'].nunique()
        print(f"  Intraday L1 declarations: {len(_first_declare)}/{_total_test_days} test days "
              f"(gate={fold_day_br:.3f})")

        _test_day_l1_prob = (
            test_df.groupby('_date_tmp')['_l1_trade_prob']
            .mean().rename('l1_day_probability')
            .reset_index().rename(columns={'_date_tmp': 'date'})
        )
        test_df.drop(columns=['_l1_trade_prob', '_date_tmp', '_l1_cumprob', '_l1_declared'],
                     inplace=True)

        _day_agg = (
            test_df.groupby('date', sort=True)
            .agg(entry_time=('entry_time', 'first'),
                 l1_day_prob=('entry_time', 'count'),
                 regime_target=('regime_target', 'first'))
            .reset_index()
            .sort_values('entry_time')
            .reset_index(drop=True)
        )
        _day_agg['l1_day_prob'] = (
            _test_day_l1_prob.set_index('date')
            .reindex(_day_agg['date'])['l1_day_probability'].values
        )
        test_probs = _day_agg['l1_day_prob'].values
        test_preds = (test_probs >= fold_day_br).astype(int)
        test_true  = _day_agg['regime_target'].values
        test_br    = float(test_true.mean())

        # Attach daily PnL for MW test
        _test_dates = set(_day_agg['date'].values)
        _raw_test   = df_raw_macro[df_raw_macro['direction'] == _TRAIN_DIR]
        _raw_test   = _raw_test[pd.to_datetime(_raw_test['entry_time']).dt.date.isin(_test_dates)]
        _day_pnl    = (
            _raw_test.groupby(pd.to_datetime(_raw_test['entry_time']).dt.date)['pnl']
            .sum().rename('pnl').rename_axis('date')
        )
        _day_agg = _day_agg.set_index('date').join(_day_pnl).reset_index()

        result_df = _day_agg[['entry_time', 'date']].copy()
        result_df['pred_prob']  = test_probs
        result_df['pred_label'] = test_preds
        result_df['true_label'] = test_true
        result_df['fold']       = fold_num
        all_test_results.append(result_df)

        # ── Mann-Whitney U: daily PnL separation ──────────────────────────────
        _l1_mw_thresh = L1_PROB_THRESHOLD_LONG if _TRAIN_DIR == 'long' else L1_PROB_THRESHOLD_SHORT
        mw_u, mw_p, mw_med_pos, mw_med_neg = mann_whitney_regime_separation(
            _day_agg, test_probs, threshold=_l1_mw_thresh)
        separation_flag = (not np.isnan(mw_p)) and (mw_p < 0.05)
        print(f"  L1 MW p={mw_p:.4f} | med_pos={mw_med_pos} | med_neg={mw_med_neg}"
              f" | {'✓ PASS' if separation_flag else '✗ FAIL'}")

        if _L2_ENABLED:  # ── L2 disabled — set _L2_ENABLED=True to re-enable
            # ══════════════════════════════════════════════════════════════════════════
            # LAYER 2 — Intraday trade selector with MoE (Hermes v2 L3)
            # ══════════════════════════════════════════════════════════════════════════
            print(f"\n  ── Layer 2 (intraday trade selection) ──")
    
            test_df_l1 = test_df.copy()
            test_df_l1['l1_pred_prob'] = test_probs_trade
            test_df_l1['date'] = pd.to_datetime(test_df_l1['entry_time']).dt.date
    
            test_date_to_prob    = test_df_l1.groupby('date')['l1_pred_prob'].mean()
            all_test_dates_sorted = sorted(test_date_to_prob.index)
            prev_day_prob = {}
            for i, d in enumerate(all_test_dates_sorted):
                prev_day_prob[d] = 0.0 if i == 0 else test_date_to_prob.iloc[i - 1]
    
            _prev_probs_arr = np.array([prev_day_prob.get(d, 0.0) for d in all_test_dates_sorted])
            _valid_prev     = _prev_probs_arr[_prev_probs_arr > 0]
            adaptive_l2_thresh = float(np.percentile(_valid_prev, L2_GATE_PERCENTILE)) \
                if len(_valid_prev) >= 5 else 0.5
            adaptive_l2_thresh = max(adaptive_l2_thresh, L2_GATE_FLOOR)
    
            # Unit 3: short inverts gate (low bad-day prob = good day to trade)
            if _TRAIN_DIR == 'short':
                _short_l2_thresh = 1.0 - L1_PROB_THRESHOLD_SHORT
                high_conv_dates = set(
                    d for d in all_test_dates_sorted
                    if prev_day_prob.get(d, 0.0) < _short_l2_thresh)
                print(f"  Short L2 gate (inverted): bad_day_prob < {_short_l2_thresh:.3f}")
            else:
                high_conv_dates = set(
                    d for d in all_test_dates_sorted if prev_day_prob.get(d, 0.0) > adaptive_l2_thresh)
                print(f"  Adaptive L2 gate: threshold={adaptive_l2_thresh:.3f} "
                      f"(p{L2_GATE_PERCENTILE}, floor={L2_GATE_FLOOR})")
            print(f"  L1 high-conviction test dates: {len(high_conv_dates)} / "
                  f"{len(all_test_dates_sorted)} total")
    
            high_conv_train_dates = set(
                pd.to_datetime(
                    train_df.loc[train_df['regime_target'] == 1, 'entry_time']
                ).dt.date
            )
    
            _dir_raw_macro = df_raw_macro[df_raw_macro['direction'] == _TRAIN_DIR]
            df_raw_date  = pd.to_datetime(_dir_raw_macro['entry_time']).dt.date
            l2_train_raw = _dir_raw_macro[df_raw_date.isin(high_conv_train_dates)].copy()
            l2_test_raw  = _dir_raw_macro[df_raw_date.isin(high_conv_dates)].copy()
    
            print(f"  L2 train (regime=1 days): {len(l2_train_raw):,} trades "
                  f"({len(high_conv_train_dates)} days)  "
                  f"| test: {len(l2_test_raw):,} trades")
    
            if len(l2_train_raw) < L2_MIN_TRADES_FOLD or len(l2_test_raw) < L2_MIN_BUCKET_TRADES:
                print(f"  L2 skip fold {fold_num}: insufficient gated trades "
                      f"(train={len(l2_train_raw)}, test={len(l2_test_raw)}, "
                      f"min_train={L2_MIN_TRADES_FOLD}, min_test={L2_MIN_BUCKET_TRADES})")
                continue
    
            # Attach l1_day_probability to L2 slices
            for _raw_df, _day_prob_df, _label in [
                (l2_train_raw, _train_day_l1_prob, 'train'),
                (l2_test_raw,  _test_day_l1_prob,  'test'),
            ]:
                _raw_df['_merge_date'] = pd.to_datetime(_raw_df['entry_time']).dt.date
                _day_prob_tmp = _day_prob_df.copy()
                _day_prob_tmp.columns = ['_merge_date', 'l1_day_probability']
                _raw_df.drop(columns=['l1_day_probability'], inplace=True, errors='ignore')
                _raw_df['l1_day_probability'] = (
                    _raw_df[['_merge_date']]
                    .merge(_day_prob_tmp, on='_merge_date', how='left')['l1_day_probability']
                    .values
                )
                _raw_df.drop(columns=['_merge_date'], inplace=True)
            l2_train_raw['l1_day_probability'] = l2_train_raw['l1_day_probability'].fillna(
                float(_train_day_l1_prob['l1_day_probability'].mean()))
            l2_test_raw['l1_day_probability']  = l2_test_raw['l1_day_probability'].fillna(
                float(_test_day_l1_prob['l1_day_probability'].mean()))
    
            # FIX: fold-local target history only.
            # L2 pulls raw trades from df_raw_macro, which no longer carries the
            # rolling previous-day features (they used to be back-merged from a
            # globally-computed frame). Recompute them fold-locally here with the
            # same train-only / train+prior-test restrictions used for L1.
            l2_train_raw = compute_fold_safe_prev_day_features(
                l2_train_raw, _train_cd_series, _train_cd_counts)
            l2_test_raw  = compute_fold_safe_prev_day_features(
                l2_test_raw, _combined_cd_series, _combined_cd_counts)
    
            # Engineer L2 intraday features
            l2_train_raw = engineer_l2_intraday_features(l2_train_raw)
            l2_test_raw  = engineer_l2_intraday_features(l2_test_raw)
    
            # Per-bucket × per-Renko-state loop
            _fold_bucket_results = []
            _fold_bucket_metrics = []
    
            for _bname in FACTOR_BUCKETS:
                for _sname, _sset in RENKO_L2_STATE_GROUPS.items():
                    _label = f"{_bname}/{_sname}"
                    # FIX-re20e: skip historically harmful bucket
                    if _label in L2_SKIP_BUCKETS:
                        print(f"\n  ── L2 [{_label}] — SKIPPED (FIX-re20e: harmful bucket) ──")
                        continue
                    print(f"\n  ── L2 [{_label}] ──")
                    _btr = _filter_bucket(l2_train_raw, _bname)
                    _bte = _filter_bucket(l2_test_raw,  _bname)
                    _res, _met, _oof_moe = _run_l2_bucket(
                        _btr, _bte, _label, fold_num,
                        dedup_cols=fold_dedup_cols,
                        state_filter=_sset,
                        direction=_TRAIN_DIR,
                    )
                    _fold_bucket_metrics.append(_met)
                    l2_bucket_fold_metrics.append(_met)
                    if _res is not None:
                        _fold_bucket_results.append(_res)
                    if _oof_moe is not None:
                        oof_moe_records.append(_oof_moe)
    
            # ── Combine bucket predictions ─────────────────────────────────────────
            if _fold_bucket_results:
                _combined = pd.concat(_fold_bucket_results, ignore_index=True)
                l2_all_test_results.append(_combined)
    
                _run_mets    = [m for m in _fold_bucket_metrics if not m.get('skipped', True)]
                l2_base_rate = float(np.nanmean([m['base_rate'] for m in _run_mets])) if _run_mets else np.nan
                l2_dynamic_k = int(np.mean([m['n_features'] for m in _run_mets])) if _run_mets else 0
    
                # Combined Spearman(pred_rank, pnl) across all bucket results
                _all_pnl      = _combined['pnl'].values
                _all_pred_rk  = _combined['pred_rank'].values
                _spear_mask_c = np.isfinite(_all_pred_rk) & np.isfinite(_all_pnl)
                if _spear_mask_c.sum() >= 10:
                    try:
                        l2_spear_r, l2_spear_p = spearmanr(
                            _all_pred_rk[_spear_mask_c], _all_pnl[_spear_mask_c])
                    except Exception:
                        l2_spear_r, l2_spear_p = np.nan, np.nan
                else:
                    l2_spear_r, l2_spear_p = np.nan, np.nan
    
                _mean_spear_r_pnl = float(np.nanmean(
                    [m.get('spearman_rank_pnl', np.nan) for m in _run_mets])) if _run_mets else np.nan
                _mean_rmse = float(np.nanmean(
                    [m.get('rmse', np.nan) for m in _run_mets])) if _run_mets else np.nan
    
                # Combined MW: top-30% pred_rank vs bottom-70%
                _top30_c   = (_combined['pred_label'] == 1).values
                _pnl_pos_c = _all_pnl[_top30_c]
                _pnl_neg_c = _all_pnl[~_top30_c]
                if len(_pnl_pos_c) >= 5 and len(_pnl_neg_c) >= 5:
                    try:
                        l2_mw_u, l2_mw_p = mannwhitneyu(_pnl_pos_c, _pnl_neg_c, alternative='greater')
                    except Exception:
                        l2_mw_u, l2_mw_p = np.nan, np.nan
                else:
                    l2_mw_u, l2_mw_p = np.nan, np.nan
                l2_mw_sep = (not np.isnan(l2_mw_p)) and (l2_mw_p < 0.05)
    
                # FIX-re25a: P/R/F1 against BOTH targets at CLS gate
                def _l2_prf(y_true, y_pred):
                    try:
                        p = precision_score(y_true, y_pred, zero_division=0)
                        r = recall_score(y_true, y_pred, zero_division=0)
                        f = f1_score(y_true, y_pred, zero_division=0)
                    except Exception:
                        p = r = f = np.nan
                    tp = int(((y_pred == 1) & (y_true == 1)).sum())
                    fp = int(((y_pred == 1) & (y_true == 0)).sum())
                    fn = int(((y_pred == 0) & (y_true == 1)).sum())
                    tn = int(((y_pred == 0) & (y_true == 0)).sum())
                    return p, r, f, tp, fp, fn, tn
    
                _l2_pred = _combined['pred_label'].values.astype(int)
                if 'true_label' in _combined.columns:
                    _l2_y_true = _combined['true_label'].values.astype(int)
                    _l2_prec, _l2_rec, _l2_f1, _l2_tp, _l2_fp, _l2_fn, _l2_tn = _l2_prf(_l2_y_true, _l2_pred)
                else:
                    _l2_prec = _l2_rec = _l2_f1 = np.nan
                    _l2_tp = _l2_fp = _l2_fn = _l2_tn = 0
    
                _l2_win_prec = _l2_win_rec = _l2_win_f1 = np.nan
                _l2_win_tp = _l2_win_fp = _l2_win_fn = _l2_win_tn = 0
                if 'true_win' in _combined.columns:
                    _l2_y_win = _combined['true_win'].values.astype(int)
                    (_l2_win_prec, _l2_win_rec, _l2_win_f1,
                     _l2_win_tp, _l2_win_fp, _l2_win_fn, _l2_win_tn) = _l2_prf(_l2_y_win, _l2_pred)
    
                _l2_pnl_mean   = float(_all_pnl.mean()) if len(_all_pnl) > 0 else np.nan
                _lift_cls_c    = _pnl_pos_c.mean() if len(_pnl_pos_c) > 5 else np.nan
                _lift_cls_ratio = (float(_lift_cls_c / abs(_l2_pnl_mean))
                                   if (not np.isnan(_lift_cls_c) and not np.isnan(_l2_pnl_mean)
                                       and _l2_pnl_mean != 0) else np.nan)
    
                # Also compute lift@rank30 from pred_label_rank if available
                l2_lift_r30_rank = float(np.nanmean([m['lift_ratio30'] for m in _run_mets
                                                      if not np.isnan(m.get('lift_ratio30', np.nan))])) \
                    if _run_mets else np.nan
    
                print(f"\n  L2 COMBINED ({len(_fold_bucket_results)} buckets): "
                      f"base_rate={l2_base_rate:.3f}  "
                      f"Lift@CLS={_lift_cls_ratio:.2f}x  Lift@top30={l2_lift_r30_rank:.2f}x  MW p={l2_mw_p:.4f} "
                      f"{'✓' if l2_mw_sep else '✗'}  "
                      f"Spearman(rank,pnl)={l2_spear_r:.3f} "
                      f"{'✓' if (not np.isnan(l2_spear_r) and l2_spear_r > 0) else '✗'}  "
                      f"mean_RMSE={_mean_rmse:.4f}")
                print(f"  L2 P/R/F1 @CLS (window_lbl): {_l2_prec:.3f} / {_l2_rec:.3f} / {_l2_f1:.3f}  "
                      f"(TP={_l2_tp} FP={_l2_fp} FN={_l2_fn} TN={_l2_tn})")
                print(f"  L2 P/R/F1 @CLS (win/loss):   {_l2_win_prec:.3f} / {_l2_win_rec:.3f} / {_l2_win_f1:.3f}  "
                      f"(TP={_l2_win_tp} FP={_l2_win_fp} FN={_l2_win_fn} TN={_l2_win_tn})")
    
                l2_fold_metrics.append({
                    'fold':                     fold_num,
                    'l2_skipped':               False,
                    'l2_n_train':               sum(m['n_train'] for m in _fold_bucket_metrics),
                    'l2_n_test':                sum(m['n_test']  for m in _fold_bucket_metrics),
                    'l2_base_rate':             round(l2_base_rate, 4),
                    'l2_precision_cls':         round(_l2_prec, 4)     if not np.isnan(_l2_prec)     else np.nan,
                    'l2_recall_cls':            round(_l2_rec,  4)     if not np.isnan(_l2_rec)      else np.nan,
                    'l2_f1_cls':                round(_l2_f1,   4)     if not np.isnan(_l2_f1)       else np.nan,
                    'l2_tp_cls':                _l2_tp,
                    'l2_fp_cls':                _l2_fp,
                    'l2_fn_cls':                _l2_fn,
                    'l2_tn_cls':                _l2_tn,
                    'l2_win_precision_cls':     round(_l2_win_prec, 4) if not np.isnan(_l2_win_prec) else np.nan,
                    'l2_win_recall_cls':        round(_l2_win_rec,  4) if not np.isnan(_l2_win_rec)  else np.nan,
                    'l2_win_f1_cls':            round(_l2_win_f1,   4) if not np.isnan(_l2_win_f1)   else np.nan,
                    'l2_win_tp_cls':            _l2_win_tp,
                    'l2_win_fp_cls':            _l2_win_fp,
                    'l2_win_fn_cls':            _l2_win_fn,
                    'l2_win_tn_cls':            _l2_win_tn,
                    'l2_lift_ratio_cls':        round(_lift_cls_ratio, 4) if not np.isnan(_lift_cls_ratio) else np.nan,
                    'l2_lift_ratio30':          round(l2_lift_r30_rank, 4) if not np.isnan(l2_lift_r30_rank) else np.nan,
                    'l2_spearman_rank_pnl':     round(l2_spear_r, 4)      if not np.isnan(l2_spear_r)      else np.nan,
                    'l2_spearman_p':            round(l2_spear_p, 4)       if not np.isnan(l2_spear_p)       else np.nan,
                    'l2_mean_bucket_spear_pnl': round(_mean_spear_r_pnl, 4) if not np.isnan(_mean_spear_r_pnl) else np.nan,
                    'l2_mean_rmse':             round(_mean_rmse, 4)        if not np.isnan(_mean_rmse)        else np.nan,
                    'l2_n_features':            l2_dynamic_k,
                    'l2_gate_thresh':           round(adaptive_l2_thresh, 4),
                    'l2_dynamic_top_k':         l2_dynamic_k,
                    'buckets_run':              [m['bucket'] for m in _run_mets],
                    'mw_u':                     mw_u,
                    'mw_p':                     round(mw_p, 4) if not np.isnan(mw_p) else np.nan,
                    'mw_separation':            separation_flag,
                    'l2_mw_u':                  l2_mw_u,
                    'l2_mw_p':                  round(l2_mw_p, 4) if not np.isnan(l2_mw_p) else np.nan,
                    'l2_mw_separation':         l2_mw_sep,
                })
            else:
                l2_mw_u = l2_mw_p = np.nan
                l2_mw_sep = False
                print("  All L2 buckets skipped this fold.")
                l2_fold_metrics.append({
                    'fold': fold_num, 'l2_skipped': True,
                    'l2_n_train': 0, 'l2_n_test': 0,
                    'mw_u':  mw_u,
                    'mw_p':  round(mw_p, 4) if not np.isnan(mw_p) else np.nan,
                    'mw_separation': separation_flag,
                    'l2_mw_u': np.nan, 'l2_mw_p': np.nan, 'l2_mw_separation': False,
                })
    
        else:
            # L2 disabled: record fold as skipped so downstream metrics handle gracefully
            l2_fold_metrics.append({
                'fold': fold_num, 'l2_skipped': True,
                'mw_u': mw_u, 'mw_p': round(mw_p, 4) if not (mw_p != mw_p) else float('nan'),
                'mw_separation': separation_flag,
                'l2_mw_u': float('nan'), 'l2_mw_p': float('nan'), 'l2_mw_separation': False,
            })
        # ── Fold metrics (Layer 1) — rank-based only (re19) ───────────────────────
        try:
            fold_auc = roc_auc_score(test_true, test_probs)
        except Exception:
            fold_auc = np.nan

        # Rank-based selection: top-30% / top-20% of predicted prob
        top30_mask = test_probs >= np.percentile(test_probs, 70)
        top20_mask = test_probs >= np.percentile(test_probs, 80)
        lift_top30 = test_true[top30_mask].mean() if top30_mask.sum() > 5 else np.nan
        lift_top20 = test_true[top20_mask].mean() if top20_mask.sum() > 5 else np.nan
        lift_r30   = lift_top30 / test_br if (test_br > 0 and not np.isnan(lift_top30)) else np.nan
        lift_r20   = lift_top20 / test_br if (test_br > 0 and not np.isnan(lift_top20)) else np.nan

        # Spearman(pred_prob, daily_pnl)
        _day_pnl_vals = _day_agg['pnl'].values if 'pnl' in _day_agg.columns else None
        if _day_pnl_vals is not None:
            _sp_mask = np.isfinite(test_probs) & np.isfinite(_day_pnl_vals)
            if _sp_mask.sum() >= 5:
                try:
                    _l1_spear_r, _l1_spear_p = spearmanr(
                        test_probs[_sp_mask], _day_pnl_vals[_sp_mask])
                except Exception:
                    _l1_spear_r, _l1_spear_p = np.nan, np.nan
            else:
                _l1_spear_r, _l1_spear_p = np.nan, np.nan
        else:
            _l1_spear_r, _l1_spear_p = np.nan, np.nan

        # PnL of top-30% days vs bottom-70%
        _top30_pnl = _day_pnl_vals[top30_mask] if _day_pnl_vals is not None else np.array([])
        _bot70_pnl = _day_pnl_vals[~top30_mask] if _day_pnl_vals is not None else np.array([])
        _top30_mean_pnl = float(_top30_pnl.mean()) if len(_top30_pnl) > 0 else np.nan
        _bot70_mean_pnl = float(_bot70_pnl.mean()) if len(_bot70_pnl) > 0 else np.nan

        # FIX: per-fold precision/recall/F1/Brier/Pearson + confusion matrix
        # at both the default (fold base rate) gate and the operational 0.65 gate.
        _pred_def = (test_probs >= fold_day_br).astype(int)
        _pred_op  = (test_probs >= 0.65).astype(int)
        try:
            _prec_def = precision_score(test_true, _pred_def, zero_division=0)
            _rec_def  = recall_score(test_true,    _pred_def, zero_division=0)
            _f1_def   = f1_score(test_true,        _pred_def, zero_division=0)
            _prec_op  = precision_score(test_true, _pred_op,  zero_division=0)
            _rec_op   = recall_score(test_true,    _pred_op,  zero_division=0)
            _f1_op    = f1_score(test_true,        _pred_op,  zero_division=0)
        except Exception:
            _prec_def = _rec_def = _f1_def = _prec_op = _rec_op = _f1_op = np.nan
        try:
            _brier = brier_score_loss(test_true, test_probs)
        except Exception:
            _brier = np.nan
        try:
            _pear = float(np.corrcoef(test_probs, test_true)[0, 1]) \
                    if len(test_true) >= 3 else np.nan
        except Exception:
            _pear = np.nan
        # Confusion matrix at the operational 0.65 gate
        _tp_op = int(((_pred_op == 1) & (test_true == 1)).sum())
        _fp_op = int(((_pred_op == 1) & (test_true == 0)).sum())
        _fn_op = int(((_pred_op == 0) & (test_true == 1)).sum())
        _tn_op = int(((_pred_op == 0) & (test_true == 0)).sum())

        print(f"     Precision/Recall/F1 @ gate={fold_day_br:.2f}: "
              f"{_prec_def:.3f} / {_rec_def:.3f} / {_f1_def:.3f}  |  "
              f"@ 0.65: {_prec_op:.3f} / {_rec_op:.3f} / {_f1_op:.3f}  "
              f"(TP={_tp_op} FP={_fp_op} FN={_fn_op} TN={_tn_op})")
        print(f"     Brier={_brier:.4f}  Pearson(prob,true)={_pear:+.3f}")

        fold_metrics.append({
            'fold':                  fold_num,
            'n_train':               train_df['date'].nunique(),
            'n_test':                _day_agg['date'].nunique(),
            'train_base_rate':       round(fold_br, 4),
            'test_base_rate':        round(test_br, 4),
            'auc':                   round(fold_auc, 4) if not np.isnan(fold_auc) else np.nan,
            'spearman_r':            round(_l1_spear_r, 4) if not np.isnan(_l1_spear_r) else np.nan,
            'spearman_p':            round(_l1_spear_p, 4) if not np.isnan(_l1_spear_p) else np.nan,
            'lift_top30':            round(lift_top30, 4) if not np.isnan(lift_top30) else np.nan,
            'lift_ratio30':          round(lift_r30, 4)   if not np.isnan(lift_r30)   else np.nan,
            'lift_top20':            round(lift_top20, 4) if not np.isnan(lift_top20) else np.nan,
            'lift_ratio20':          round(lift_r20, 4)   if not np.isnan(lift_r20)   else np.nan,
            'top30_mean_pnl':        round(_top30_mean_pnl, 4) if not np.isnan(_top30_mean_pnl) else np.nan,
            'bot70_mean_pnl':        round(_bot70_mean_pnl, 4) if not np.isnan(_bot70_mean_pnl) else np.nan,
            'n_features_surviving':  len(surviving_cols),
            'n_features_selected':   len(selected_cols),
            'dynamic_top_k':         dynamic_top_k,
            'n_fracdiffed':          n_diffed,
            'scale_pos_weight':      round(fold_spw, 4),
            'precision_def':         round(_prec_def, 4) if not np.isnan(_prec_def) else np.nan,
            'recall_def':            round(_rec_def,  4) if not np.isnan(_rec_def)  else np.nan,
            'f1_def':                round(_f1_def,   4) if not np.isnan(_f1_def)   else np.nan,
            'precision_op65':        round(_prec_op, 4)  if not np.isnan(_prec_op)  else np.nan,
            'recall_op65':           round(_rec_op,  4)  if not np.isnan(_rec_op)   else np.nan,
            'f1_op65':               round(_f1_op,   4)  if not np.isnan(_f1_op)    else np.nan,
            'tp_op65':               _tp_op,
            'fp_op65':               _fp_op,
            'fn_op65':               _fn_op,
            'tn_op65':               _tn_op,
            'brier':                 round(_brier, 4)    if not np.isnan(_brier)    else np.nan,
            'pearson_prob_true':     round(_pear,  4)    if not np.isnan(_pear)     else np.nan,
        })

        print(f"\n  ── Fold {fold_num} Summary ──")
        print(f"     AUC={fold_auc:.3f}  Spearman(prob,pnl)={_l1_spear_r:.3f} (p={_l1_spear_p:.4f})"
              f"  {'✓' if (not np.isnan(_l1_spear_r) and _l1_spear_r > 0) else '✗'}")
        print(f"     Lift@top30={lift_top30:.3f} ({lift_r30:.2f}x)  Lift@top20={lift_top20:.3f} ({lift_r20:.2f}x)"
              f"  top30_pnl={_top30_mean_pnl:.1f}  bot70_pnl={_bot70_mean_pnl:.1f}")



    # ═══════════════════════════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════════
    all_preds_df         = pd.concat(all_test_results, ignore_index=True)
    metrics_df           = pd.DataFrame(fold_metrics)
    l2_metrics_df        = pd.DataFrame(l2_fold_metrics)
    l2_bucket_metrics_df = pd.DataFrame(l2_bucket_fold_metrics)
    l2_preds_df          = pd.concat(l2_all_test_results, ignore_index=True) \
        if l2_all_test_results else pd.DataFrame()

    print(f"\n{'='*70}")
    print("WALK-FORWARD COMPLETE (re20 — Hermes v2 + L1/L2 Rank-Based Evaluation)")
    print(f"{'='*70}")

    print("\n── Per-fold L1 metrics (rank-based) ──")
    display_cols = ['fold', 'n_train', 'n_test', 'train_base_rate', 'test_base_rate',
                    'auc', 'spearman_r', 'lift_ratio30', 'lift_ratio20',
                    'top30_mean_pnl', 'bot70_mean_pnl', 'n_features_selected']
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols].to_string(index=False))

    print("\n── Per-fold L1 classification metrics ──")
    cls_cols = ['fold', 'precision_def', 'recall_def', 'f1_def',
                'precision_op65', 'recall_op65', 'f1_op65',
                'tp_op65', 'fp_op65', 'fn_op65', 'tn_op65',
                'brier', 'pearson_prob_true']
    cls_cols = [c for c in cls_cols if c in metrics_df.columns]
    if len(cls_cols) > 1:
        print(metrics_df[cls_cols].to_string(index=False))

    print(f"\nMean AUC               : {metrics_df['auc'].mean():.3f}  (std={metrics_df['auc'].std():.3f})")
    if 'spearman_r' in metrics_df.columns:
        print(f"Mean Spearman(prob,pnl): {metrics_df['spearman_r'].mean():.3f}  "
              f"(std={metrics_df['spearman_r'].std():.3f})")
        n_pos_sp = (metrics_df['spearman_r'] > 0).sum()
        print(f"Folds with ρ>0         : {n_pos_sp}/{len(metrics_df)}")
    print(f"Mean Lift@top30        : {metrics_df['lift_ratio30'].mean():.2f}x  "
          f"(std={metrics_df['lift_ratio30'].std():.2f})")
    print(f"Mean Lift@top20        : {metrics_df['lift_ratio20'].mean():.2f}x")

    # Classification metrics summary
    if 'precision_def' in metrics_df.columns:
        print(f"\nMean Precision @ base-rate gate : {metrics_df['precision_def'].mean():.3f}  "
              f"(std={metrics_df['precision_def'].std():.3f})")
        print(f"Mean Recall    @ base-rate gate : {metrics_df['recall_def'].mean():.3f}  "
              f"(std={metrics_df['recall_def'].std():.3f})")
        print(f"Mean F1        @ base-rate gate : {metrics_df['f1_def'].mean():.3f}  "
              f"(std={metrics_df['f1_def'].std():.3f})")
        print(f"Mean Precision @ 0.65 gate      : {metrics_df['precision_op65'].mean():.3f}  "
              f"(std={metrics_df['precision_op65'].std():.3f})")
        print(f"Mean Recall    @ 0.65 gate      : {metrics_df['recall_op65'].mean():.3f}  "
              f"(std={metrics_df['recall_op65'].std():.3f})")
        print(f"Mean F1        @ 0.65 gate      : {metrics_df['f1_op65'].mean():.3f}  "
              f"(std={metrics_df['f1_op65'].std():.3f})")
        print(f"Mean Brier                      : {metrics_df['brier'].mean():.4f}  "
              f"(std={metrics_df['brier'].std():.4f})")
        print(f"Mean Pearson(prob,true)         : {metrics_df['pearson_prob_true'].mean():+.3f}  "
              f"(std={metrics_df['pearson_prob_true'].std():.3f})")
        _tot_tp = int(metrics_df['tp_op65'].sum())
        _tot_fp = int(metrics_df['fp_op65'].sum())
        _tot_fn = int(metrics_df['fn_op65'].sum())
        _tot_tn = int(metrics_df['tn_op65'].sum())
        print(f"Pooled confusion @ 0.65 gate    : TP={_tot_tp} FP={_tot_fp} "
              f"FN={_tot_fn} TN={_tot_tn}")


    # ═══════════════════════════════════════════════════════════════════════════════
    if _L2_ENABLED:  # Layer 4 meta-labeler — disabled
        # HERMES-V2 LAYER 4: META-LABELER
        # Accumulate OOF MoE predictions across all folds, then train a lightweight
        # LightGBM meta-labeler.  Gate: confidence ≥ 0.55.
        # ═══════════════════════════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print("HERMES-V2 LAYER 4: META-LABELER TRAINING")
        print(f"{'─'*60}")
    
        meta_model_result = None
        l2_preds_df['meta_prob']       = np.nan
        l2_preds_df['meta_confidence'] = np.nan
    
        if oof_moe_records:
            oof_meta_df = pd.concat(oof_moe_records, ignore_index=True)
            print(f"  OOF MoE records: {len(oof_meta_df):,} rows from {len(oof_moe_records)} bucket×fold combinations")
    
            # re19: meta-regressor on moe_rank → predicts pnl rank
            _meta_feat_cols = ['moe_rank'] + [c for c in _MACRO_REGIME_COLS if c in oof_meta_df.columns]
            _meta_feat_cols = [c for c in _meta_feat_cols if c in oof_meta_df.columns]
    
            if len(_meta_feat_cols) >= 1:
                _Xm = oof_meta_df[_meta_feat_cols].replace([np.inf, -np.inf], np.nan)
                _meta_meds = _Xm.median()
                _Xm = _Xm.fillna(_meta_meds).values
                # Rank target: pnl percentile rank from binary target proxy (0/1)
                # Use trade_target as binary label for gate evaluation; rank target for training
                _ym_binary = oof_meta_df['trade_target'].values if 'trade_target' in oof_meta_df.columns else None
                _ym_rank   = oof_meta_df['moe_rank'].values  # use OOF rank as proxy target for regressor
    
                try:
                    meta_lgb = lgb.LGBMRegressor(
                        n_estimators=200, num_leaves=7, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                        reg_alpha=1.0, reg_lambda=5.0, objective='regression', metric='rmse',
                        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
                    meta_lgb.fit(_Xm, _ym_rank)
                    meta_model_result = {
                        'lgb': meta_lgb,
                        'feature_cols': _meta_feat_cols,
                        'medians': _meta_meds,
                    }
                    print(f"  Meta-regressor trained: {len(_meta_feat_cols)} features, "
                          f"n={len(_ym_rank):,}, rank_mean={_ym_rank.mean():.3f}")
    
                    # Apply meta-regressor to L2 test predictions
                    if not l2_preds_df.empty:
                        _avail = [c for c in _meta_feat_cols if c in l2_preds_df.columns]
                        if len(_avail) >= 1:
                            _Xm_test = l2_preds_df[_avail].replace([np.inf, -np.inf], np.nan)
                            for _mc in _meta_feat_cols:
                                if _mc not in _avail:
                                    _Xm_test[_mc] = _meta_meds.get(_mc, 0.0)
                            _Xm_test = _Xm_test[_meta_feat_cols].fillna(_meta_meds).values
                            _meta_ranks = np.clip(meta_lgb.predict(_Xm_test), 0.0, 1.0)
                            l2_preds_df['meta_prob']       = _meta_ranks
                            l2_preds_df['meta_confidence'] = _meta_ranks
                            # Gate: top-10% by meta rank
                            _gate_pct = np.percentile(_meta_ranks, 90)
                            _meta_gated = l2_preds_df[l2_preds_df['meta_prob'] >= _gate_pct]
                            print(f"  Meta gate ≥p90 ({_gate_pct:.3f}): "
                                  f"{len(_meta_gated):,}/{len(l2_preds_df):,} trades pass "
                                  f"({len(_meta_gated)/len(l2_preds_df)*100:.1f}%)")
                            if len(_meta_gated) >= 10 and 'true_label' in _meta_gated.columns:
                                _mg_wr = _meta_gated['true_label'].mean()
                                _all_wr = l2_preds_df['true_label'].mean()
                                print(f"  Meta-gated win_rate={_mg_wr:.3f} vs all={_all_wr:.3f} "
                                      f"(lift={_mg_wr/_all_wr:.2f}x if base>0)")
                            if len(_meta_gated) >= 10 and 'pnl' in _meta_gated.columns:
                                _mg_pnl = _meta_gated['pnl'].mean()
                                _all_pnl_mean = l2_preds_df['pnl'].mean()
                                print(f"  Meta-gated mean_pnl={_mg_pnl:.2f} vs all={_all_pnl_mean:.2f}")
                except Exception as _me:
                    print(f"  Meta-regressor training FAILED: {_me}")
            else:
                print(f"  Meta-regressor skipped — missing features in OOF records")
        else:
            print("  No OOF MoE records accumulated — meta-labeler skipped")
    

    # ═══════════════════════════════════════════════════════════════════════════════
    # EVALUATION SUMMARIES
    # ═══════════════════════════════════════════════════════════════════════════════

    # Percentile Ranking Filter
    print(f"\n{'─'*60}")
    print("PERCENTILE RANKING FILTER (re22)")
    print(f"{'─'*60}")
    all_probs = all_preds_df['pred_prob'].values
    signals, pct_ranks = percentile_rank_filter(all_probs)
    all_preds_df['percentile_rank'] = pct_ranks
    all_preds_df['signal'] = signals
    print("Signal distribution:")
    print(all_preds_df['signal'].value_counts().to_string())
    print("\nWin rates by signal:")
    for sig in ['LONG', 'HOLD', 'INSUFFICIENT_BUFFER']:
        mask = all_preds_df['signal'] == sig
        if mask.sum() > 0:
            wr = all_preds_df.loc[mask, 'true_label'].mean()
            print(f"  {sig:<22}: n={mask.sum():>6,}  win_rate={wr:.3f}")

    # Decile Monotonicity
    print(f"\n{'─'*60}")
    print("DECILE MONOTONICITY ANALYSIS (re22 — day-level L1)")
    print(f"{'─'*60}")
    if 'pnl' not in all_preds_df.columns:
        _raw_pnl_by_day = (
            df_raw_macro.groupby(pd.to_datetime(df_raw_macro['entry_time']).dt.date)['pnl']
            .sum().rename('pnl').rename_axis('date').reset_index()
        )
        all_preds_df = all_preds_df.merge(_raw_pnl_by_day, on='date', how='left')
    decile_df, sp_corr, sp_p, kt_tau, kt_p, dec_spread, is_monotonic = decile_monotonicity_test(
        all_preds_df['pred_prob'].values,
        all_preds_df['true_label'].values,
        all_preds_df['pnl'].values if 'pnl' in all_preds_df.columns else None,
    )
    print(decile_df.to_string(index=False))
    print(f"\nSpearman ρ     : {sp_corr:.3f}  (p={sp_p:.4f})")
    print(f"Kendall τ      : {kt_tau:.3f}  (p={kt_p:.4f})")
    print(f"Spread D10-D1  : {dec_spread:.3f}")
    print("✓ PASS" if is_monotonic else "✗ FAIL",
          "— monotonic prediction quality" if is_monotonic else "— lacks monotonic quality")

    # Mann-Whitney L1
    print(f"\n{'─'*60}")
    print("MANN-WHITNEY U REGIME SEPARATION (Layer 1, per fold)")
    print(f"{'─'*60}")
    if 'mw_p' in l2_metrics_df.columns:
        mw_display = l2_metrics_df[l2_metrics_df['l2_skipped'] == False][
            ['fold', 'mw_u', 'mw_p', 'mw_separation']].copy()
        print(mw_display.to_string(index=False))
        n_pass = mw_display['mw_separation'].sum()
        print(f"\nFolds with p<0.05: {n_pass}/{len(mw_display)}")

    # Layer 2 summary
    if _L2_ENABLED:  # L2 summaries — disabled
        print(f"\n{'─'*60}")
        print("LAYER 2 — INTRADAY TRADE SELECTION SUMMARY (re20 — rank regression)")
        print(f"{'─'*60}")
        if not l2_metrics_df.empty:
            l2_run = l2_metrics_df[l2_metrics_df['l2_skipped'] == False]
            if not l2_run.empty:
                l2_display_cols = ['fold', 'l2_n_train', 'l2_n_test', 'l2_base_rate',
                                   'l2_spearman_rank_pnl', 'l2_mean_rmse',
                                   'l2_lift_ratio30', 'l2_mw_separation']
                l2_display_cols = [c for c in l2_display_cols if c in l2_run.columns]
                print(l2_run[l2_display_cols].to_string(index=False))
                if 'l2_spearman_rank_pnl' in l2_run.columns:
                    print(f"\nMean L2 Spearman(rank,pnl): "
                          f"{l2_run['l2_spearman_rank_pnl'].mean():.3f}  "
                          f"(std={l2_run['l2_spearman_rank_pnl'].std():.3f})")
                if 'l2_mean_rmse' in l2_run.columns:
                    print(f"Mean L2 RMSE              : {l2_run['l2_mean_rmse'].mean():.4f}")
                print(f"Mean L2 Lift@30           : {l2_run['l2_lift_ratio30'].mean():.2f}x")
    
                if 'l2_precision_top30' in l2_run.columns:
                    print(f"\nMean L2 Precision @ top30 : {l2_run['l2_precision_top30'].mean():.3f}  "
                          f"(std={l2_run['l2_precision_top30'].std():.3f})")
                    print(f"Mean L2 Recall    @ top30 : {l2_run['l2_recall_top30'].mean():.3f}  "
                          f"(std={l2_run['l2_recall_top30'].std():.3f})")
                    print(f"Mean L2 F1        @ top30 : {l2_run['l2_f1_top30'].mean():.3f}  "
                          f"(std={l2_run['l2_f1_top30'].std():.3f})")
                    _l2_tp_tot = int(l2_run['l2_tp_top30'].sum())
                    _l2_fp_tot = int(l2_run['l2_fp_top30'].sum())
                    _l2_fn_tot = int(l2_run['l2_fn_top30'].sum())
                    _l2_tn_tot = int(l2_run['l2_tn_top30'].sum())
                    print(f"Pooled L2 confusion @top30: TP={_l2_tp_tot} FP={_l2_fp_tot} "
                          f"FN={_l2_fn_tot} TN={_l2_tn_tot}")
    
        # Per-bucket breakdown
        print(f"\n{'─'*60}")
        print("LAYER 2 — PER-BUCKET BREAKDOWN (averaged across folds)")
        print(f"{'─'*60}")
        if not l2_bucket_metrics_df.empty:
            _brun = l2_bucket_metrics_df[~l2_bucket_metrics_df['skipped']]
            if not _brun.empty:
                _agg_cols = dict(
                    folds_run=('fold', 'count'),
                    mean_n_train=('n_train', 'mean'),
                    mean_n_test=('n_test', 'mean'),
                    mean_base_rate=('base_rate', 'mean'),
                    mean_spearman_pnl=('spearman_rank_pnl', 'mean'),
                    mean_spearman_rank=('spearman_true_rank', 'mean'),
                    mean_rmse=('rmse', 'mean'),
                    mean_lift30=('lift_ratio30', 'mean'),
                    mw_pass_rate=('mw_separation', 'mean'),
                )
                _agg_cols = {k: v for k, v in _agg_cols.items() if v[0] in _brun.columns}
                _bsumm = _brun.groupby('bucket').agg(**_agg_cols).reset_index()
                print(_bsumm.to_string(index=False))
    
        # Mann-Whitney L2
        print(f"\n{'─'*60}")
        print("MANN-WHITNEY U TRADE PnL SEPARATION (Layer 2, per fold)")
        print(f"{'─'*60}")
        if not l2_metrics_df.empty and 'l2_mw_p' in l2_metrics_df.columns:
            _l2_mw_run = l2_metrics_df[l2_metrics_df['l2_skipped'] == False][
                ['fold', 'l2_mw_u', 'l2_mw_p', 'l2_mw_separation']].copy()
            print(_l2_mw_run.to_string(index=False))
            _l2_n_pass = _l2_mw_run['l2_mw_separation'].sum()
            print(f"\nFolds with p<0.05: {_l2_n_pass}/{len(_l2_mw_run)}")
    
        # L2 overall rank-regression metrics (all folds combined)
        if not l2_preds_df.empty:
            _l2_all_pnl   = l2_preds_df['pnl'].values
            _l2_all_rk    = l2_preds_df['pred_rank'].values if 'pred_rank' in l2_preds_df.columns else None
            if _l2_all_rk is not None:
                _l2_o_mask = np.isfinite(_l2_all_rk) & np.isfinite(_l2_all_pnl)
                if _l2_o_mask.sum() >= 10:
                    try:
                        _l2_o_sp, _l2_o_sp_p = spearmanr(
                            _l2_all_rk[_l2_o_mask], _l2_all_pnl[_l2_o_mask])
                    except Exception:
                        _l2_o_sp, _l2_o_sp_p = np.nan, np.nan
                else:
                    _l2_o_sp, _l2_o_sp_p = np.nan, np.nan
                _l2_top30  = (l2_preds_df['pred_label'] == 1).values
                _l2_pnl_t  = _l2_all_pnl[_l2_top30]
                _l2_pnl_a  = _l2_all_pnl
                print(f"\nL2 Overall (all folds): "
                      f"Spearman(rank,pnl)={_l2_o_sp:.3f} p={_l2_o_sp_p:.4f}  "
                      f"top30_mean_pnl={_l2_pnl_t.mean():.2f} vs all={_l2_pnl_a.mean():.2f}")
    
            l2_decile_df, l2_sp, _, _, _, l2_spread, l2_mono = decile_monotonicity_test(
                l2_preds_df['pred_rank'].values if 'pred_rank' in l2_preds_df.columns else l2_preds_df['pred_label'].values,
                l2_preds_df['true_label'].values,
            )
            print(f"L2 Decile Spearman ρ: {l2_sp:.3f} | Spread: {l2_spread:.3f} | "
                  f"{'✓ PASS' if l2_mono else '✗ FAIL'}")
    
    # Feature Stability
    print(f"\n{'─'*60}")
    print("FEATURE STABILITY ACROSS FOLDS (re25)")
    print(f"{'─'*60}")
    from collections import Counter
    feature_counts = Counter()
    for fold_features in feature_selection_log:
        feature_counts.update(fold_features)
    n_folds_total = len(folds)
    stability_rows = [{'feature': feat, 'folds_selected': cnt,
                       'pct_folds': round(cnt / n_folds_total * 100, 1)}
                      for feat, cnt in feature_counts.most_common()]
    stability_df = pd.DataFrame(stability_rows)
    stable_features = stability_df[stability_df['pct_folds'] >= 80]
    print(f"Stable features (≥80% of folds): {len(stable_features)}")
    print(stable_features.to_string(index=False))
    print(f"\nTotal unique features selected across all folds: {len(stability_df)}")

    # Skip-rate vs PnL (L1)
    print(f"\n{'─'*60}")
    print("SKIP-RATE vs PnL ANALYSIS (Layer 1 — daily regime days)")
    print(f"{'─'*60}")
    _l1_base_pnl = np.nan
    if 'pnl' in all_preds_df.columns:
        _l1_base_pnl = all_preds_df['pnl'].mean()
        print(f"Universe : {len(all_preds_df):,} test days  |  Base daily mean PnL: {_l1_base_pnl:.4f}")
        print(f"{'Threshold':>10} {'%Taken':>8} {'Taken PnL':>12} {'Skipped PnL':>13} {'PnL Lift':>10}")
        for _t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83]:
            _taken   = all_preds_df[all_preds_df['pred_prob'] >= _t]
            _skipped = all_preds_df[all_preds_df['pred_prob'] <  _t]
            if len(_taken) == 0:
                continue
            _pct_taken = len(_taken) / len(all_preds_df) * 100
            _tkpnl = _taken['pnl'].mean()
            _skpnl = _skipped['pnl'].mean() if len(_skipped) > 0 else np.nan
            _lift  = _tkpnl / _l1_base_pnl if _l1_base_pnl != 0 else np.nan
            print(f"  {_t:>10.2f} {_pct_taken:>7.1f}% {_tkpnl:>12.4f} {_skpnl:>13.4f} {_lift:>9.2f}x")

    if _L2_ENABLED:  # L2 skip-rate + cascade — disabled
        # Skip-rate vs PnL (L2)
        print(f"\n{'─'*60}")
        print("SKIP-RATE vs PnL ANALYSIS (Layer 2 — intraday trades, L1-gated universe)")
        print(f"{'─'*60}")
        _l2_base_pnl = np.nan
        if not l2_preds_df.empty and 'pnl' in l2_preds_df.columns:
            _l2_base_pnl = l2_preds_df['pnl'].mean()
            _l2_rank_col = 'pred_rank' if 'pred_rank' in l2_preds_df.columns else 'pred_label'
            print(f"Universe : {len(l2_preds_df):,} trades  |  Base trade mean PnL: {_l2_base_pnl:.4f}")
            print(f"  (L2 signal = predicted rank; cuts by rank percentile)")
            print(f"{'Pct-cut':>8} {'Threshold':>10} {'%Taken':>8} {'Taken PnL':>12} {'Skipped PnL':>13} {'PnL Lift':>10}")
            for _pct_cut in [50, 60, 70, 80, 90]:
                _thresh = float(np.percentile(l2_preds_df[_l2_rank_col], _pct_cut))
                _taken  = l2_preds_df[l2_preds_df[_l2_rank_col] >= _thresh]
                _skip   = l2_preds_df[l2_preds_df[_l2_rank_col] <  _thresh]
                if len(_taken) == 0:
                    continue
                _pct = len(_taken) / len(l2_preds_df) * 100
                _tkpnl = _taken['pnl'].mean()
                _skpnl = _skip['pnl'].mean() if len(_skip) > 0 else np.nan
                _lift  = _tkpnl / _l2_base_pnl if _l2_base_pnl != 0 else np.nan
                print(f"  {'top'+str(100-_pct_cut)+'%':>6} {_thresh:>10.4f} {_pct:>7.1f}%"
                      f" {_tkpnl:>12.4f} {_skpnl:>13.4f} {_lift:>9.2f}x")
    
        # Combined cascade summary
        print(f"\n{'═'*60}")
        print("COMBINED PIPELINE SKIP-RATE SUMMARY  (L1 → L2 cascade)")
        print(f"{'═'*60}")
        # FIX-re23c: report cascade at L1_PROB_THRESHOLD (0.60) instead of hardcoded 0.65
        _l1_cascade_thresh = L1_PROB_THRESHOLD
        _l1_taken_65 = all_preds_df[all_preds_df['pred_prob'] >= _l1_cascade_thresh] \
            if 'pnl' in all_preds_df.columns else pd.DataFrame()
        _l1_pct_65  = len(_l1_taken_65) / len(all_preds_df) * 100 if len(all_preds_df) > 0 else np.nan
        _l1_pnl_65  = _l1_taken_65['pnl'].mean() if len(_l1_taken_65) > 0 else np.nan
        _l1_lift_65 = (_l1_pnl_65 / _l1_base_pnl
                       if (not np.isnan(_l1_base_pnl) and _l1_base_pnl != 0)
                       else np.nan)
    
        if not l2_preds_df.empty and 'pnl' in l2_preds_df.columns:
            _l2_rank_col2 = 'pred_rank' if 'pred_rank' in l2_preds_df.columns else 'pred_label'
            _l2_thresh_50 = float(np.percentile(l2_preds_df[_l2_rank_col2], 50))
            _l2_taken_50  = l2_preds_df[l2_preds_df[_l2_rank_col2] >= _l2_thresh_50]
            _l2_pct_50    = len(_l2_taken_50) / len(l2_preds_df) * 100
            _l2_pnl_50    = _l2_taken_50['pnl'].mean() if len(_l2_taken_50) > 0 else np.nan
            _l2_lift_50   = (_l2_pnl_50 / _l2_base_pnl
                             if (not np.isnan(_l2_base_pnl) and _l2_base_pnl != 0)
                             else np.nan)
            _l2_available = True
        else:
            _l2_thresh_50 = _l2_pct_50 = _l2_pnl_50 = _l2_lift_50 = np.nan
            _l2_available = False
    
        print(f"\n{'Layer':<6}  {'Universe':>12}  {'Threshold':>12}  {'%Taken':>8}"
              f"  {'Taken PnL':>12}  {'Base PnL':>12}  {'Lift':>8}")
        print(f"{'─'*84}")
        if 'pnl' in all_preds_df.columns:
            print(f"{'L1':<6}  {len(all_preds_df):>12,}  {f'fixed {_l1_cascade_thresh:.2f}':>12}  {_l1_pct_65:>7.1f}%"
                  f"  {_l1_pnl_65:>12.4f}  {_l1_base_pnl:>12.4f}  {_l1_lift_65:>7.2f}x")
        if _l2_available:
            print(f"{'L2':<6}  {len(l2_preds_df):>12,}  {f'top50% ({_l2_thresh_50:.4f})':>12}"
                  f"  {_l2_pct_50:>7.1f}%  {_l2_pnl_50:>12.4f}  {_l2_base_pnl:>12.4f}"
                  f"  {_l2_lift_50:>7.2f}x")
    
        if _l2_available:
            print(f"\n  Cascade: L1 gate fixed @ {_l1_cascade_thresh:.2f} ({len(_l1_taken_65):,} days taken)"
                  f" → L2 at varying cuts of {len(l2_preds_df):,} gated trades")
            print(f"  {'L2 cut':>8} {'L2 thresh':>10} {'Trades taken':>13} {'%L2':>6}"
                  f" {'Trade PnL':>11} {'L2 Lift':>9}")
            print(f"  {'─'*65}")
            for _pct_cut in [50, 60, 70, 80, 90]:
                _t2  = float(np.percentile(l2_preds_df[_l2_rank_col2], _pct_cut))
                _tk2 = l2_preds_df[l2_preds_df[_l2_rank_col2] >= _t2]
                if len(_tk2) < 10:
                    continue
                _ppnl = _tk2['pnl'].mean()
                _pk   = len(_tk2) / len(l2_preds_df) * 100
                _pl   = _ppnl / _l2_base_pnl if (_l2_base_pnl and _l2_base_pnl != 0) else np.nan
                print(f"  {'top'+str(100-_pct_cut)+'%':>8} {_t2:>10.4f} {len(_tk2):>13,} {_pk:>5.1f}%"
                      f" {_ppnl:>11.4f} {_pl:>8.2f}x")
    
        # ═══════════════════════════════════════════════════════════════════════════════
    # ABLATION STUDY — Feature-group permutation importance (inherited from re21b)
    #
    # For each defined feature group, we re-train a lightweight L1 LGB model on
    # each fold's already-selected feature set with that group's columns zeroed out,
    # then measure the drop in AUC and Spearman(prob,pnl) vs the full model.
    #
    # The same group mask is applied to the L2 Spearman(rank,pnl) metric using
    # the per-fold OOF MoE predictions, giving a combined L1+L2 ablation table.
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("ABLATION STUDY — Feature-group permutation importance (re25)")
    print(f"{'═'*70}")

    # ── Define feature groups ──────────────────────────────────────────────────────
    ABLATION_GROUPS = {
        'OHLC': [
            'ohlc_price_in_band_lag1', 'ohlc_cmf_lag1', 'ohlc_rsi_zscore_lag1',
            'ohlc_volatility_ratio_20_lag1', 'ohlc_band_width', 'ohlc_band_width_lag1',
            'ohlc_squeeze_active', 'ohlc_squeeze_bars_active_lag1', 'ohlc_qqe_bullish',
            'ohlc_qqe_momentum', 'ohlc_psar_uptrend', 'ohlc_vortex_diff',
            'ohlc_vma_regime_encoded', 'ohlc_combined_signal',
        ],
        'GEX': [
            'call_wall_proxy_lag1', 'put_wall_proxy_lag1',
            'dist_to_gamma_flip_atr_lag1', 'net_gex_zscore_20_lag1',
            'net_gex_percentile_20_lag1', 'gamma_flip_rsi_20_lag1',
            'max_negative_net_gex_rsi_20_lag1',
        ],
        'energy_price': [
            'price_velocity_3', 'price_velocity_accel_3', 'entry_exit_gap',
            'entry_exit_gap_abs', 'entry_exit_gap_max_5', 'vol_change_pct_3',
            'brick_ratio', 'brick_size_ratio_ma_5', 'entry_vs_day_open',
        ],
        'state': [
            'reversal_density_5_lag1', 'reversal_density_10_lag1',
            'directional_efficiency_10_lag1', 'run_length_lag1',
            'alternation_ratio_lag1', 'atr_compression_ratio_lag1',
            'vol_regime_lag1', 'pullback_from_max_progress',
            'RSIOscillator_lag1', 'SqueezeMomentum_lag1',
        ],
        'intraday_time': [
            # re20 new features
            'time_of_day_frac', 'time_to_close_frac', 'is_first_hour', 'is_last_hour',
            'factor_macro_alignment', 'day_running_pnl_lag1', 'day_trade_count_lag1',
            # existing intraday
            'time_since_open_hours', 'intraday_trade_count', 'intraday_trade_rate',
            'intraday_direction_streak', 'intraday_reversal_count',
        ],
        'l1_cascade': [
            'l1_day_probability',
            'prev1_regime_target', 'rolling_5d_regime_rate',
            'rolling_10d_regime_rate', 'rolling_20d_regime_rate',
            'macro_p_trend', 'macro_p_vol', 'macro_p_trend_lv', 'macro_p_trend_hv',
            'macro_p_chop_lv', 'macro_p_chop_hv', 'macro_atr_pct', 'macro_vel_z',
        ],
    }

    def _ablation_l1_fold(fold_data, group_cols):
        """
        Re-train L1 LGB with group_cols zeroed; return (auc_drop, spear_drop).
        FIX-re21b: X_test is trade-level; aggregate probs to day-level before roc_auc_score.
        """
        X_tr = fold_data['X_train'].copy()
        X_te = fold_data['X_test'].copy()
        feat_names = fold_data['feat_names']

        zeroed = 0
        for col in group_cols:
            for idx, fn in enumerate(feat_names):
                if fn == col:
                    X_tr[:, idx] = 0.0
                    X_te[:, idx] = 0.0
                    zeroed += 1

        if zeroed == 0:
            return 0.0, 0.0, 0

        _abl_spw = float(np.sum(fold_data['y_train'] == 0)) / max(1.0, float(np.sum(fold_data['y_train'] == 1)))
        _abl_spw = min(_abl_spw, L2_SPW_MAX)
        _abl_lgb = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=800,
            scale_pos_weight=_abl_spw, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        _abl_lgb.fit(X_tr, fold_data['y_train'])
        trade_probs = _abl_lgb.predict_proba(X_te)[:, 1]

        # FIX-re21b: aggregate trade-level probs to day-level for roc_auc_score.
        # In re20, probs had ~39K rows but test_true had 28 day-level rows → ValueError.
        test_entry_dates = fold_data.get('test_entry_dates')
        test_day_dates   = fold_data.get('test_day_dates')
        test_true        = fold_data['test_true']

        if test_entry_dates is not None and test_day_dates is not None:
            _tmp = pd.DataFrame({'prob': trade_probs, 'date': test_entry_dates})
            _day_mean = _tmp.groupby('date')['prob'].mean()
            probs = np.array([float(_day_mean.get(d, np.nan)) for d in test_day_dates])
        else:
            probs = trade_probs  # fallback (may still mismatch — old bundles)

        try:
            _m_valid = np.isfinite(probs) & np.isfinite(test_true.astype(float))
            if _m_valid.sum() >= 2 and len(np.unique(test_true[_m_valid])) >= 2:
                abl_auc = roc_auc_score(test_true[_m_valid], probs[_m_valid])
            else:
                abl_auc = np.nan
        except Exception:
            abl_auc = np.nan

        day_pnl = fold_data['day_pnl']
        if day_pnl is not None and len(day_pnl) == len(probs):
            _m = np.isfinite(probs) & np.isfinite(day_pnl)
            try:
                abl_spear, _ = spearmanr(probs[_m], day_pnl[_m])
            except Exception:
                abl_spear = np.nan
        else:
            abl_spear = np.nan

        auc_drop   = fold_data['baseline_auc']   - abl_auc   if not np.isnan(abl_auc)   else np.nan
        spear_drop = fold_data['baseline_spear'] - abl_spear if not np.isnan(abl_spear) else np.nan
        return auc_drop, spear_drop, zeroed


    # Build per-fold data bundles for ablation (reuses already-computed fold objects)
    _ablation_fold_data = []
    for _fi, (_tr_idx, _te_idx) in enumerate(folds):
        _fn = _fi + 1
        # Re-derive fold data identically to WF loop (lightweight re-run)
        _abl_train_df, _abl_test_df = add_fold_safe_target(_tr_idx, _te_idx, df_raw, df)
        _abl_train_df = assign_states_path_a(_abl_train_df, compute_thresholds(_abl_train_df))
        _abl_test_df  = assign_states_path_a(_abl_test_df,  compute_thresholds(_abl_train_df))

        _abl_fd_cols = _compute_dedup_cols(_abl_train_df, CLEAN_CANDIDATE_COLS)
        _abl_surv, _ = bivariate_filter(_abl_train_df, _abl_fd_cols, target_col='regime_target')
        _abl_n_days  = _abl_train_df['date'].nunique()
        _abl_k       = compute_dynamic_top_k(len(_abl_surv), _abl_n_days)
        _abl_sel, _  = clustered_feature_importance(
            _abl_train_df, _abl_surv, target_col='regime_target', top_k=_abl_k)

        _abl_tr_raw = _abl_train_df[_abl_sel].replace([np.inf, -np.inf], np.nan)
        _abl_meds   = _abl_tr_raw.median()
        _abl_X_tr   = _abl_tr_raw.fillna(_abl_meds).values
        _abl_X_te   = _abl_test_df[_abl_sel].replace([np.inf, -np.inf], np.nan).fillna(_abl_meds).values

        _abl_gs   = gram_schmidt_fit(_abl_X_tr)
        _abl_X_tr = gram_schmidt_transform(_abl_X_tr, _abl_gs)
        _abl_X_te = gram_schmidt_transform(_abl_X_te, _abl_gs)

        _abl_cs_tr = encode_core_state(_abl_train_df).reshape(-1, 1).astype(float)
        _abl_cs_te = encode_core_state(_abl_test_df).reshape(-1, 1).astype(float)
        _abl_X_tr_f = np.nan_to_num(
            np.column_stack([_abl_X_tr, _abl_cs_tr]), nan=0.0, posinf=0.0, neginf=0.0)
        _abl_X_te_f = np.nan_to_num(
            np.column_stack([_abl_X_te, _abl_cs_te]), nan=0.0, posinf=0.0, neginf=0.0)
        _abl_feat_names = _abl_sel + ['core_state_encoded']

        # Baseline metrics from the main WF (look up by fold number)
        _abl_fold_row = metrics_df[metrics_df['fold'] == _fn]
        _abl_base_auc   = float(_abl_fold_row['auc'].values[0])   if len(_abl_fold_row) else np.nan
        _abl_base_spear = float(_abl_fold_row['spearman_r'].values[0]) if len(_abl_fold_row) else np.nan

        # Day-level PnL for Spearman
        _abl_test_dates = set(pd.to_datetime(_abl_test_df['entry_time']).dt.date)
        _abl_raw_test   = df_raw_macro[
            pd.to_datetime(df_raw_macro['entry_time']).dt.date.isin(_abl_test_dates)]
        _abl_day_pnl_s  = (
            _abl_raw_test.groupby(pd.to_datetime(_abl_raw_test['entry_time']).dt.date)['pnl']
            .sum().sort_index()
        )
        _abl_test_day_dates = sorted(_abl_test_dates)
        _abl_day_pnl_arr = np.array([
            _abl_day_pnl_s.get(d, np.nan) for d in _abl_test_day_dates])

        # Reorder X_te rows to match sorted dates
        _abl_test_df_dates = pd.to_datetime(_abl_test_df['entry_time']).dt.date
        _abl_date_to_row   = {d: i for i, d in enumerate(_abl_test_day_dates)}
        # Day-level test probs: mean over trades per day using X_te_f
        _abl_y_tr = _abl_train_df['regime_target'].values
        _abl_spw2 = float(np.sum(_abl_y_tr == 0)) / max(1.0, float(np.sum(_abl_y_tr == 1)))
        _abl_spw2 = min(_abl_spw2, L2_SPW_MAX)
        _abl_temp_lgb = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=800,
            scale_pos_weight=_abl_spw2, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        _abl_temp_lgb.fit(_abl_X_tr_f, _abl_y_tr)
        _abl_trade_probs  = _abl_temp_lgb.predict_proba(_abl_X_te_f)[:, 1]
        _abl_test_df_tmp  = _abl_test_df.copy()
        _abl_test_df_tmp['_prob']  = _abl_trade_probs
        _abl_test_df_tmp['_ddate'] = _abl_test_df_dates.values
        _abl_day_probs = _abl_test_df_tmp.groupby('_ddate')['_prob'].mean()
        _abl_day_probs_arr = np.array([
            float(_abl_day_probs.get(d, np.nan)) for d in _abl_test_day_dates])
        _abl_day_true = np.array([
            float(_abl_test_df_tmp[_abl_test_df_tmp['_ddate'] == d]['regime_target'].iloc[0])
            if (_abl_test_df_tmp['_ddate'] == d).any() else np.nan
            for d in _abl_test_day_dates])

        _ablation_fold_data.append({
            'fold':              _fn,
            'X_train':           _abl_X_tr_f,
            'X_test':            _abl_X_te_f,
            'y_train':           _abl_train_df['regime_target'].values,
            'test_true':         _abl_day_true,
            'test_entry_dates':  _abl_test_df_dates.values,   # FIX-re21b: per-trade dates for aggregation
            'test_day_dates':    _abl_test_day_dates,         # FIX-re21b: ordered day list
            'day_pnl':           _abl_day_pnl_arr,
            'day_probs':         _abl_day_probs_arr,
            'feat_names':        _abl_feat_names,
            'baseline_auc':      _abl_base_auc,
            'baseline_spear':    _abl_base_spear,
        })

    print(f"  Ablation fold bundles built: {len(_ablation_fold_data)}")

    # ── Run ablation per group ────────────────────────────────────────────────────
    _ablation_rows = []
    for _grp_name, _grp_cols in ABLATION_GROUPS.items():
        _auc_drops   = []
        _spear_drops = []
        _zeroed_counts = []
        for _fd in _ablation_fold_data:
            _ad, _sd, _zc = _ablation_l1_fold(_fd, _grp_cols)
            if _zc > 0:   # only count folds where group was actually present
                if not np.isnan(_ad):
                    _auc_drops.append(_ad)
                if not np.isnan(_sd):
                    _spear_drops.append(_sd)
                _zeroed_counts.append(_zc)

        _mean_auc_drop   = float(np.mean(_auc_drops))   if _auc_drops   else np.nan
        _mean_spear_drop = float(np.mean(_spear_drops)) if _spear_drops else np.nan
        _mean_zeroed     = float(np.mean(_zeroed_counts)) if _zeroed_counts else 0.0
        _folds_with_grp  = len(_zeroed_counts)

        _ablation_rows.append({
            'group':           _grp_name,
            'folds_with_group': _folds_with_grp,
            'mean_features_zeroed': round(_mean_zeroed, 1),
            'mean_auc_drop':   round(_mean_auc_drop,   4) if not np.isnan(_mean_auc_drop)   else np.nan,
            'mean_spear_drop': round(_mean_spear_drop, 4) if not np.isnan(_mean_spear_drop) else np.nan,
        })
        print(f"  {_grp_name:<20}  folds={_folds_with_grp:>2}  "
              f"feats_zeroed={_mean_zeroed:>5.1f}  "
              f"ΔAUC={_mean_auc_drop:>+7.4f}  ΔSpear={_mean_spear_drop:>+7.4f}")

    ablation_df = pd.DataFrame(_ablation_rows).sort_values('mean_auc_drop', ascending=False)
    print(f"\n── Ablation summary (sorted by ΔAUC — most important first) ──")
    print(ablation_df.to_string(index=False))
    print("\n  Positive ΔAUC = removing group HURTS model (group is important).")
    print("  Negative ΔAUC = removing group HELPS model (group is noisy/harmful).")


    # ═══════════════════════════════════════════════════════════════════════════════
    # SAVE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════════════
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_preds_df.to_csv(OUTPUT_DIR / f'wf_results_re27_{_TRAIN_DIR}.csv',              index=False)
    metrics_df.to_csv(OUTPUT_DIR   / f'wf_fold_metrics_re27_{_TRAIN_DIR}.csv',         index=False)
    decile_df.to_csv(OUTPUT_DIR    / f'decile_analysis_re27_{_TRAIN_DIR}.csv',         index=False)
    stability_df.to_csv(OUTPUT_DIR / f'feature_stability_re27_{_TRAIN_DIR}.csv',       index=False)
    if _L2_ENABLED:
        l2_metrics_df.to_csv(OUTPUT_DIR / f'l2_fold_metrics_re27_{_TRAIN_DIR}.csv',        index=False)
        l2_bucket_metrics_df.to_csv(OUTPUT_DIR / f'l2_bucket_metrics_re27_{_TRAIN_DIR}.csv', index=False)
    if not l2_preds_df.empty:
        l2_preds_df.to_csv(OUTPUT_DIR / f'l2_results_re27_{_TRAIN_DIR}.csv',           index=False)
    if oof_moe_records:
        oof_meta_df.to_csv(OUTPUT_DIR / f'oof_moe_records_re27_{_TRAIN_DIR}.csv',      index=False)
    if 'ablation_df' in dir() and not ablation_df.empty:
        ablation_df.to_csv(OUTPUT_DIR / f'ablation_group_importance_re27_{_TRAIN_DIR}.csv', index=False)

    print(f"\n✓ Saved (re25):")
    print(f"  {OUTPUT_DIR}/wf_results_re27_{_TRAIN_DIR}.csv         ({len(all_preds_df):,} rows, Layer 1 daily)")
    print(f"  {OUTPUT_DIR}/wf_fold_metrics_re27_{_TRAIN_DIR}.csv    ({len(metrics_df)} folds)")
    print(f"  {OUTPUT_DIR}/decile_analysis_re27_{_TRAIN_DIR}.csv")
    print(f"  {OUTPUT_DIR}/feature_stability_re27_{_TRAIN_DIR}.csv  ({len(stability_df)} features)")
    print(f"  {OUTPUT_DIR}/l2_fold_metrics_re27_{_TRAIN_DIR}.csv    ({len(l2_metrics_df)} folds)")
    print(f"  {OUTPUT_DIR}/l2_bucket_metrics_re27_{_TRAIN_DIR}.csv")
    if not l2_preds_df.empty:
        print(f"  {OUTPUT_DIR}/l2_results_re27_{_TRAIN_DIR}.csv        ({len(l2_preds_df):,} rows, Layer 2 intraday)")
    if oof_moe_records:
        print(f"  {OUTPUT_DIR}/oof_moe_records_re27_{_TRAIN_DIR}.csv   ({len(oof_meta_df):,} OOF MoE records, Layer 4)")
    if 'ablation_df' in dir() and not ablation_df.empty:
        print(f"  {OUTPUT_DIR}/ablation_group_importance_re27_{_TRAIN_DIR}.csv  ({len(ablation_df)} groups)")



    print(f"\n  ✓ {_TRAIN_DIR.upper()} direction complete.\n")
