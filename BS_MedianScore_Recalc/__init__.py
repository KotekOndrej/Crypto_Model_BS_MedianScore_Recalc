import os
import io
import sys
import logging
from datetime import datetime, timezone

# ============ ENV helpers ============
def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return float(default)
    try:
        return float(v)
    except ValueError:
        logging.warning(f"[ENV] {name}='{v}' není číslo, používám default {default}")
        return float(default)

def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return int(default)
    try:
        return int(v)
    except ValueError:
        logging.warning(f"[ENV] {name}='{v}' není integer, používám default {default}")
        return int(default)

# ============ Parametry ============
N_DAYS = _get_env_int("N_DAYS", 20)
MIN_DAYS_REQUIRED = _get_env_int("MIN_DAYS_REQUIRED", 12)
COSTS_PCT = _get_env_float("COSTS_PCT", 0.001)     # 0.1 % na cyklus
GAP_MIN_PCT = _get_env_float("GAP_MIN_PCT", 0.003) # 0.3 %

PEAKS_K = 60
MODEL_NAME = "BS_MedianScore"

# Storage
WEBJOBS_CONN = os.getenv("AzureWebJobsStorage")
IN_CONTAINER  = os.getenv("INPUT_CONTAINER", "market-data")
OUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "market-signals")

MASTER_CSV_NAME = "bs_levels_master.csv"
DAILY_PREFIX    = "bs_levels_"

# ============ Utility ============
def extract_pair_from_filename(blob_name: str) -> str:
    base = os.path.splitext(os.path.basename(blob_name))[0]
    for sep in ['_', '-']:
        if sep in base:
            return base.split(sep)[0]
    return base

def find_local_peaks(y, k_peaks=60, min_separation=2):
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append((y[i], i))
    peaks.sort(reverse=True)
    taken, res = set(), []
    for _, idx in peaks:
        if all(abs(idx - j) > min_separation for j in taken):
            taken.add(idx)
            res.append(idx)
        if len(res) >= k_peaks:
            break
    return res

def bin_center(bins, i):
    return 0.5 * (bins[i] + bins[i + 1])

def simulate_day_no_timeout_side(B, S, L, H, costs_abs=0.0, side="both"):
    pnl = 0.0
    cycles = 0
    pos = 0
    for (l, h) in zip(L, H):
        tb = (l <= B <= h)
        ts = (l <= S <= h)
        if pos == 0:
            if side in ("both", "long") and tb:
                pos = +1
            elif side in ("both", "short") and ts:
                pos = -1
        else:
            if pos == +1 and ts:
                pnl += (S - B) - costs_abs
                pos = 0
                cycles += 1
            elif pos == -1 and tb:
                pnl += (S - B) - costs_abs
                pos = 0
                cycles += 1
    return pnl, cycles

def rolling_mean(arr, w):
    out = [0.0] * len(arr)
    for i in range(len(arr)):
        j0 = max(0, i - w + 1)
        seg = arr[j0:i+1]
        out[i] = sum(seg) / max(len(seg), 1)
    return out

def rolling_std(arr, w):
    out = [0.0] * len(arr)
    for i in range(len(arr)):
        j0 = max(0, i - w + 1)
        seg = arr[j0:i+1]
        m = sum(seg) / max(len(seg), 1)
        var = sum((v - m) ** 2 for v in seg) / max(len(seg), 1)
        out[i] = var ** 0.5
    return out

def price_only_features(opens, highs, lows, closes):
    ret_1d = [0.0] + [(closes[i] - closes[i-1]) / (closes[i-1] or 1e-9) for i in range(1, len(closes))]
    day_range = [(highs[i] - lows[i]) / (closes[i] or 1e-9) for i in range(len(closes))]
    close_open = [(closes[i] - opens[i]) / (opens[i] or 1e-9) for i in range(len(closes))]
    close_pos = [(closes[i] - lows[i]) / ((highs[i] - lows[i]) or 1e-9) for i in range(len(closes))]
    ret_3d = rolling_mean(ret_1d, 3)
    ret_5d = rolling_mean(ret_1d, 5)
    rng_3d = rolling_mean(day_range, 3)
    rng_5d = rolling_mean(day_range, 5)
    vol_3d = rolling_std(ret_1d, 3)
    vol_5d = rolling_std(ret_1d, 5)
    return list(zip(ret_1d, day_range, close_open, close_pos,
                    ret_3d, ret_5d, rng_3d, rng_5d, vol_3d, vol_5d))

def logistic_regression_irls(X, y, max_iter=100, tol=1e-6, reg_lambda=1e-4):
    import math
    n = len(X)
    if n == 0:
        return [0.0] * (len(X[0]) + 1)
    d = len(X[0])
    Xb = [[1.0] + list(row) for row in X]
    w = [0.0] * (d + 1)
    def dot(u, v): return sum(ui*vi for ui,vi in zip(u,v))
    for _ in range(max_iter):
        z = [dot(Xb[i], w) for i in range(n)]
        p = [1.0 / (1.0 + math.exp(-zi)) for zi in z]
        W = [pi * (1 - pi) for pi in p]
        H = [[0.0]*(d+1) for _ in range(d+1)]
        g = [0.0]*(d+1)
        for i in range(n):
            wi = W[i]
            for a in range(d+1):
                g[a] += Xb[i][a] * (p[i] - y[i])
                for b in range(d+1):
                    H[a][b] += Xb[i][a] * wi * Xb[i][b]
        for a in range(d+1):
            H[a][a] += reg_lambda
            g[a] += reg_lambda * w[a]
        A = [row[:] + [g[i]] for i, row in enumerate(H)]
        for col in range(d+1):
            piv = col
            for r in range(col+1, d+1):
                if abs(A[r][col]) > abs(A[piv][col]):
                    piv = r
            A[col], A[piv] = A[piv], A[col]
            pivot = A[col][col] or 1e-12
            for c in range(col, d+2):
                A[col][c] /= pivot
            for r in range(d+1):
                if r == col: continue
                factor = A[r][col]
                for c in range(col, d+2):
                    A[r][c] -= factor * A[col][c]
        step = [A[i][d+1] for i in range(d+1)]
        w_new = [w[i] - step[i] for i in range(d+1)]
        diff = sum((w_new[i]-w[i])**2 for i in range(d+1)) ** 0.5
        w = w_new
        if diff < tol:
            break
    return w

def predict_proba_logreg(w, xrow):
    import math
    xb = [1.0] + list(xrow)
    z = sum(a*b for a,b in zip(xb, w))
    return 1.0 / (1.0 + math.exp(-z))

# ============ Hlavní výpočet ============
def compute_bs_for_csv_bytes(csv_bytes: bytes, pair_name: str):
    import numpy as np
    import pandas as pd

    def _make_bins(p_min, p_max, n_bins=600):
        if p_max <= p_min:
            p_max = p_min * 1.001
        return np.linspace(p_min, p_max, n_bins + 1)

    def _gaussian_smooth(x, sigma_bins=3):
        if sigma_bins <= 0: return x
        radius = int(3 * sigma_bins)
        i = np.arange(-radius, radius + 1)
        k = np.exp(-(i * i) / (2 * sigma_bins * sigma_bins))
        k = k / k.sum()
        return np.convolve(x, k, mode='same')

    def _build_touch_hist_day(L, H, bins):
        hist = np.zeros(len(bins) - 1, dtype=float)
        for l, h in zip(L, H):
            if h < bins[0] or l > bins[-1]:
                continue
            i0 = max(0, np.searchsorted(bins, l, side='right') - 1)
            i1 = min(len(bins) - 1, np.searchsorted(bins, h, side='left'))
            if i1 > i0:
                hist[i0:i1] += 1.0
        return hist

    # --- load ---
    df = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"closeTimeISO", "low", "high"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pair_name}: CSV musí obsahovat {required}, mám {set(df.columns)}")

    df["closeTimeISO"] = pd.to_datetime(df["closeTimeISO"], errors="coerce", utc=True)
    df = df.dropna(subset=["closeTimeISO", "low", "high"])
    df["date"] = df["closeTimeISO"].dt.date

    days = sorted(df["date"].unique())
    if len(days) < MIN_DAYS_REQUIRED:
        return None

    last_days = days[-N_DAYS:] if len(days) >= N_DAYS else days
    history = []
    for d in last_days:
        sub = df[df["date"] == d]
        L = sub["low"].to_numpy(dtype=float)
        H = sub["high"].to_numpy(dtype=float)
        if len(L) == 0:
            continue
        history.append((L, H, d))
    if len(history) < MIN_DAYS_REQUIRED:
        return None

    # --- histogram ---
    p_min = min(float(l.min()) for l, _, _ in history)
    p_max = max(float(h.max()) for _, h, _ in history)
    bins = _make_bins(p_min, p_max, n_bins=600)
    hist = np.zeros(len(bins) - 1, dtype=float)
    for L, H, _ in history:
        hist += _build_touch_hist_day(L, H, bins)
    hist_smooth = _gaussian_smooth(hist, sigma_bins=3)

    # --- daily proxies + features for side selection ---
    daily_closes = [float(np.mean((L + H) / 2)) for (L, H, _) in history]
    daily_opens  = [float((L[0] + H[0]) / 2)    for (L, H, _) in history]
    daily_highs  = [float(H.max())              for (_, H, _) in history]
    daily_lows   = [float(L.min())              for (L, _, _) in history]

    closes = np.array(daily_closes, dtype=float)
    opens  = np.array(daily_opens, dtype=float)
    highs  = np.array(daily_highs, dtype=float)
    lows   = np.array(daily_lows, dtype=float)

    X = price_only_features(list(opens), list(highs), list(lows), list(closes))
    y = (np.roll(closes, -1) > closes).astype(int).tolist()[:-1]
    X = X[:-1]

    if len(y) < 5:
        side = "both"
        p_up = 0.5
    else:
        w = logistic_regression_irls(X[:-1], y[:-1])
        p_up = predict_proba_logreg(w, X[-1])
        side = "long" if p_up >= 0.6 else "short" if p_up <= 0.4 else "both"

    # --- candidates ---
    peak_idx = find_local_peaks(hist_smooth, k_peaks=PEAKS_K, min_separation=2)
    levels = [bin_center(bins, i) for i in peak_idx]
    current_price = closes[-1]
    avg_price = float(np.mean([np.mean((L + H) / 2) for (L, H, _) in history]))
    min_gap_abs = GAP_MIN_PCT * avg_price

    lower = sorted([lv for lv in levels if lv <= current_price], key=lambda x: abs(x - current_price))[:12]
    upper = sorted([lv for lv in levels if lv >= current_price], key=lambda x: abs(x - current_price))[:12]
    pairs = [(B, S) for B in lower for S in upper if (S - B) >= min_gap_abs]
    if not pairs:
        lower = sorted([lv for lv in levels if lv <= current_price])[:30]
        upper = sorted([lv for lv in levels if lv >= current_price])[:30]
        pairs = [(B, S) for B in lower for S in upper if (S - B) >= min_gap_abs]
        if not pairs:
            return None
    n_candidates = len(pairs)

    def eval_pair(B, S):
        costs_abs = COSTS_PCT * ((B + S) / 2.0)
        pnls = []
        cycles = []
        for L, H, _ in history:
            p, c = simulate_day_no_timeout_side(B, S, L, H, costs_abs=costs_abs, side=side)
            pnls.append(p)
            cycles.append(c)
        pnls = np.array(pnls, dtype=float)
        med = float(np.median(pnls))
        iqr = float(np.percentile(pnls, 75) - np.percentile(pnls, 25))
        score = med - 0.25 * iqr
        total_cycles = int(np.sum(cycles))
        base = max(((B + S) / 2.0), 1e-12)
        avg_pnl_pct = float(np.mean(pnls) / base) * 100.0
        median_pnl_pct = float(med / base) * 100.0
        gap_pct = (S - B) / max(B, 1e-12)
        dist = abs(((B + S) / 2) - current_price) / max(current_price, 1e-12)
        return total_cycles, score, gap_pct, dist, avg_pnl_pct, median_pnl_pct

    best = None
    best_metrics = None
    for (B, S) in pairs:
        met = eval_pair(B, S)
        if best is None:
            best = (B, S)
            best_metrics = met
            continue
        if (met[0], met[1], -met[2], -met[3]) > (best_metrics[0], best_metrics[1], -best_metrics[2], -best_metrics[3]):
            best = (B, S)
            best_metrics = met

    if best is None:
        return None

    bestB, bestS = best
    total_cycles, score, gap_pct, dist_rel, avg_pnl_pct, median_pnl_pct = best_metrics
    gap_pct *= 100.0
    costs_abs = COSTS_PCT * ((bestB + bestS) / 2.0)

    # detailní metriky pro best pár
    pnls = []
    day_cycles = []
    for L, H, _ in history:
        p, c = simulate_day_no_timeout_side(bestB, bestS, L, H, costs_abs=costs_abs, side=side)
        pnls.append(p)
        day_cycles.append(c)

    # použijeme stejné np jako výše (žádné aliasy)
    import numpy as np  # lokální, už je importované, ale je to bezpečné
    pnls_arr = np.array(pnls, dtype=float)
    iqr_pnl = float(np.percentile(pnls_arr, 75) - np.percentile(pnls_arr, 25))
    std_pnl = float(np.std(pnls_arr))
    pnl_p05 = float(np.percentile(pnls_arr, 5))
    pnl_p95 = float(np.percentile(pnls_arr, 95))
    hit_rate_days = int(np.sum(np.array(day_cycles) > 0))
    dist_to_price = float(dist_rel)

    return {
        "pair": pair_name,
        "B": bestB,
        "S": bestS,
        "gap_pct": gap_pct,
        "model": MODEL_NAME,
        "total_cycles": total_cycles,
        "score": score,
        "avg_pnl_pct": avg_pnl_pct,
        "median_pnl_pct": median_pnl_pct,
        # nové metriky
        "side": side,
        "p_up": float(p_up),
        "iqr_pnl": iqr_pnl,
        "std_pnl": std_pnl,
        "pnl_p05": pnl_p05,
        "pnl_p95": pnl_p95,
        "hit_rate_days": hit_rate_days,
        "dist_to_price": dist_to_price,
        "min_gap_abs": float(min_gap_abs),
        "costs_abs": float(costs_abs),
        "n_candidates": int(n_candidates),
    }

# ============ ENTRYPOINT ============
def main(myTimer):
    try:
        import pandas as pd
        from azure.storage.blob import BlobServiceClient
    except Exception:
        logging.exception("[BSMedianScoreRecalc] Import balíčků selhal.")
        return

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    load_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    func_name = "BSMedianScoreRecalc"

    logging.info(
        f"[{func_name}] Start {date_str} UTC | Python={sys.version.split()[0]} | "
        f"N_DAYS={N_DAYS} MIN_DAYS_REQUIRED={MIN_DAYS_REQUIRED} "
        f"COSTS_PCT={COSTS_PCT} GAP_MIN_PCT={GAP_MIN_PCT} | "
        f"IN_CONTAINER={IN_CONTAINER} OUT_CONTAINER={OUT_CONTAINER}"
    )

    if not WEBJOBS_CONN:
        logging.error(f"[{func_name}] Chybí App Setting 'AzureWebJobsStorage'.")
        return

    try:
        blob_service = BlobServiceClient.from_connection_string(WEBJOBS_CONN)
    except Exception:
        logging.exception(f"[{func_name}] Nelze vytvořit BlobServiceClient z AzureWebJobsStorage.")
        return

    in_container_client  = blob_service.get_container_client(IN_CONTAINER)
    out_container_client = blob_service.get_container_client(OUT_CONTAINER)
    try:
        out_container_client.create_container()
    except Exception:
        pass

    try:
        blobs = list(in_container_client.list_blobs())
        logging.info(f"[{func_name}] Nalezeno {len(blobs)} objektů v '{IN_CONTAINER}'.")
    except Exception:
        logging.exception(f"[{func_name}] Chyba při listování blobů v '{IN_CONTAINER}'")
        return

    new_rows = []
    for blob in blobs:
        if not blob.name.lower().endswith(".csv"):
            continue
        pair_name = extract_pair_from_filename(blob.name)
        try:
            csv_bytes = in_container_client.get_blob_client(blob).download_blob().readall()
            res = compute_bs_for_csv_bytes(csv_bytes, pair_name)
            if res is None:
                logging.warning(f"[{func_name}] Přeskakuji {blob.name} – žádný výsledek.")
                continue
            new_rows.append({
                "pair": res["pair"],
                "B": float(res["B"]),
                "S": float(res["S"]),
                "gap_pct": float(res["gap_pct"]),
                "date": date_str,
                "model": res["model"],
                "is_active": True,
                "total_cycles": int(res.get("total_cycles", 0)),
                "score": float(res.get("score", 0.0)),
                "avg_pnl_pct": float(res.get("avg_pnl_pct", 0.0)),
                "median_pnl_pct": float(res.get("median_pnl_pct", 0.0)),
                "load_time_utc": load_time,
                # nové metriky
                "side": res.get("side"),
                "p_up": float(res.get("p_up", 0.5)),
                "iqr_pnl": float(res.get("iqr_pnl", 0.0)),
                "std_pnl": float(res.get("std_pnl", 0.0)),
                "pnl_p05": float(res.get("pnl_p05", 0.0)),
                "pnl_p95": float(res.get("pnl_p95", 0.0)),
                "hit_rate_days": int(res.get("hit_rate_days", 0)),
                "dist_to_price": float(res.get("dist_to_price", 0.0)),
                "min_gap_abs": float(res.get("min_gap_abs", 0.0)),
                "costs_abs": float(res.get("costs_abs", 0.0)),
                "n_candidates": int(res.get("n_candidates", 0)),
            })
            logging.info(
                f"[{func_name}] OK {pair_name}: B={res['B']:.6f}, S={res['S']:.6f}, "
                f"gap={res['gap_pct']:.3f}%, cycles={int(res.get('total_cycles',0))}, "
                f"avg_pnl={float(res.get('avg_pnl_pct',0.0)):.3f}%, side={res.get('side')} "
                f"(p_up={float(res.get('p_up',0.5)):.3f}), iqr={float(res.get('iqr_pnl',0.0)):.6f}"
            )
        except Exception:
            logging.exception(f"[{func_name}] Chyba při zpracování {blob.name}")
            continue

    if not new_rows:
        logging.warning(f"[{func_name}] Nebyly vyprodukovány žádné výsledky.")
        return

    import pandas as pd
    cols = ["pair","B","S","gap_pct","date","model","is_active",
            "total_cycles","score","avg_pnl_pct","median_pnl_pct","load_time_utc",
            "side","p_up","iqr_pnl","std_pnl","pnl_p05","pnl_p95",
            "hit_rate_days","dist_to_price","min_gap_abs","costs_abs","n_candidates"]
    new_df = pd.DataFrame(new_rows, columns=cols)

    # 3) denní snapshot
    daily_name = f"{DAILY_PREFIX}{date_str}.csv"
    try:
        out_container_client.get_blob_client(daily_name).upload_blob(
            new_df.to_csv(index=False).encode("utf-8"),
            overwrite=True
        )
        logging.info(f"[{func_name}] Zapsán denní snapshot: {OUT_CONTAINER}/{daily_name} ({len(new_df)} řádků).")
    except Exception:
        logging.exception(f"[{func_name}] Zápis denního snapshotu selhal.")

    # 4) Master CSV – deaktivace starých aktivních a append nových
    master_cols = cols
    try:
        master_blob = out_container_client.get_blob_client(MASTER_CSV_NAME)
        if master_blob.exists():
            master_bytes = master_blob.download_blob().readall()
            master_df = pd.read_csv(io.BytesIO(master_bytes))
            for c in master_cols:
                if c not in master_df.columns:
                    master_df[c] = None
            master_df = master_df[master_cols]
        else:
            master_df = pd.DataFrame(columns=master_cols)
    except Exception:
        logging.exception(f"[{func_name}] Chyba při načtení master CSV – vytvořím nový.")
        master_df = pd.DataFrame(columns=master_cols)

    if not master_df.empty:
        for _, row in new_df.iterrows():
            mask = (
                (master_df["pair"] == row["pair"]) &
                (master_df["model"] == MODEL_NAME) &   # <- explicitně jen BS_MedianScore
                (master_df["is_active"] == True)    
            )
            if mask.any():
                master_df.loc[mask, "is_active"] = False

    master_df = pd.concat([master_df, new_df], ignore_index=True)

    try:
        out_container_client.get_blob_client(MASTER_CSV_NAME).upload_blob(
            master_df.to_csv(index=False).encode("utf-8"),
            overwrite=True
        )
        logging.info(f"[{func_name}] Aktualizován master: {OUT_CONTAINER}/{MASTER_CSV_NAME} (n={len(master_df)}).")
    except Exception:
        logging.exception(f"[{func_name}] Zápis master CSV selhal.")
