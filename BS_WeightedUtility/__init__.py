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

def _get_env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default

# ============ Parametry ============
MODEL_NAME = "BS_WeightedUtility"

# Datové okno a váhy
N_DAYS = _get_env_int("N_DAYS", 30)                  # počet dní pro konstrukci profilu a evaluaci
MIN_DAYS_REQUIRED = _get_env_int("MIN_DAYS_REQUIRED", 15)
HALF_LIFE_DAYS = _get_env_float("HALF_LIFE_DAYS", 8) # poločas pro time-decay váhy (dny)

# Náklady a gap
BASE_COSTS_PCT = _get_env_float("COSTS_PCT", 0.001)    # 0.1 % na cyklus (z průměrné ceny B/S)
GAP_MIN_BPS = _get_env_float("GAP_MIN_BPS", 30.0)      # minimální gap v basis bodech fallback (0.30 %)

# Scoring utility parametry
LAMBDA_IQR = _get_env_float("LAMBDA_IQR", 0.25)
MU_MAXDD   = _get_env_float("MU_MAXDD", 1.0)

# Profil
BIN_BPS = _get_env_float("BIN_BPS", 5.0)               # šířka log-binu v basis bodech (0.05 %)
PEAKS_K = _get_env_int("PEAKS_K", 60)
SMOOTH_SIGMA_BINS = _get_env_float("SMOOTH_SIGMA_BINS", 3.0)

# Storage (beze změny názvů výstupů)
WEBJOBS_CONN = os.getenv("AzureWebJobsStorage")
IN_CONTAINER  = os.getenv("INPUT_CONTAINER", "market-data")
OUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "market-signals")
MASTER_CSV_NAME = "bs_levels_master.csv"
DAILY_PREFIX    = "bs_levels_"

# ============ Utility ============
def extract_pair_from_filename(blob_name: str) -> str:
    import os as _os
    base = _os.path.splitext(_os.path.basename(blob_name))[0]
    for sep in ['_', '-']:
        if sep in base:
            return base.split(sep)[0]
    return base

def exp_decay_weights(n_days: int, half_life: float):
    import numpy as np
    ages = np.arange(n_days-1, -1, -1)  # age 0 = nejnovější den (poslední v historii)
    alpha = (np.log(2.0) / max(half_life, 1e-6))
    w = np.exp(-alpha * ages)
    sw = w.sum() or 1.0
    return w / sw  # normalizace na 1

def rolling_mean(arr, w):
    out = [0.0] * len(arr)
    for i in range(len(arr)):
        j0 = max(0, i - w + 1)
        seg = arr[j0:i+1]
        out[i] = sum(seg) / max(len(seg), 1)
    return out

def price_only_features(opens, highs, lows, closes):
    ret_1d = [0.0] + [(closes[i] - closes[i-1]) / (closes[i-1] or 1e-9) for i in range(1, len(closes))]
    day_range = [(highs[i] - lows[i]) / (closes[i] or 1e-9) for i in range(len(closes))]
    close_open = [(closes[i] - opens[i]) / (opens[i] or 1e-9) for i in range(len(closes))]
    close_pos = [(closes[i] - lows[i]) / ((highs[i] - lows[i]) or 1e-9) for i in range(len(closes))]
    ret_3d = rolling_mean(ret_1d, 3)
    ret_5d = rolling_mean(ret_1d, 5)
    return list(zip(ret_1d, day_range, close_open, close_pos, ret_3d, ret_5d))

def logistic_regression_irls(X, y, sample_weight=None, max_iter=100, tol=1e-6, reg_lambda=1e-4):
    import math
    n = len(X)
    if n == 0:
        return [0.0] * (len(X[0]) + 1)
    d = len(X[0])
    Xb = [[1.0] + list(row) for row in X]
    w = [0.0] * (d + 1)
    if sample_weight is None:
        sample_weight = [1.0]*n

    def dot(u, v): return sum(ui*vi for ui,vi in zip(u,v))
    for _ in range(max_iter):
        z = [dot(Xb[i], w) for i in range(n)]
        p = [1.0 / (1.0 + math.exp(-zi)) for zi in z]
        W = [sample_weight[i] * p[i] * (1 - p[i]) for i in range(n)]
        H = [[0.0]*(d+1) for _ in range(d+1)]
        g = [0.0]*(d+1)
        for i in range(n):
            wi = W[i]
            for a in range(d+1):
                g[a] += sample_weight[i] * Xb[i][a] * (p[i] - y[i])
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

def make_log_bins(pmin, pmax, bps):
    import numpy as np
    if pmax <= pmin:
        pmax = pmin * 1.001
    step = max(bps, 0.1) / 10000.0
    n = int(np.ceil(np.log(pmax / pmin) / step))
    edges = pmin * np.exp(step * np.arange(n+1))
    return edges

def gaussian_smooth_1d(x, sigma_bins):
    import numpy as np
    if sigma_bins <= 0: return x
    radius = int(3 * sigma_bins)
    i = np.arange(-radius, radius + 1)
    k = np.exp(-(i * i) / (2 * sigma_bins * sigma_bins))
    k = k / k.sum()
    return np.convolve(x, k, mode='same')

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

def daily_stats_1m(df_day):
    import numpy as np
    if df_day.empty:
        return None
    c = df_day["close"].to_numpy(dtype=float)
    h = df_day["high"].to_numpy(dtype=float)
    l = df_day["low"].to_numpy(dtype=float)
    qv = df_day.get("quoteVolume", None)
    nt = df_day.get("numTrades", None)
    tbq = df_day.get("takerBuyQuote", None)

    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(prev_c - l)])
    atr_d = float(np.mean(tr))

    ret = np.diff(np.log(c + 1e-12), prepend=np.log(c[0] + 1e-12))
    rv_d = float(np.sqrt(np.sum(ret * ret)))

    avg_qv = float(np.mean(qv.to_numpy(dtype=float))) if qv is not None else 0.0
    avg_nt = float(np.mean(nt.to_numpy(dtype=float))) if nt is not None else 0.0

    imb = 0.0
    if tbq is not None and qv is not None:
        q = qv.to_numpy(dtype=float)
        tb = tbq.to_numpy(dtype=float)
        denom = np.maximum(q, 1e-12)
        imb = float(np.mean(tb / denom))

    vwap = float(np.sum(c * (qv.to_numpy(dtype=float) if qv is not None else 1.0)) /
                 np.maximum(np.sum((qv.to_numpy(dtype=float) if qv is not None else 1.0)), 1e-12))

    return {
        "atr_d": atr_d,
        "rv_d": rv_d,
        "avg_qv": avg_qv,
        "avg_nt": avg_nt,
        "imbalance_d": imb,
        "vwap_d": vwap
    }

def classify_regime(win_stats):
    import numpy as np
    if len(win_stats) < 5:
        return "range"
    atr = np.array([s["atr_d"] for s in win_stats], dtype=float)
    rv  = np.array([s["rv_d"]  for s in win_stats], dtype=float)
    imb = np.array([s["imbalance_d"] for s in win_stats], dtype=float)
    imb_slope = float(np.mean(imb[-3:]) - np.mean(imb[:3])) if len(imb) >= 6 else float(np.mean(imb) - 0.5)
    vol_now = rv[-1]
    vol_p90 = float(np.percentile(rv, 90))
    if vol_now >= vol_p90:
        return "vol"
    if abs(imb_slope) > 0.05:
        return "trend"
    return "range"

def build_profile_weighted(df_day, bins):
    import numpy as np
    lows  = df_day["low"].to_numpy(dtype=float)
    highs = df_day["high"].to_numpy(dtype=float)
    qv    = (df_day.get("quoteVolume", None).to_numpy(dtype=float)
             if "quoteVolume" in df_day.columns else np.ones_like(lows))
    nt    = (df_day.get("numTrades", None).to_numpy(dtype=float)
             if "numTrades" in df_day.columns else np.ones_like(lows))

    hist = np.zeros(len(bins)-1, dtype=float)
    for l, h, q, n in zip(lows, highs, qv, nt):
        if h < bins[0] or l > bins[-1]: continue
        i0 = max(0, np.searchsorted(bins, l, side='right')-1)
        i1 = min(len(bins)-1, np.searchsorted(bins, h, side='left'))
        if i1 <= i0: continue
        w = (q**0.5) * (max(1.0, n)**0.5)
        hist[i0:i1] += w
    return hist

def max_drawdown_path(vals):
    import numpy as np
    if len(vals) == 0:
        return 0.0
    cs = np.cumsum(np.asarray(vals, dtype=float))
    peak = -1e18
    mdd = 0.0
    for x in cs:
        if x > peak: peak = x
        dd = peak - x
        if dd > mdd: mdd = dd
    return float(mdd)

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

def compute_bs_for_csv_bytes(csv_bytes: bytes, pair_name: str):
    import numpy as np
    import pandas as pd

    # --- load ---
    df = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"closeTimeISO", "low", "high", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pair_name}: CSV musí obsahovat {required}, mám {set(df.columns)}")

    df["closeTimeISO"] = pd.to_datetime(df["closeTimeISO"], errors="coerce", utc=True)
    df = df.dropna(subset=["closeTimeISO", "low", "high", "close"])
    df["date"] = df["closeTimeISO"].dt.date

    days = sorted(df["date"].unique())
    if len(days) < MIN_DAYS_REQUIRED:
        return None

    last_days = days[-N_DAYS:] if len(days) >= N_DAYS else days
    history = []      # [(L,H, date, df_day, stats_day)]
    stats_win = []    # denní statistiky pro klasifikaci a adaptivní pravidla

    for d in last_days:
        sub = df[df["date"] == d]
        L = sub["low"].to_numpy(dtype=float)
        H = sub["high"].to_numpy(dtype=float)
        if len(L) == 0:
            continue
        st = daily_stats_1m(sub)
        history.append((L, H, d, sub, st))
        stats_win.append(st)

    if len(history) < MIN_DAYS_REQUIRED:
        return None

    # --- time-decay váhy na dny (nejnovější den má nejvyšší váhu) ---
    w_days = exp_decay_weights(len(history), HALF_LIFE_DAYS)

    # --- profil (log-biny + vážení time*volume + time-decay dny) ---
    p_min = min(float(l.min()) for l, _, _, _, _ in history)
    p_max = max(float(h.max()) for _, h, _, _, _ in history)
    bins = make_log_bins(p_min, p_max, BIN_BPS)
    profile = np.zeros(len(bins)-1, dtype=float)

    daily_closes = []
    daily_opens  = []
    daily_highs  = []
    daily_lows   = []
    for idx, (L, H, _, df_day, _) in enumerate(history):
        prof_d = build_profile_weighted(df_day, bins)
        profile += w_days[idx] * prof_d
        c = float(np.mean((L + H)/2.0))
        daily_closes.append(c)
        daily_opens.append(float((L[0] + H[0]) / 2.0))
        daily_highs.append(float(H.max()))
        daily_lows.append(float(L.min()))

    profile_smooth = gaussian_smooth_1d(profile, SMOOTH_SIGMA_BINS)

    # --- režim trhu podle denních statistik ---
    regime = classify_regime(stats_win)

    # --- predikce strany: featury + LR s váhami w_days ---
    opens  = np.array(daily_opens, dtype=float)
    highs  = np.array(daily_highs, dtype=float)
    lows   = np.array(daily_lows, dtype=float)
    closes = np.array(daily_closes, dtype=float)

    feats_basic = price_only_features(list(opens), list(highs), list(lows), list(closes))
    imb = [s["imbalance_d"] for s in stats_win]
    atr = [s["atr_d"] for s in stats_win]
    rv  = [s["rv_d"] for s in stats_win]
    nt  = [s["avg_nt"] for s in stats_win]
    qv  = [s["avg_qv"] for s in stats_win]
    def _nz(x): return x if abs(x) < 1e6 else 0.0
    X_ext = [tuple(list(f) + [_nz(imb[i]-0.5), _nz(atr[i]), _nz(rv[i]), _nz(nt[i]**0.5), _nz(qv[i]**0.5)])
             for i, f in enumerate(feats_basic)]

    y_dir = (np.roll(closes, -1) > closes).astype(int).tolist()[:-1]
    X_ext = X_ext[:-1]
    w_trn = w_days[:-1].tolist()

    if len(y_dir) < 5:
        side = "both"
        p_up = 0.5
    else:
        w_lr = logistic_regression_irls(X_ext, y_dir, sample_weight=w_trn)
        p_up = predict_proba_logreg(w_lr, X_ext[-1])
        side = "long" if p_up >= 0.6 else "short" if p_up <= 0.4 else "both"

    # --- adaptivní GAP a náklady z posledního dne (a režimu) ---
    stats_last = stats_win[-1]
    close_last = closes[-1]
    gap_min_abs = max(stats_last["atr_d"] * 0.5, close_last * (GAP_MIN_BPS / 10000.0))
    if regime == "trend":
        gap_min_abs *= 1.2
    elif regime == "vol":
        gap_min_abs *= 1.5
    nt_last = max(stats_last["avg_nt"], 1.0)
    costs_pct_eff = BASE_COSTS_PCT * ((stats_last["avg_nt"] / nt_last) ** 0.5)
    costs_pct_eff = float(min(max(costs_pct_eff, 0.5*BASE_COSTS_PCT), 2.0*BASE_COSTS_PCT))

    # --- kandidátní úrovně z peaků profilu (okolo aktuální ceny) ---
    peak_idx = find_local_peaks(profile_smooth, k_peaks=PEAKS_K, min_separation=2)
    levels = [bin_center(bins, i) for i in peak_idx]
    current_price = close_last

    lower = sorted([lv for lv in levels if lv <= current_price], key=lambda x: abs(x - current_price))[:12]
    upper = sorted([lv for lv in levels if lv >= current_price], key=lambda x: abs(x - current_price))[:12]
    pairs = [(B, S) for B in lower for S in upper if (S - B) >= gap_min_abs]
    if not pairs:
        lower = sorted([lv for lv in levels if lv <= current_price])[:30]
        upper = sorted([lv for lv in levels if lv >= current_price])[:30]
        pairs = [(B, S) for B in lower for S in upper if (S - B) >= gap_min_abs]
        if not pairs:
            return None
    n_candidates = len(pairs)

    # --- evaluace kandidátů s time-decay vahami a utility ---
    def eval_pair(B, S):
        base = max(((B + S) / 2.0), 1e-12)
        costs_abs = costs_pct_eff * base
        pnls = []
        cycles = []
        for (L, H, _, _, _), wd in zip(history, w_days):
            p, c = simulate_day_no_timeout_side(B, S, L, H, costs_abs=costs_abs, side=side)
            pnls.append(p * wd)     # vážený PnL (pro utility)
            cycles.append(c)        # POZOR: raw cykly (bez vah) -> pro total_cycles
        pnls_w = pnls                  # už vážené
        med = float(np.median(pnls_w))
        iqr = float(np.percentile(pnls_w, 75) - np.percentile(pnls_w, 25))
        maxdd = max_drawdown_path(pnls_w)
        utility = med - LAMBDA_IQR * iqr - MU_MAXDD * maxdd

        # metriky cyklů: RAW (int) + vážené (float)
        total_cycles_raw = int(np.sum(cycles))
        total_cycles_w = float(np.sum(np.array(cycles, dtype=float) * w_days))

        avg_pnl_pct = float((np.mean(pnls_w) / base) * 100.0)
        median_pnl_pct = float((med / base) * 100.0)
        gap_pct = (S - B) / max(B, 1e-12)
        dist_rel = abs(((B + S) / 2.0) - current_price) / max(current_price, 1e-12)
        return {
            "utility": utility,
            "total_cycles": total_cycles_raw,
            "total_cycles_w": total_cycles_w,
            "gap_pct": gap_pct,
            "dist": dist_rel,
            "avg_pnl_pct": avg_pnl_pct,
            "median_pnl_pct": median_pnl_pct
        }

    best = None
    best_key = None
    eval_cache = {}
    for (B, S) in pairs:
        met = eval_pair(B, S)
        eval_cache[(B, S)] = met
        # Primárně: vyšší utility; pak více RAW cyklů; menší gap; blíž k ceně
        key = (met["utility"], met["total_cycles"], -met["gap_pct"], -met["dist"])
        if best is None or key > best_key:
            best = (B, S)
            best_key = key

    if best is None:
        return None

    bestB, bestS = best
    met = eval_cache[best]
    gap_pct = met["gap_pct"] * 100.0

    # Diagnostika za celé okno (nevážené metriky pro rozšířený zápis)
    base = max(((bestB + bestS)/2.0), 1e-12)
    costs_abs_best = costs_pct_eff * base
    pnls_day = []
    day_cycles = []
    day_hit = []
    for (L, H, _, _, _), wd in zip(history, w_days):
        p, c = simulate_day_no_timeout_side(bestB, bestS, L, H, costs_abs=costs_abs_best, side=side)
        pnls_day.append(p)
        day_cycles.append(c)
        day_hit.append(1 if c > 0 else 0)
    pnls_arr = np.array(pnls_day, dtype=float)
    iqr_pnl = float(np.percentile(pnls_arr, 75) - np.percentile(pnls_arr, 25))
    std_pnl = float(np.std(pnls_arr))
    pnl_p05 = float(np.percentile(pnls_arr, 5))
    pnl_p95 = float(np.percentile(pnls_arr, 95))
    hit_rate_days = int(np.sum(np.array(day_hit)))
    hit_rate_w = float(np.sum(np.array(day_hit) * w_days))

    return {
        "pair": pair_name,
        "B": float(bestB),
        "S": float(bestS),
        "gap_pct": float(gap_pct),
        "model": MODEL_NAME,
        "regime": regime,
        "total_cycles": int(met["total_cycles"]),             # RAW integer
        "total_cycles_w": float(met["total_cycles_w"]),       # vážený součet
        "utility": float(met["utility"]),
        "score": float(met["utility"]),  # pro kompatibilitu
        "avg_pnl_pct": float(met["avg_pnl_pct"]),
        "median_pnl_pct": float(met["median_pnl_pct"]),
        "side": side,
        "p_up": float(p_up),
        "iqr_pnl": float(iqr_pnl),
        "std_pnl": float(std_pnl),
        "pnl_p05": float(pnl_p05),
        "pnl_p95": float(pnl_p95),
        "hit_rate_days": int(hit_rate_days),
        "hit_rate_w": float(hit_rate_w),
        "dist_to_price": float(met["dist"]),
        "min_gap_abs": float(gap_min_abs),
        "costs_abs": float(costs_abs_best),
        "n_candidates": int(n_candidates),
        "bin_bps": float(BIN_BPS),
        "smooth_sigma_bins": float(SMOOTH_SIGMA_BINS),
        "hl_days": float(HALF_LIFE_DAYS),
        "lambda_iqr": float(LAMBDA_IQR),
        "mu_maxdd": float(MU_MAXDD),
    }

# ============ ENTRYPOINT ============
def main(myTimer):
    try:
        import pandas as pd
        from azure.storage.blob import BlobServiceClient
    except Exception:
        logging.exception("[BSWeightedUtility] Import balíčků selhal.")
        return

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    load_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    func_name = "BSWeightedUtilityRecalc"

    logging.info(
        f"[{func_name}] Start {date_str} UTC | Python={sys.version.split()[0]} | "
        f"N_DAYS={N_DAYS} MIN_DAYS_REQUIRED={MIN_DAYS_REQUIRED} "
        f"COSTS_PCT={BASE_COSTS_PCT} GAP_MIN_BPS={GAP_MIN_BPS} BIN_BPS={BIN_BPS} | "
        f"IN_CONTAINER={IN_CONTAINER} OUT_CONTAINER={OUT_CONTAINER} MODEL={MODEL_NAME}"
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
                "total_cycles": int(res.get("total_cycles", 0)),          # RAW int
                "total_cycles_w": float(res.get("total_cycles_w", 0.0)),  # vážený float
                "score": float(res.get("score", 0.0)),                    # = utility
                "utility": float(res.get("utility", 0.0)),
                "avg_pnl_pct": float(res.get("avg_pnl_pct", 0.0)),
                "median_pnl_pct": float(res.get("median_pnl_pct", 0.0)),
                "load_time_utc": load_time,
                "side": res.get("side"),
                "p_up": float(res.get("p_up", 0.5)),
                "iqr_pnl": float(res.get("iqr_pnl", 0.0)),
                "std_pnl": float(res.get("std_pnl", 0.0)),
                "pnl_p05": float(res.get("pnl_p05", 0.0)),
                "pnl_p95": float(res.get("pnl_p95", 0.0)),
                "hit_rate_days": int(res.get("hit_rate_days", 0)),
                "hit_rate_w": float(res.get("hit_rate_w", 0.0)),
                "dist_to_price": float(res.get("dist_to_price", 0.0)),
                "min_gap_abs": float(res.get("min_gap_abs", 0.0)),
                "costs_abs": float(res.get("costs_abs", 0.0)),
                "n_candidates": int(res.get("n_candidates", 0)),
                "regime": res.get("regime", None),
                "bin_bps": float(res.get("bin_bps", 0.0)),
                "smooth_sigma_bins": float(res.get("smooth_sigma_bins", 0.0)),
                "hl_days": float(res.get("hl_days", 0.0)),
                "lambda_iqr": float(res.get("lambda_iqr", 0.0)),
                "mu_maxdd": float(res.get("mu_maxdd", 0.0)),
            })
            logging.info(
                f"[{func_name}] OK {pair_name} ({res['model']}): "
                f"B={res['B']:.6f}, S={res['S']:.6f}, gap={res['gap_pct']:.3f}%, "
                f"cycles_raw={int(res.get('total_cycles',0))}, cycles_w={float(res.get('total_cycles_w',0.0)):.3f}, "
                f"utility={float(res.get('utility',0.0)):.6f}, side={res.get('side')} "
                f"(p_up={float(res.get('p_up',0.5)):.3f}), regime={res.get('regime')}"
            )
        except Exception:
            logging.exception(f"[{func_name}] Chyba při zpracování {blob.name}")
            continue

    if not new_rows:
        logging.warning(f"[{func_name}] Nebyly vyprodukovány žádné výsledky.")
        return

    import pandas as pd
    cols = ["pair","B","S","gap_pct","date","model","is_active",
            "total_cycles","total_cycles_w","score","utility","avg_pnl_pct","median_pnl_pct","load_time_utc",
            "side","p_up","iqr_pnl","std_pnl","pnl_p05","pnl_p95",
            "hit_rate_days","hit_rate_w","dist_to_price","min_gap_abs","costs_abs","n_candidates",
            "regime","bin_bps","smooth_sigma_bins","hl_days","lambda_iqr","mu_maxdd"]
    new_df = pd.DataFrame(new_rows, columns=cols)

    # 3) Zapiš denní snapshot (stejné jméno jako dřív)
    daily_name = f"{DAILY_PREFIX}{date_str}.csv"
    try:
        out_container_client.get_blob_client(daily_name).upload_blob(
            new_df.to_csv(index=False).encode("utf-8"),
            overwrite=True
        )
        logging.info(f"[{func_name}] Zapsán denní snapshot: {OUT_CONTAINER}/{daily_name} ({len(new_df)} řádků).")
    except Exception:
        logging.exception(f"[{func_name}] Zápis denního snapshotu selhal.")

    # 4) Master CSV – deaktivace pouze pro shodné (pair, model)
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
                (master_df["model"] == row["model"]) &  # invalidovat pouze stejný model!
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
