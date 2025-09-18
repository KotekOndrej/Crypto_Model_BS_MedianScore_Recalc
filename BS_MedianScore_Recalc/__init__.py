import os
import io
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
import azure.functions as func


# =========================
# ---- Helpers: ENV -------
# =========================

def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        logging.warning(f"[ENV] {name}='{v}' není číslo, používám default {default}")
        return float(default)

def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return int(default)
    try:
        return int(v)
    except ValueError:
        logging.warning(f"[ENV] {name}='{v}' není integer, používám default {default}")
        return int(default)


# =========================
# ------- Parametry -------
# =========================
# Přes App Settings / ENV lze přepsat:
#   N_DAYS (int), MIN_DAYS_REQUIRED (int),
#   COSTS_PCT (float)  -> náklady v procentech ceny na cyklus (např. 0.001 = 0.1 %)
#   GAP_MIN_PCT (float)-> minimální gap v procentech průměrné ceny (0.003 = 0.3 %)

N_DAYS = _get_env_int("N_DAYS", 20)
MIN_DAYS_REQUIRED = _get_env_int("MIN_DAYS_REQUIRED", 12)
COSTS_PCT = _get_env_float("COSTS_PCT", 0.001)     # 0.001 => 0.1 %
GAP_MIN_PCT = _get_env_float("GAP_MIN_PCT", 0.003) # 0.003 => 0.3 %

PEAKS_K = 60
EU_TZ = ZoneInfo("Europe/Prague")
MODEL_NAME = "BS_MedianScore"

# Azure připojení
INPUT_BLOB_CONNECTION_STRING = os.getenv("INPUT_BLOB_CONNECTION_STRING")
INPUT_CONTAINER = os.getenv("INPUT_CONTAINER", "market-data")
OUTPUT_BLOB_CONNECTION_STRING = os.getenv("OUTPUT_BLOB_CONNECTION_STRING")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "market-signals")


# =========================
# ----- Utility výpočet ---
# =========================

def extract_pair_from_filename(blob_name: str) -> str:
    base = os.path.splitext(os.path.basename(blob_name))[0]
    for sep in ['_', '-']:
        if sep in base:
            return base.split(sep)[0]
    return base

def make_bins(p_min, p_max, n_bins=600):
    if p_max <= p_min:
        p_max = p_min * 1.001
    return np.linspace(p_min, p_max, n_bins + 1)

def gaussian_smooth(x, sigma_bins=3):
    if sigma_bins <= 0:
        return x
    radius = int(3 * sigma_bins)
    i = np.arange(-radius, radius + 1)
    k = np.exp(-(i * i) / (2 * sigma_bins * sigma_bins))
    k = k / k.sum()
    return np.convolve(x, k, mode='same')

def build_touch_hist_day(L, H, bins):
    hist = np.zeros(len(bins) - 1, dtype=float)
    for l, h in zip(L, H):
        if h < bins[0] or l > bins[-1]:
            continue
        i0 = max(0, np.searchsorted(bins, l, side='right') - 1)
        i1 = min(len(bins) - 1, np.searchsorted(bins, h, side='left'))
        if i1 > i0:
            hist[i0:i1] += 1.0
    return hist

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
        touched_B = (l <= B <= h)
        touched_S = (l <= S <= h)
        if pos == 0:
            if side in ("both", "long") and touched_B:
                pos = +1
            elif side in ("both", "short") and touched_S:
                pos = -1
        else:
            if pos == +1 and touched_S:
                pnl += (S - B) - costs_abs
                pos = 0
                cycles += 1
            elif pos == -1 and touched_B:
                pnl += (S - B) - costs_abs
                pos = 0
                cycles += 1
    return pnl, cycles

def rolling_mean(arr, w):
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        j0 = max(0, i - w + 1)
        out[i] = np.mean(arr[j0:i+1])
    return out

def rolling_std(arr, w):
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        j0 = max(0, i - w + 1)
        out[i] = np.std(arr[j0:i+1])
    return out

def price_only_features(opens, highs, lows, closes):
    ret_1d = np.concatenate([[0.0], np.diff(closes)/np.maximum(closes[:-1], 1e-9)])
    day_range = (highs - lows) / np.maximum(closes, 1e-9)
    close_open = (closes - opens) / np.maximum(opens, 1e-9)
    close_pos = (closes - lows) / np.maximum(highs - lows, 1e-9)
    ret_3d = rolling_mean(ret_1d, 3)
    ret_5d = rolling_mean(ret_1d, 5)
    rng_3d = rolling_mean(day_range, 3)
    rng_5d = rolling_mean(day_range, 5)
    vol_3d = rolling_std(ret_1d, 3)
    vol_5d = rolling_std(ret_1d, 5)
    X = np.column_stack([ret_1d, day_range, close_open, close_pos,
                         ret_3d, ret_5d, rng_3d, rng_5d, vol_3d, vol_5d])
    return X

def logistic_regression_irls(X, y, max_iter=100, tol=1e-6, reg_lambda=1e-4):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        H = (Xb.T * W) @ Xb + reg_lambda * np.eye(Xb.shape[1])
        g = Xb.T @ (p - y) + reg_lambda * w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w

def predict_proba_logreg(w, xrow):
    xb = np.hstack([[1.0], xrow])
    z = float(xb @ w)
    return 1.0 / (1.0 + np.exp(-z))


# =========================
# ---- Hlavní logika  -----
# =========================

def compute_bs_for_csv_bytes(csv_bytes: bytes, pair_name: str):
    """
    Vstupní CSV: sloupce closeTimeISO, low, high
    COSTS_PCT: z App Settings, např. 0.001 = 0.1 % náklad na cyklus (přepočteme na absolutní cenu).
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"closeTimeISO", "low", "high"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pair_name}: CSV musí obsahovat sloupce {required}, mám {set(df.columns)}")

    df["closeTimeISO"] = pd.to_datetime(df["closeTimeISO"], errors="coerce")
    df = df.dropna(subset=["closeTimeISO", "low", "high"])
    df["date"] = df["closeTimeISO"].dt.date

    days = sorted(df["date"].unique())
    if len(days) < MIN_DAYS_REQUIRED:
        logging.warning(f"{pair_name}: málo dnů ({len(days)}) – přeskočeno.")
        return None

    last_days = days[-N_DAYS:] if len(days) >= N_DAYS else days
    history = []
    for d in last_days:
        sub = df[df["date"] == d]
        L = sub["low"].to_numpy(dtype=float)
        H = sub["high"].to_numpy(dtype=float)
        if len(L) == 0:
            continue
        history.append((L, H))
    if len(history) < MIN_DAYS_REQUIRED:
        logging.warning(f"{pair_name}: málo použitelných dnů v okně – přeskočeno.")
        return None

    # Histogram dotyků
    p_min = min(float(l.min()) for l, _ in history)
    p_max = max(float(h.max()) for _, h in history)
    bins = make_bins(p_min, p_max, n_bins=600)
    hist = np.zeros(len(bins) - 1, dtype=float)
    for L, H in history:
        hist += build_touch_hist_day(L, H, bins)
    hist_smooth = gaussian_smooth(hist, sigma_bins=3)

    # Denní OHLC proxy + featury
    daily_closes = [float(np.mean((L + H) / 2)) for (L, H) in history]
    daily_opens  = [float((L[0] + H[0]) / 2)    for (L, H) in history]
    daily_highs  = [float(H.max())              for (_, H) in history]
    daily_lows   = [float(L.min())              for (L, _) in history]

    closes = np.array(daily_closes, dtype=float)
    opens  = np.array(daily_opens, dtype=float)
    highs  = np.array(daily_highs, dtype=float)
    lows   = np.array(daily_lows, dtype=float)

    X = price_only_features(opens, highs, lows, closes)
    y = (np.roll(closes, -1) > closes).astype(int)[:-1]
    X = X[:-1, :]

    # Režim dne
    if len(y) < 5:
        side = "both"
    else:
        w = logistic_regression_irls(X[:-1], y[:-1])
        p_up = predict_proba_logreg(w, X[-1])
        side = "long" if p_up >= 0.6 else "short" if p_up <= 0.4 else "both"

    # Kandidáti levelů
    peak_idx = find_local_peaks(hist_smooth, k_peaks=PEAKS_K, min_separation=2)
    levels = [bin_center(bins, i) for i in peak_idx]
    current_price = closes[-1]
    avg_price = float(np.mean([np.mean((L + H) / 2) for (L, H) in history]))
    min_gap_abs = GAP_MIN_PCT * avg_price

    lower = sorted([lv for lv in levels if lv <= current_price], key=lambda x: abs(x - current_price))[:12]
    upper = sorted([lv for lv in levels if lv >= current_price], key=lambda x: abs(x - current_price))[:12]

    pairs = [(B, S) for B in lower for S in upper if (S - B) >= min_gap_abs]
    if not pairs:
        # fallback: větší výběr
        lower = sorted([lv for lv in levels if lv <= current_price])[:30]
        upper = sorted([lv for lv in levels if lv >= current_price])[:30]
        pairs = [(B, S) for B in lower for S in upper if (S - B) >= min_gap_abs]
        if not pairs:
            logging.warning(f"{pair_name}: žádné páry nesplňují gap ≥ {GAP_MIN_PCT*100:.2f}% – přeskočeno.")
            return None

    # Ohodnocení kandidátů přes celé okno (bez timeoutu)
    best = None
    for (B, S) in pairs:
        costs_abs = COSTS_PCT * ((B + S) / 2.0)  # náklad v absolutní ceně na cyklus
        pnls, cyc = [], []
        for L, H in history:
            p, c = simulate_day_no_timeout_side(B, S, L, H, costs_abs=costs_abs, side=side)
            pnls.append(p)
            cyc.append(c)
        pnls = np.array(pnls, dtype=float)
        med = float(np.median(pnls))
        iqr = float(np.percentile(pnls, 75) - np.percentile(pnls, 25))
        score = med - 0.25 * iqr
        cand = (score, B, S)
        if (best is None) or (cand > best):
            best = cand

    if best is None:
        logging.error(f"{pair_name}: nepodařilo se vybrat nejlepší pár.")
        return None

    _, bestB, bestS = best
    gap_pct = 100.0 * (bestS - bestB) / max(bestB, 1e-12)

    return {
        "pair": pair_name,
        "B": bestB,
        "S": bestS,
        "gap_pct": gap_pct,
        "model": MODEL_NAME
    }


# =========================
# ------ Entry point ------
# =========================

def main(myTimer: func.TimerRequest) -> None:
    """
    Entry point funkce BSMedianScoreRecalc (timer trigger je definován ve function.json).
    """
    now_prg = datetime.now(tz=EU_TZ)
    date_str = now_prg.strftime("%Y-%m-%d")
    func_name = "BSMedianScoreRecalc"

    logging.info(f"[{func_name}] Start {date_str} | N_DAYS={N_DAYS} MIN_DAYS_REQUIRED={MIN_DAYS_REQUIRED} COSTS_PCT={COSTS_PCT} GAP_MIN_PCT={GAP_MIN_PCT}")

    in_client = BlobServiceClient.from_connection_string(IN_CONN_STR)
    out_client = BlobServiceClient.from_connection_string(OUT_CONN_STR)

    in_container = in_client.get_container_client(IN_CONTAINER)
    out_container = out_client.get_container_client(OUT_CONTAINER)
    try:
        out_container.create_container()
    except Exception:
        pass

    rows = []
    for blob in in_container.list_blobs():
        if not blob.name.lower().endswith(".csv"):
            continue
        pair_name = extract_pair_from_filename(blob.name)
        try:
            csv_bytes = in_container.get_blob_client(blob).download_blob().readall()
            res = compute_bs_for_csv_bytes(csv_bytes, pair_name)
            if res is None:
                continue
            rows.append({
                "pair": res["pair"],
                "B": f"{res['B']:.10f}",
                "S": f"{res['S']:.10f}",
                "gap_pct": f"{res['gap_pct']:.4f}",
                "date": date_str,
                "model": res["model"]
            })
            logging.info(f"[{func_name}] {pair_name}: B={float(res['B']):.6f}, S={float(res['S']):.6f}, gap={float(res['gap_pct']):.3f}%")
        except Exception as e:
            logging.exception(f"[{func_name}] Chyba při zpracování {pair_name}: {e}")

    if not rows:
        logging.warning(f"[{func_name}] Nebyly vyprodukovány žádné výsledky (žádné validní páry).")
        return

    out_df = pd.DataFrame(rows, columns=["pair", "B", "S", "gap_pct", "date", "model"])
    out_csv = out_df.to_csv(index=False).encode("utf-8")

    out_name = f"bs_levels_{date_str}.csv"
    out_blob = out_client.get_container_client(OUT_CONTAINER).get_blob_client(out_name)
    out_blob.upload_blob(out_csv, overwrite=True)

    logging.info(f"[{func_name}] Zapsáno do {OUT_CONTAINER}/{out_name} ({len(rows)} řádků).")
