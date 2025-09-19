import os
import io
import sys
import logging
from datetime import datetime, timezone

# ============ ENV helpers (jen stdlib) ============

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

# ============ Parametry/konstanty (z ENV, s defaulty) ============
N_DAYS = _get_env_int("N_DAYS", 20)
MIN_DAYS_REQUIRED = _get_env_int("MIN_DAYS_REQUIRED", 12)
COSTS_PCT = _get_env_float("COSTS_PCT", 0.001)     # 0.001 => 0.1 % na cyklus
GAP_MIN_PCT = _get_env_float("GAP_MIN_PCT", 0.003) # 0.003 => 0.3 %

PEAKS_K = 60
MODEL_NAME = "BS_MedianScore"

# Storage připojení (původní názvy proměnných)
IN_CONN_STR  = os.getenv("INPUT_BLOB_CONNECTION_STRING")
OUT_CONN_STR = os.getenv("OUTPUT_BLOB_CONNECTION_STRING")
IN_CONTAINER  = os.getenv("INPUT_CONTAINER", "market-data")
OUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "market-signals")

# Výstupní soubory
MASTER_CSV_NAME = "bs_levels_master.csv"      # historie + is_active
DAILY_PREFIX    = "bs_levels_"                # denní snapshot

# ============ Utility (nevyžadují import 3rd party při def.) ============

def extract_pair_from_filename(blob_name: str) -> str:
    base = os.path.splitext(os.path.basename(blob_name))[0]
    for sep in ['_', '-']:
        if sep in base:
            return base.split(sep)[0]
    return base

def make_bins(p_min, p_max, n_bins=600):
    if p_max <= p_min:
        p_max = p_min * 1.001
    # np.linspace použijeme až runtime – zde jen signatura

def gaussian_smooth(x, sigma_bins=3):
    # implementace až po importu numpy v main()
    pass

def build_touch_hist_day(L, H, bins):
    # implementace až po importu numpy v main()
    pass

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
    # použijeme čistý Python; konverzi na numpy uděláme až v simulaci
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
    X = list(zip(ret_1d, day_range, close_open, close_pos,
                 ret_3d, ret_5d, rng_3d, rng_5d, vol_3d, vol_5d))
    return X

def logistic_regression_irls(X, y, max_iter=100, tol=1e-6, reg_lambda=1e-4):
    # jednoduchá IRLS nad python listy (převedeme na float matice uvnitř)
    import math

    n = len(X)
    if n == 0:
        return [0.0] * (len(X[0]) + 1)
    d = len(X[0])
    # Xb = [1|X]
    Xb = [[1.0] + list(row) for row in X]
    w = [0.0] * (d + 1)

    def dot(u, v):
        return sum(ui*vi for ui,vi in zip(u,v))

    for _ in range(max_iter):
        z = [dot(Xb[i], w) for i in range(n)]
        p = [1.0 / (1.0 + math.exp(-zi)) for zi in z]
        W = [pi * (1 - pi) for pi in p]
        # H = Xb^T * W * Xb + lambda*I
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
        # vyřeš H*step = g
        # jednoduché Gaussovo řešení
        # zkopíruj H a g do rozšířené matice
        A = [row[:] + [g[i]] for i, row in enumerate(H)]
        # Gaussova eliminace
        for col in range(d+1):
            # pivot
            piv = col
            for r in range(col+1, d+1):
                if abs(A[r][col]) > abs(A[piv][col]):
                    piv = r
            A[col], A[piv] = A[piv], A[col]
            pivot = A[col][col] or 1e-12
            # normalizace
            for c in range(col, d+2):
                A[col][c] /= pivot
            # vynuluj ostatní řádky v tomto sloupci
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

def compute_bs_for_csv_bytes(csv_bytes: bytes, pair_name: str):
    """
    Vstupní CSV: sloupce closeTimeISO, low, high
    COSTS_PCT: např. 0.001 = 0.1 % náklad/cyklus (přepočet na absolutní cenu dle středu páru)
    """
    # --- IMPORTY 3rd party AŽ TADY (aby případný problém šel vidět v logu) ---
    import numpy as np
    import pandas as pd

    # --- lokální implementace funkcí závislých na numpy ---
    def _make_bins(p_min, p_max, n_bins=600):
        if p_max <= p_min:
            p_max = p_min * 1.001
        return np.linspace(p_min, p_max, n_bins + 1)

    def _gaussian_smooth(x, sigma_bins=3):
        if sigma_bins <= 0:
            return x
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

    # --- načtení dat ---
    df = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"closeTimeISO", "low", "high"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pair_name}: CSV musí obsahovat sloupce {required}, mám {set(df.columns)}")

    df["closeTimeISO"] = pd.to_datetime(df["closeTimeISO"], errors="coerce", utc=True)
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
    bins = _make_bins(p_min, p_max, n_bins=600)
    hist = np.zeros(len(bins) - 1, dtype=float)
    for L, H in history:
        hist += _build_touch_hist_day(L, H, bins)
    hist_smooth = _gaussian_smooth(hist, sigma_bins=3)

    # Denní OHLC proxy + featury
    daily_closes = [float(np.mean((L + H) / 2)) for (L, H) in history]
    daily_opens  = [float((L[0] + H[0]) / 2)    for (L, H) in history]
    daily_highs  = [float(H.max())              for (_, H) in history]
    daily_lows   = [float(L.min())              for (L, _) in history]

    closes = np.array(daily_closes, dtype=float)
    opens  = np.array(daily_opens, dtype=float)
    highs  = np.array(daily_highs, dtype=float)
    lows   = np.array(daily_lows, dtype=float)

    X = price_only_features(list(opens), list(highs), list(lows), list(closes))
    # label: close_{t+1} > close_t
    y = (np.roll(closes, -1) > closes).astype(int).tolist()[:-1]
    X = X[:-1]

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
        lower = sorted([lv for lv in levels if lv <= current_price])[:30]
        upper = sorted([lv for lv in levels if lv >= current_price])[:30]
        pairs = [(B, S) for B in lower for S in upper if (S - B) >= min_gap_abs]
        if not pairs:
            logging.warning(f"{pair_name}: žádné páry nesplňují gap ≥ {GAP_MIN_PCT*100:.2f}% – přeskočeno.")
            return None

    # Ohodnocení kandidátů přes celé okno (bez timeoutu)
    best = None
    for (B, S) in pairs:
        costs_abs = COSTS_PCT * ((B + S) / 2.0)
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

# ============ ENTRYPOINT ============

def main(myTimer):
    """
    Timer trigger definován ve function.json.
    - Vytvoří denní snapshot s is_active=1 pro nové záznamy.
    - Udržuje master CSV (bs_levels_master.csv) s jediným aktivním záznamem pro (pair, model).
    """
    # Importy třetích stran až tady: když by něco chybělo, uvidíme to v logu.
    try:
        import numpy as np
        import pandas as pd
        from azure.storage.blob import BlobServiceClient
    except Exception:
        logging.exception("[BSMedianScoreRecalc] Import balíčků selhal (numpy/pandas/azure-storage-blob).")
        return

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    func_name = "BSMedianScoreRecalc"

    logging.info(
        f"[{func_name}] Start {date_str} UTC | Python={sys.version.split()[0]} "
        f"| N_DAYS={N_DAYS} MIN_DAYS_REQUIRED={MIN_DAYS_REQUIRED} "
        f"COSTS_PCT={COSTS_PCT} GAP_MIN_PCT={GAP_MIN_PCT} | "
        f"IN_CONTAINER={IN_CONTAINER} OUT_CONTAINER={OUT_CONTAINER}"
    )

    # Validace připojení
    missing = []
    if not IN_CONN_STR:  missing.append("INPUT_BLOB_CONNECTION_STRING")
    if not OUT_CONN_STR: missing.append("OUTPUT_BLOB_CONNECTION_STRING")
    if not IN_CONTAINER:  missing.append("INPUT_CONTAINER")
    if not OUT_CONTAINER: missing.append("OUTPUT_CONTAINER")
    if missing:
        logging.error(f"[{func_name}] Chybí App Settings: {', '.join(missing)}")
        return

    # Blob klienti
    try:
        in_client  = BlobServiceClient.from_connection_string(IN_CONN_STR)
        out_client = BlobServiceClient.from_connection_string(OUT_CONN_STR)
    except Exception:
        logging.exception(f"[{func_name}] Nelze vytvořit BlobServiceClient – zkontroluj connection stringy.")
        return

    in_container_client  = in_client.get_container_client(IN_CONTAINER)
    out_container_client = out_client.get_container_client(OUT_CONTAINER)

    # vytvoř výstupní container (pokud neexistuje)
    try:
        out_container_client.create_container()
    except Exception:
        pass

    # 1) Načti vstupní CSV soubory
    try:
        blobs = list(in_container_client.list_blobs())
        logging.info(f"[{func_name}] Nalezeno {len(blobs)} objektů v '{IN_CONTAINER}'.")
    except Exception:
        logging.exception(f"[{func_name}] Chyba při listování blobů v '{IN_CONTAINER}'.")
        return

    # 2) Spočítej nové řádky
    new_rows = []
    for blob in blobs:
        if not blob.name.lower().endswith(".csv"):
            continue
        pair_name = extract_pair_from_filename(blob.name)
        try:
            csv_bytes = in_container_client.get_blob_client(blob).download_blob().readall()
            res = compute_bs_for_csv_bytes(csv_bytes, pair_name)
            if res is None:
                logging.warning(f"[{func_name}] Přeskakuji {blob.name} – žádný výsledek (málo dat / žádné páry).")
                continue
            new_rows.append({
                "pair": res["pair"],
                "B": float(res["B"]),
                "S": float(res["S"]),
                "gap_pct": float(res["gap_pct"]),
                "date": date_str,
                "model": res["model"],
                "is_active": True
            })
            logging.info(f"[{func_name}] OK {pair_name}: B={res['B']:.6f}, S={res['S']:.6f}, gap={res['gap_pct']:.3f}%")
        except Exception:
            logging.exception(f"[{func_name}] Chyba při zpracování {blob.name}")
            continue

    if not new_rows:
        logging.warning(f"[{func_name}] Nebyly vyprodukovány žádné výsledky (žádné validní páry / žádná CSV).")
        return

    import pandas as pd  # jistota v této scope
    new_df = pd.DataFrame(new_rows, columns=["pair", "B", "S", "gap_pct", "date", "model", "is_active"])

    # 3) Zapiš denní snapshot
    daily_name = f"{DAILY_PREFIX}{date_str}.csv"
    try:
        out_container_client.get_blob_client(daily_name).upload_blob(
            new_df.to_csv(index=False).encode("utf-8"),
            overwrite=True
        )
        logging.info(f"[{func_name}] Zapsán denní snapshot: {OUT_CONTAINER}/{daily_name} ({len(new_df)} řádků).")
    except Exception:
        logging.exception(f"[{func_name}] Zápis denního snapshotu selhal.")

    # 4) Master CSV – vždy jen jeden aktivní záznam na (pair, model)
    master_cols = ["pair", "B", "S", "gap_pct", "date", "model", "is_active"]
    try:
        master_blob = out_container_client.get_blob_client(MASTER_CSV_NAME)
        if master_blob.exists():
            master_bytes = master_blob.download_blob().readall()
            master_df = pd.read_csv(io.BytesIO(master_bytes))
            # doplň chybějící sloupce
            for c in master_cols:
                if c not in master_df.columns:
                    master_df[c] = False if c == "is_active" else None
            master_df = master_df[master_cols]
        else:
            master_df = pd.DataFrame(columns=master_cols)
    except Exception:
        logging.exception(f"[{func_name}] Chyba při načtení master CSV – vytvořím nový.")
        master_df = pd.DataFrame(columns=master_cols)

    # deaktivuj staré aktivní pro stejné (pair, model)
    if not master_df.empty:
        for _, row in new_df.iterrows():
            mask = (master_df["pair"] == row["pair"]) & (master_df["model"] == row["model"]) & (master_df["is_active"] == True)
            if mask.any():
                master_df.loc[mask, "is_active"] = False

    # přidej nové aktivní řádky
    master_df = pd.concat([master_df, new_df], ignore_index=True)

    # ulož master zpět
    try:
        out_container_client.get_blob_client(MASTER_CSV_NAME).upload_blob(
            master_df.to_csv(index=False).encode("utf-8"),
            overwrite=True
        )
        logging.info(f"[{func_name}] Aktualizován master: {OUT_CONTAINER}/{MASTER_CSV_NAME} (n={len(master_df)}).")
    except Exception:
        logging.exception(f"[{func_name}] Zápis master CSV selhal.")
