import os
import io
import json
import logging
from datetime import datetime, timezone, timedelta, date
from typing import Optional, List, Tuple, Dict

import azure.functions as func

# ------------------- ENV & CONFIG -------------------

def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise RuntimeError(f"Missing env var: {name}")
    return v

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except:
        return default

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except:
        return default

# Storage & containers
WEBJOBS_CONN        = _get_env("AzureWebJobsStorage", required=True)

# MASTER (signály)
OUTPUT_CONTAINER    = _get_env("OUTPUT_CONTAINER", "models-recalc")  # container s master CSV
MASTER_CSV_NAME     = _get_env("MASTER_CSV_NAME", "bs_levels_master.csv")

# DATA (minutové CSV)
INPUT_CONTAINER     = _get_env("INPUT_CONTAINER", "market-data")

# SIM výstup (jeden CSV s obchody; idempotentně přepisovat)
SIM_OUTPUT_CONTAINER= _get_env("SIM_OUTPUT_CONTAINER", "trade-sim-logs")
SIM_OUTPUT_BLOB_NAME= _get_env("SIM_OUTPUT_BLOB_NAME", "sim_trades_{FROM}_{TO}.csv")

# Parametry simulace
ORDER_USDT          = _get_float("SIM_ORDER_USDT", 50.0)
COSTS_PCT_PER_SIDE  = _get_float("COSTS_PCT", 0.001)         # 0.1 % per side (taker)
SPREAD_BPS          = _get_int("SPREAD_BPS", 5)              # 5 bps = 0.05 %
INTRABAR_SEQUENCE   = (_get_env("INTRABAR_SEQUENCE", "optimistic") or "optimistic").lower()  # 'optimistic'|'conservative'

logger = logging.getLogger("BacktestTradeSim")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ------------------- UTIL -------------------

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _iter_days(from_d: date, to_d: date):
    d = from_d
    while d <= to_d:
        yield d
        d = d + timedelta(days=1)

# ------------------- AZURE BLOB HELPERS -------------------

def _make_blob_clients():
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
    bs = BlobServiceClient.from_connection_string(WEBJOBS_CONN)
    master_cc = bs.get_container_client(OUTPUT_CONTAINER)
    data_cc   = bs.get_container_client(INPUT_CONTAINER)
    out_cc    = bs.get_container_client(SIM_OUTPUT_CONTAINER)
    for cc in (master_cc, data_cc, out_cc):
        try:
            cc.create_container()
        except ResourceExistsError:
            pass
    return bs, master_cc, data_cc, out_cc

# ------------------- MASTER: načtení všech řádků (bez ohledu na is_active) -------------------

def _load_master_all(master_cc) -> Optional["pd.DataFrame"]:
    try:
        import pandas as pd
    except Exception as e:
        logger.error("[master] pandas import failed: %s", e)
        return None

    from azure.core.exceptions import ResourceNotFoundError
    blob = master_cc.get_blob_client(MASTER_CSV_NAME)
    try:
        raw = blob.download_blob().readall()
    except ResourceNotFoundError:
        logger.error("[master] CSV '%s' not found in container '%s'", MASTER_CSV_NAME, master_cc.container_name)
        return None

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        logger.exception("[master] Failed to parse CSV via pandas")
        return None

    req = {"pair","model","B","S","date","load_time_utc"}
    if not req.issubset(df.columns):
        logger.error("[master] Missing required columns. Have: %s", df.columns.tolist())
        return None

    # normalize
    df["pair"] = df["pair"].astype(str).str.upper()
    df["model"] = df["model"].astype(str)

    import pandas as pd
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["load_time_utc"] = pd.to_datetime(df["load_time_utc"], errors="coerce", utc=True)

    # drop řádků bez klíčových hodnot
    df = df.dropna(subset=["pair","model","B","S","date"])

    # pokud je víc řádků pro stejné (pair, model, date), ponecháme poslední podle load_time_utc
    df = df.sort_values(["pair","model","date","load_time_utc"])
    df = df.drop_duplicates(subset=["pair","model","date"], keep="last")

    return df

# ------------------- DATA: nalezení souboru pro pár -------------------

def _find_data_blob_for_pair(data_cc, pair: str) -> Optional[str]:
    """
    Najdi blob pro daný pair podle prefixu '{PAIR}_'. Preferuj '*_1m.csv'.
    Pokud nic nenajdeme, vrátíme None.
    """
    prefix = f"{pair}_"
    blobs = list(data_cc.list_blobs(name_starts_with=prefix))
    if not blobs:
        return None

    # preferuj _1m.csv (nejnovější)
    one_min = [b for b in blobs if b.name.endswith("_1m.csv")]
    if one_min:
        one_min_sorted = sorted(one_min, key=lambda x: x.last_modified or datetime.min, reverse=True)
        return one_min_sorted[0].name

    # fallback: první dostupné CSV (nejnovější)
    csvs = [b for b in blobs if b.name.lower().endswith(".csv")]
    if csvs:
        csvs_sorted = sorted(csvs, key=lambda x: x.last_modified or datetime.min, reverse=True)
        return csvs_sorted[0].name

    return None

# ------------------- DATA: načtení minut pro konkrétní den -------------------

def _load_minutes_for_day(data_cc, blob_name: str, target_day: date) -> List[Dict]:
    """
    Načti minuty pro daný den (UTC) s 'closeTimeISO','low','high' (+ volitelně close, volume).
    """
    try:
        import pandas as pd
    except Exception as e:
        logger.error("[data] pandas import failed: %s", e)
        return []

    from azure.core.exceptions import ResourceNotFoundError
    blob = data_cc.get_blob_client(blob_name)
    try:
        raw = blob.download_blob().readall()
    except ResourceNotFoundError:
        logger.error("[data] blob not found: %s", blob_name)
        return []

    df = pd.read_csv(io.BytesIO(raw))
    if "closeTimeISO" not in df.columns:
        if "closeTime" in df.columns:
            df["closeTimeISO"] = (df["closeTime"] // 1000).map(
                lambda s: datetime.utcfromtimestamp(int(s)).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        else:
            raise RuntimeError("CSV must contain 'closeTimeISO' or 'closeTime'")

    df["time"] = pd.to_datetime(df["closeTimeISO"], utc=True, errors="coerce")
    day_start = datetime(target_day.year, target_day.month, target_day.day, tzinfo=timezone.utc)
    day_end   = day_start + timedelta(days=1) - timedelta(seconds=1)
    df = df[(df["time"] >= day_start) & (df["time"] <= day_end)].copy()
    if df.empty:
        return []

    rows: List[Dict] = []
    for _, r in df.iterrows():
        low = float(r["low"]) if "low" in r and r["low"] == r["low"] else None
        high = float(r["high"]) if "high" in r and r["high"] == r["high"] else None
        close = float(r["close"]) if "close" in r and r["close"] == r["close"] else None
        volume = float(r["volume"]) if "volume" in r and r["volume"] == r["volume"] else None
        rows.append({
            "time": r["time"].to_pydatetime(),
            "low": low, "high": high, "close": close, "volume": volume
        })
    return rows

# ------------------- STATE (přenášení přes půlnoc) -------------------

def _new_state():
    return {
        "position": "flat",          # 'flat' | 'long'
        "qty": 0.0,
        "entry_px_effective": None,  # vstupní cena vč. fee
        "B_active": None,            # B pro AKTUÁLNÍ OTEVŘENÝ CYKLUS
        "S_active": None,            # S pro AKTUÁLNÍ OTEVŘENÝ CYKLUS
        "signal_date": None,         # YYYY-MM-DD ze dne vstupu
        "model": None                # model použitý pro tento cyklus
    }

# ------------------- EMIT trade řádku -------------------

def _emit_trade(trades: List[Dict], *, t: datetime, signal_date: str, model: str, pair: str,
                side: str, qty: float, px: float, fee_per_side: float, B: float, S: float,
                pnl_pct: float):
    trades.append({
        "time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_date": signal_date,
        "model": model,
        "pair": pair,
        "side": side,
        "qty": qty,
        "avg_price": px,
        "quote_usdt": (ORDER_USDT if side == "BUY" else qty * px),
        "fee": ((ORDER_USDT if side == "BUY" else qty * px) * fee_per_side),
        "fee_asset": "USDT",
        "order_id": f"SIM-{side}-{int(t.timestamp())}",
        "b_level": B,
        "s_level": S,
        "pnl_pct_since_open": pnl_pct
    })

# ------------------- SIMULACE JEDNOHO DNE S PŘENOSEM STATE -------------------

def _simulate_day_with_carry(minutes: List[Dict],
                             pair: str,
                             today_signal: Optional[Dict],   # {'B','S','model','date',...} nebo None
                             state: Dict) -> Tuple[List[Dict], Dict]:
    """
    Přenos pozice přes půlnoc:
      - SELL (pokud long) testujeme na S_active ze dne vstupu (pevné pro celý cyklus).
      - BUY otevřeme jen pokud existuje dnešní signál (date == dnešní den) a jsme flat.
      - Intrabar pořadí: SELL → BUY (konzervativní); v 'optimistic' po BUY ještě dovolíme okamžitý SELL.
    """
    fee_per_side = COSTS_PCT_PER_SIDE
    trades: List[Dict] = []
    B_today = today_signal["B"] if today_signal else None
    S_today = today_signal["S"] if today_signal else None
    model_today = today_signal["model"] if today_signal else None
    date_today = today_signal["date"] if today_signal else None

    for m in minutes:
        t = m["time"]
        low = m["low"]; high = m["high"]
        if low is None or high is None:
            continue

        # 1) SELL carry
        if state["position"] == "long" and state["S_active"] is not None and high >= state["S_active"]:
            sell_px = state["S_active"]
            pnl_pct = ((sell_px - state["entry_px_effective"]) / state["entry_px_effective"]) * 100.0
            pnl_pct -= (fee_per_side * 100.0)   # výstupní fee
            pnl_pct -= (SPREAD_BPS / 100.0)     # spread odhad

            _emit_trade(trades, t=t, signal_date=state["signal_date"], model=state["model"], pair=pair,
                        side="SELL", qty=state["qty"], px=sell_px, fee_per_side=fee_per_side,
                        B=state["B_active"], S=state["S_active"], pnl_pct=pnl_pct)

            # reset stavu
            state["position"] = "flat"
            state["qty"] = 0.0
            state["entry_px_effective"] = None
            state["B_active"] = None
            state["S_active"] = None
            state["signal_date"] = None
            state["model"] = None

        # 2) BUY (jen pokud existuje dnešní signál a jsme flat)
        if state["position"] == "flat" and today_signal is not None and low <= B_today:
            buy_px = B_today
            qty = ORDER_USDT / max(buy_px, 1e-12)
            entry_eff = buy_px * (1.0 + fee_per_side)

            _emit_trade(trades, t=t, signal_date=date_today, model=model_today, pair=pair,
                        side="BUY", qty=qty, px=buy_px, fee_per_side=fee_per_side,
                        B=B_today, S=S_today, pnl_pct=0.0)

            # aktivní cyklus nese dnešní B/S až do SELL
            state["position"] = "long"
            state["qty"] = qty
            state["entry_px_effective"] = entry_eff
            state["B_active"] = B_today
            state["S_active"] = S_today
            state["signal_date"] = date_today
            state["model"] = model_today

            # 'optimistic': po čerstvém BUY povol i SELL, pokud dosáhneme S v téže minutě
            if INTRABAR_SEQUENCE == "optimistic" and high >= state["S_active"]:
                sell_px = state["S_active"]
                pnl_pct = ((sell_px - state["entry_px_effective"]) / state["entry_px_effective"]) * 100.0
                pnl_pct -= (fee_per_side * 100.0)
                pnl_pct -= (SPREAD_BPS / 100.0)

                _emit_trade(trades, t=t, signal_date=state["signal_date"], model=state["model"], pair=pair,
                            side="SELL", qty=state["qty"], px=sell_px, fee_per_side=fee_per_side,
                            B=state["B_active"], S=state["S_active"], pnl_pct=pnl_pct)

                # reset
                state["position"] = "flat"
                state["qty"] = 0.0
                state["entry_px_effective"] = None
                state["B_active"] = None
                state["S_active"] = None
                state["signal_date"] = None
                state["model"] = None

    return trades, state

# ------------------- ZÁPIS: idempotentní 1 CSV -------------------

def _write_trades_csv_idempotent(out_cc, blob_name: str, trades: List[Dict]) -> str:
    """
    Jeden CSV (BlockBlob, overwrite=True) s čistými trade řádky:
    time_utc,signal_date,model,pair,side,qty,avg_price,quote_usdt,fee,fee_asset,order_id,b_level,s_level,pnl_pct_since_open
    """
    from io import StringIO
    buf = StringIO()
    header = ("time_utc,signal_date,model,pair,side,qty,avg_price,quote_usdt,fee,fee_asset,order_id,b_level,s_level,pnl_pct_since_open\n")
    buf.write(header)
    for r in trades:
        buf.write(",".join([
            r.get("time_utc",""),
            r.get("signal_date",""),
            r.get("model",""),
            r.get("pair",""),
            r.get("side",""),
            f"{r.get('qty',0.0):.8f}",
            f"{r.get('avg_price',0.0):.8f}",
            f"{r.get('quote_usdt',0.0):.8f}",
            f"{r.get('fee',0.0):.8f}",
            r.get("fee_asset","USDT"),
            r.get("order_id",""),
            f"{r.get('b_level',0.0):.8f}",
            f"{r.get('s_level',0.0):.8f}",
            f"{r.get('pnl_pct_since_open',0.0):.6f}"
        ]) + "\n")

    blob = out_cc.get_blob_client(blob_name)
    blob.upload_blob(buf.getvalue().encode("utf-8"), overwrite=True)
    return blob_name

# ------------------- ENTRYPOINT (TIMER) -------------------

def main(mytimer: func.TimerRequest) -> None:
    try:
        started = _utc_now_str()
        # Init storage
        _, master_cc, data_cc, out_cc = _make_blob_clients()

        # 1) Načti master (všechny řádky, ignoruj is_active)
        df = _load_master_all(master_cc)
        if df is None or df.empty:
            logger.info("[BacktestTradeSim] Master empty — nothing to simulate.")
            return

        # 2) Rozsah dní = min/max date z masteru
        all_dates = sorted({d for d in df["date"].tolist() if isinstance(d, date)})
        if not all_dates:
            logger.info("[BacktestTradeSim] No valid dates in master.")
            return
        range_from = all_dates[0]
        range_to   = all_dates[-1]
        logger.info("[BacktestTradeSim] Tick %s; days=%s..%s; rows=%d",
                    started, range_from.isoformat(), range_to.isoformat(), len(df))

        # 3) Příprava map: (pair -> data blob) + (day -> {(pair,model)->signal})
        pairs = sorted(df["pair"].unique().tolist())
        pair_blob_map: Dict[str, Optional[str]] = {}
        for p in pairs:
            blob_name = _find_data_blob_for_pair(data_cc, p)
            if not blob_name:
                logger.warning("[data] No data blob found for pair '%s' (prefix '%s_') — will skip its trades.", p, p)
            pair_blob_map[p] = blob_name

        signals_by_day: Dict[date, Dict[Tuple[str,str], Dict]] = {}
        for d in all_dates:
            today = df[df["date"] == d].copy()
            today = today.sort_values(["pair","model","load_time_utc"])
            m: Dict[Tuple[str,str], Dict] = {}
            for _, r in today.iterrows():
                key = (str(r["pair"]).upper(), str(r["model"]))
                m[key] = {
                    "B": float(r["B"]),
                    "S": float(r["S"]),
                    "model": str(r["model"]),
                    "date": d.isoformat()
                }
            signals_by_day[d] = m

        # 4) Simulace přes dny s přenosem state per (pair,model)
        states: Dict[Tuple[str,str], Dict] = {}
        for _, row in df[["pair","model"]].drop_duplicates().iterrows():
            states[(str(row["pair"]).upper(), str(row["model"]))] = _new_state()

        all_trades: List[Dict] = []

        for d in _iter_days(range_from, range_to):
            sigs = signals_by_day.get(d, {})
            for (pair, model), state in list(states.items()):
                blob_name = pair_blob_map.get(pair)
                if not blob_name:
                    continue  # bez dat

                minutes = _load_minutes_for_day(data_cc, blob_name, d)
                if not minutes:
                    continue

                today_signal = sigs.get((pair, model))  # může být None
                trades, new_state = _simulate_day_with_carry(minutes, pair, today_signal, state)
                all_trades.extend(trades)
                states[(pair, model)] = new_state  # přeneseme do dalšího dne

        # 5) Zápis idempotentního výstupu
        out_name = SIM_OUTPUT_BLOB_NAME.format(FROM=range_from.isoformat(), TO=range_to.isoformat())
        final_blob = _write_trades_csv_idempotent(out_cc, out_name, all_trades)
        logger.info("[BacktestTradeSim] Done. Trades written: %d -> %s", len(all_trades), final_blob)

    except Exception:
        logger.exception("[BacktestTradeSim] unhandled error")
        raise
