import logging
import os
from io import BytesIO
from datetime import datetime, date, timedelta
from typing import Optional, Dict

import pandas as pd
from azure.storage.blob import BlobServiceClient
import azure.functions as func

# ==============================================
# Function: Build 3 CSVs for daily HIGH signals
#   - Notify_High_5D.csv  (window: 15d, min N: 10)
#   - Notify_High_1M.csv  (window: 90d, min N: 60)
#   - Notify_High_3M.csv  (window: 270d, min N: 180)
# Source: CoinDeskModelsEvaluation.csv in models-recalc
# Rules:
#   - take ONLY today's rows (Europe/Prague): scrape_date == today
#   - hit_type == 'high'
#   - dedup last valid row per (symbol, horizon)
#   - compute historical HIGH metrics from window = horizon * 3
#   - include rows only if HIGH count in window >= horizon * 2
# Output columns per row (and append-only behavior):
#   symbol, horizon, current_price, predicted_price, predicted_change_pct,
#   high_period_success_rate, high_period_count, scrape_date, model_to, generated_at (UTC)
#   Existing Notify_High_*.csv files are LOADED and we APPEND only new (symbol,horizon,scrape_date)
# ==============================================

CONTAINER_MODELS = "models-recalc"
EVALUATION_BLOB_NAME = "CoinDeskModelsEvaluation.csv"
NOTIFY_5D_BLOB = "Notify_High_5D.csv"
NOTIFY_1M_BLOB = "Notify_High_1M.csv"
NOTIFY_3M_BLOB = "Notify_High_3M.csv"

# Notify columns
COL_GENERATED_AT = "generated_at"

# Timezone handling
try:
    from zoneinfo import ZoneInfo
    TIMEZONE = ZoneInfo("Europe/Prague")
except Exception:
    TIMEZONE = None

# Column names (as in Evaluation)
COL_TOKEN = "symbol"
COL_HORIZON = "horizon"
COL_PCT = "predicted_change_pct"
COL_PRICE = "predicted_price"
COL_START = "scrape_date"
COL_END = "model_to"
COL_CURRENT_PRICE = "current_price"
COL_HIT_TYPE = "hit_type"
COL_VALIDATED = "validated"
COL_VALIDATION_CLOSED = "validation_closed"
COL_VALIDATED_ON = "validated_on"  # may be NaN
COL_LAST_RUN = "last_run_utc"       # may be missing

# Horizon config: window = interval*3, min_count = interval*2
HORIZON_CFG: Dict[str, Dict[str, int]] = {
    "5D": {"window_days": 15,  "min_count": 10},
    "1M": {"window_days": 90,  "min_count": 60},
    "3M": {"window_days": 270, "min_count": 180},
}


def _now_prg_date() -> date:
    if TIMEZONE is not None:
        try:
            return datetime.now(TIMEZONE).date()
        except Exception:
            return datetime.utcnow().date()
    return datetime.utcnow().date()


def _get_blob_service() -> BlobServiceClient:
    conn = os.getenv("AzureWebJobsStorage")
    if not conn:
        logging.error("AzureWebJobsStorage env var is not set.")
        raise RuntimeError("AzureWebJobsStorage env var is not set.")
    return BlobServiceClient.from_connection_string(conn)


def _download_csv_as_df(container_client, blob_name: str) -> Optional[pd.DataFrame]:
    try:
        stream = container_client.get_blob_client(blob_name).download_blob(max_concurrency=1).readall()
        try:
            df = pd.read_csv(BytesIO(stream), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(stream), encoding="utf-8")
        return df
    except Exception as e:
        logging.warning(f"Cannot download '{blob_name}': {e}")
        return None


def _upload_df_as_csv(container_client, blob_name: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    container_client.upload_blob(name=blob_name, data=csv_bytes, overwrite=True)
    logging.info(f"Uploaded {blob_name} ({len(df)} rows)")

def _append_notify_csv(container_client, blob_name: str, new_df: pd.DataFrame, key_cols=None) -> None:
    """Append-only write: load existing CSV if present, keep existing rows,
    add rows from new_df that are not present by key.
    key_cols default: [symbol, horizon, scrape_date]
    """
    if new_df is None or new_df.empty:
        # Ensure file exists with headers if it's missing
        _tmp_existing = _download_csv_as_df(container_client, blob_name)
        if _tmp_existing is None:
            _upload_df_as_csv(container_client, blob_name, pd.DataFrame(columns=new_df.columns if new_df is not None else []))
        return

    if key_cols is None:
        key_cols = [COL_TOKEN, COL_HORIZON, COL_START]
    _tmp_existing = _download_csv_as_df(container_client, blob_name)
    existing = _tmp_existing if _tmp_existing is not None else pd.DataFrame(columns=new_df.columns)
    # Ensure the same columns order
    for c in new_df.columns:
        if c not in existing.columns:
            existing[c] = None
    for c in existing.columns:
        if c not in new_df.columns:
            new_df[c] = None
    existing = existing[new_df.columns]

    if existing.empty:
        out = new_df.copy()
    else:
        # Identify which new rows are not already present by key
        # Use merge anti-join approach
        marker = "__is_dup__"
        merged = new_df.merge(existing[key_cols], on=key_cols, how='left', indicator=True)
        to_add = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        # Re-extract only original columns from to_add (since it only carries key cols)
        # Join back with new_df to keep full columns
        to_add = to_add.merge(new_df, on=key_cols, how='left', suffixes=(None, None)).drop_duplicates(key_cols)
        out = pd.concat([existing, to_add], ignore_index=True)
    _upload_df_as_csv(container_client, blob_name, out)


def _parse_date(val) -> Optional[date]:
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val, utc=False).date()
    except Exception:
        return None


def _prep_eval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure required columns exist
    required = [COL_TOKEN, COL_HORIZON, COL_PCT, COL_PRICE, COL_START, COL_END,
                COL_CURRENT_PRICE, COL_HIT_TYPE, COL_VALIDATED, COL_VALIDATION_CLOSED]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Evaluation missing columns: {missing}")

    # Normalize types
    df[COL_START] = df[COL_START].apply(_parse_date)
    df[COL_END] = df[COL_END].apply(_parse_date)
    for c in [COL_PCT, COL_PRICE, COL_CURRENT_PRICE]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if COL_VALIDATED in df.columns:
        df[COL_VALIDATED] = df[COL_VALIDATED].fillna(False).astype(bool)
    if COL_VALIDATION_CLOSED in df.columns:
        df[COL_VALIDATION_CLOSED] = df[COL_VALIDATION_CLOSED].fillna(False).astype(bool)
    return df


def _dedup_today(df_today: pd.DataFrame) -> pd.DataFrame:
    # Keep last by (validated_on, last_run_utc) if available; fallback to index order
    sort_cols = []
    if COL_VALIDATED_ON in df_today.columns:
        sort_cols.append(COL_VALIDATED_ON)
    if COL_LAST_RUN in df_today.columns:
        sort_cols.append(COL_LAST_RUN)
    if sort_cols:
        df_today = df_today.sort_values(sort_cols)
    # drop duplicates keeping last occurrence
    df_today = df_today.drop_duplicates([COL_TOKEN, COL_HORIZON], keep='last')
    return df_today


def _hist_metrics(df_all: pd.DataFrame, horizon: str, today: date) -> pd.DataFrame:
    cfg = HORIZON_CFG[horizon]
    win_days = cfg["window_days"]
    start_d = today - timedelta(days=win_days)
    base = df_all[
        (df_all[COL_HIT_TYPE] == 'high') &
        (df_all[COL_START] >= start_d) & (df_all[COL_START] <= today) &
        (df_all[COL_VALIDATED] | df_all[COL_VALIDATION_CLOSED])
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=[COL_TOKEN, COL_HORIZON, 'high_period_count', 'high_period_success_rate'])
    g = base.groupby([COL_TOKEN, COL_HORIZON], dropna=False)
    met = g.agg(
        high_period_count=(COL_TOKEN, 'count'),
        high_period_success_rate=(COL_VALIDATED, 'mean'),
    ).reset_index()
    return met


def _build_table(df_all: pd.DataFrame, horizon: str, today: date) -> pd.DataFrame:
    # 1) today's candidates for this horizon
    df_today = df_all[(df_all[COL_START] == today) & (df_all[COL_HIT_TYPE] == 'high') & (df_all[COL_HORIZON] == horizon)].copy()
    if df_today.empty:
        return pd.DataFrame(columns=[COL_TOKEN, COL_HORIZON, 'current_price', 'predicted_price', 'predicted_change_pct', 'high_period_success_rate', 'high_period_count', COL_START, COL_END])
    df_today = _dedup_today(df_today)

    # 2) historical metrics for this horizon
    met = _hist_metrics(df_all, horizon, today)

    # 3) join
    out = df_today[[COL_TOKEN, COL_HORIZON, COL_CURRENT_PRICE, COL_PRICE, COL_PCT, COL_START, COL_END]].merge(
        met, on=[COL_TOKEN, COL_HORIZON], how='left'
    )

    # 4) apply min_count filter (interval*2)
    cfg = HORIZON_CFG[horizon]
    min_n = cfg["min_count"]
    out['high_period_count'] = out['high_period_count'].fillna(0).astype(int)
    out = out[out['high_period_count'] >= min_n]

    # 5) sort by SR desc, then Δ% desc
    out = out.sort_values(['high_period_success_rate', COL_PCT], ascending=[False, False])

    # 6) clean/format columns
    out = out.rename(columns={
        COL_CURRENT_PRICE: 'current_price',
        COL_PRICE: 'predicted_price',
        COL_PCT: 'predicted_change_pct'
    })

    return out.reset_index(drop=True)


def main(myTimer: func.TimerRequest) -> None:
    logging.info("[Notify HIGH] Function starting — building daily tables")
    try:
        service = _get_blob_service()
        cc_models = service.get_container_client(CONTAINER_MODELS)

        eval_df = _download_csv_as_df(cc_models, EVALUATION_BLOB_NAME)
        if eval_df is None or eval_df.empty:
            logging.warning("Evaluation CSV missing/empty — writing empty notify CSVs (no append)")
            # Create empty frames with headers including generated_at
            empty_cols = [COL_TOKEN, COL_HORIZON, 'current_price', 'predicted_price', 'predicted_change_pct',
                          'high_period_success_rate', 'high_period_count', COL_START, COL_END, COL_GENERATED_AT]
            _upload_df_as_csv(cc_models, NOTIFY_5D_BLOB, pd.DataFrame(columns=empty_cols))
            _upload_df_as_csv(cc_models, NOTIFY_1M_BLOB, pd.DataFrame(columns=empty_cols))
            _upload_df_as_csv(cc_models, NOTIFY_3M_BLOB, pd.DataFrame(columns=empty_cols))
            return

        eval_df = _prep_eval(eval_df)
        today = _now_prg_date()
        logging.info(f"Today (Europe/Prague): {today}")

        out_5d = _build_table(eval_df, '5D', today)
        out_1m = _build_table(eval_df, '1M', today)
        out_3m = _build_table(eval_df, '3M', today)

        # Add generated_at UTC timestamp
        gen_ts = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        for df in (out_5d, out_1m, out_3m):
            if not df.empty:
                df[COL_GENERATED_AT] = gen_ts

        # Append-only write per file using key (symbol,horizon,scrape_date)
        _append_notify_csv(cc_models, NOTIFY_5D_BLOB, out_5d)
        _append_notify_csv(cc_models, NOTIFY_1M_BLOB, out_1m)
        _append_notify_csv(cc_models, NOTIFY_3M_BLOB, out_3m)

        logging.info(f"Notify CSVs appended: 5D+={len(out_5d)} rows, 1M+={len(out_1m)} rows, 3M+={len(out_3m)} rows")

    except Exception as e:
        logging.exception(f"[Notify HIGH] Failed: {e}")
        raise

        logging.info(f"Notify CSVs created: 5D={len(out_5d)} rows, 1M={len(out_1m)} rows, 3M={len(out_3m)} rows")

    except Exception as e:
        logging.exception(f"[Notify HIGH] Failed: {e}")
        raise
