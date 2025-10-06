import logging
import os
from io import BytesIO
from datetime import datetime, date, timezone, timedelta
from typing import Optional, Tuple, Dict

import pandas as pd
from azure.storage.blob import BlobServiceClient
import azure.functions as func

# =============================
# Constants / Configuration
# =============================
CONTAINER_MODELS = "models-recalc"  # hardcoded per user
CONTAINER_MARKET = "market-data"    # container with daily OHLC data
MODEL_BLOB_NAME = "CoinDeskModels.csv"
SUMMARY_BLOB_NAME = "CoinDeskModels_Summary.csv"
EVALUATION_BLOB_NAME = "CoinDeskModelsEvaluation.csv"
# Timezone handling with fallbacks (Py3.8 safe)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TIMEZONE = ZoneInfo("Europe/Prague")
except Exception:
    try:
        from pytz import timezone as _pytz_timezone  # optional
        TIMEZONE = _pytz_timezone("Europe/Prague")
    except Exception:
        TIMEZONE = None  # fallback to UTC date

# Column names in CoinDeskModels
COL_TOKEN = "symbol"
COL_HORIZON = "horizon"
COL_PCT = "predicted_change_pct"
COL_PRICE = "predicted_price"
COL_START = "scrape_date"
COL_END = "model_to"
COL_IS_ACTIVE = "is_active"

# New/derived columns to (re)compute
COL_VALIDATED = "validated"
COL_VALIDATED_ON = "validated_on"
COL_HIT_TYPE = "hit_type"  # "low" | "high"
COL_HIT_PRICE = "hit_price"
COL_MIN_LOW = "min_low"
COL_MIN_LOW_DATE = "min_low_date"
COL_MAX_HIGH = "max_high"
COL_MAX_HIGH_DATE = "max_high_date"
COL_VALIDATION_CLOSED = "validation_closed"


# =============================
# Helpers
# =============================

def _get_blob_service() -> BlobServiceClient:
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        logging.error("AzureWebJobsStorage env var is not set.")
        raise RuntimeError("AzureWebJobsStorage env var is not set.")
    logging.info("AzureWebJobsStorage detected. Creating BlobServiceClient…")
    return BlobServiceClient.from_connection_string(conn_str)


def _download_csv_as_df(container_client, blob_name: str) -> Optional[pd.DataFrame]:
    logging.info(f"Downloading CSV: container='{container_client.container_name}', blob='{blob_name}'")
    try:
        blob_client = container_client.get_blob_client(blob_name)
        stream = blob_client.download_blob(max_concurrency=1).readall()
        # Support gzip if needed in future
        try:
            df = pd.read_csv(BytesIO(stream), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(stream), encoding="utf-8")
        logging.info(f"Downloaded {blob_name}: rows={len(df)}, cols={list(df.columns)}")
        return df
    except Exception as e:
        logging.warning(f"Could not download or parse CSV '{blob_name}': {e}")
        return None


def _check_blob_exists(container_client, blob_name: str) -> bool:
    try:
        blob_client = container_client.get_blob_client(blob_name)
        exists = blob_client.exists()
        logging.info(f"Blob exists? {blob_name} -> {exists}")
        return bool(exists)
    except Exception as e:
        logging.warning(f"exists() failed for '{blob_name}': {e}")
        return False


def _list_blobs_with_prefix(container_client, prefix: str, max_items: int = 10) -> list:
    try:
        blobs = []
        for i, b in enumerate(container_client.list_blobs(name_starts_with=prefix)):
            blobs.append(b.name)
            if i + 1 >= max_items:
                break
        logging.info(f"Found {len(blobs)} blobs with prefix '{prefix}': {blobs}")
        return blobs
    except Exception as e:
        logging.warning(f"list_blobs failed for prefix '{prefix}': {e}")
        return []


def _upload_df_as_csv(container_client, blob_name: str, df: pd.DataFrame) -> None:
    logging.info(f"Uploading CSV: container='{container_client.container_name}', blob='{blob_name}', rows={len(df)}")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    container_client.upload_blob(name=blob_name, data=csv_bytes, overwrite=True)
    logging.info(f"Uploaded CSV -> {blob_name}")


def _parse_date_safe(val) -> Optional[date]:
    if pd.isna(val):
        return None
    try:
        dt = pd.to_datetime(val, utc=False).date()
        return dt
    except Exception:
        return None


def _ensure_columns(df: pd.DataFrame, columns_with_defaults: dict) -> pd.DataFrame:
    for col, default in columns_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def _first_hit_date(interval_df: pd.DataFrame, condition_series: pd.Series) -> Optional[date]:
    hit_idx = condition_series[condition_series].index
    if len(hit_idx) == 0:
        return None
    first_idx = hit_idx[0]
    d = interval_df.loc[first_idx, "date"]
    return _parse_date_safe(d)


def _min_low_with_date(interval_df: pd.DataFrame) -> Tuple[Optional[float], Optional[date]]:
    if interval_df.empty or "low" not in interval_df.columns:
        return None, None
    idx = interval_df["low"].astype(float).idxmin()
    if pd.isna(idx):
        return None, None
    return float(interval_df.loc[idx, "low"]), _parse_date_safe(interval_df.loc[idx, "date"])


def _max_high_with_date(interval_df: pd.DataFrame) -> Tuple[Optional[float], Optional[date]]:
    if interval_df.empty or "high" not in interval_df.columns:
        return None, None
    idx = interval_df["high"].astype(float).idxmax()
    if pd.isna(idx):
        return None, None
    return float(interval_df.loc[idx, "high"]), _parse_date_safe(interval_df.loc[idx, "date"])


def _validate_row(row: pd.Series, daily_df: pd.DataFrame) -> dict:
    pred_price = float(row.get(COL_PRICE)) if not pd.isna(row.get(COL_PRICE)) else None
    pct = row.get(COL_PCT)

    hit_type = None
    if pd.notna(pct):
        try:
            pct_f = float(pct)
            if pct_f < 0:
                hit_type = "low"
            elif pct_f > 0:
                hit_type = "high"
        except Exception:
            pass

    min_low, min_low_date = _min_low_with_date(daily_df)
    max_high, max_high_date = _max_high_with_date(daily_df)

    validated = False
    validated_on: Optional[date] = None
    hit_price: Optional[float] = None

    if hit_type == "low" and pred_price is not None and not daily_df.empty:
        cond = daily_df["low"].astype(float) <= pred_price
        validated_on = _first_hit_date(daily_df, cond)
        validated = validated_on is not None
        if validated:
            hit_day = daily_df[daily_df["date"].apply(_parse_date_safe) == validated_on]
            if not hit_day.empty:
                hit_price = float(hit_day.iloc[0]["low"]) if "low" in hit_day.columns else None

    elif hit_type == "high" and pred_price is not None and not daily_df.empty:
        cond = daily_df["high"].astype(float) >= pred_price
        validated_on = _first_hit_date(daily_df, cond)
        validated = validated_on is not None
        if validated:
            hit_day = daily_df[daily_df["date"].apply(_parse_date_safe) == validated_on]
            if not hit_day.empty:
                hit_price = float(hit_day.iloc[0]["high"]) if "high" in hit_day.columns else None

    # Compute today's date in Europe/Prague with safe fallbacks
    if TIMEZONE is not None:
        try:
            today_local = datetime.now(TIMEZONE).date()
        except Exception:
            today_local = datetime.utcnow().date()
    else:
        today_local = datetime.utcnow().date()
    end_d = _parse_date_safe(row.get(COL_END))
    validation_closed = bool(end_d is not None and today_local > end_d)

    return {
        COL_HIT_TYPE: hit_type,
        COL_VALIDATED: bool(validated),
        COL_VALIDATED_ON: validated_on.isoformat() if validated_on else None,
        COL_HIT_PRICE: hit_price,
        COL_MIN_LOW: min_low,
        COL_MIN_LOW_DATE: min_low_date.isoformat() if min_low_date else None,
        COL_MAX_HIGH: max_high,
        COL_MAX_HIGH_DATE: max_high_date.isoformat() if max_high_date else None,
        COL_VALIDATION_CLOSED: validation_closed,
    }


def _filter_interval(df: pd.DataFrame, start_d: Optional[date], end_d: Optional[date]) -> pd.DataFrame:
    if start_d is None or end_d is None or df.empty:
        return df.iloc[0:0]
    df = df.copy()
    # --- Derive 'date' robustly ---
    cols_lower = {c.lower(): c for c in df.columns}
    if 'date' not in df.columns:
        if 'closetimeiso' in cols_lower:
            df['date'] = pd.to_datetime(df[cols_lower['closetimeiso']]).dt.date
        elif 'opentime' in cols_lower:
            df['date'] = pd.to_datetime(df[cols_lower['opentime']], unit='ms').dt.date
        elif 'closetime' in cols_lower:
            df['date'] = pd.to_datetime(df[cols_lower['closetime']], unit='ms').dt.date
        else:
            # As a last resort, try first datetime-like column
            for c in df.columns:
                if 'time' in c.lower() or 'date' in c.lower():
                    df['date'] = pd.to_datetime(df[c], errors='coerce').dt.date
                    break
    # Ensure numeric types on OHLC
    for k in ['open','high','low','close']:
        src = cols_lower.get(k)
        if src and src != k and k not in df.columns:
            df[k] = df[src]
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors='coerce')
    df['date'] = df['date'].apply(_parse_date_safe)
    mask = (df['date'] >= start_d) & (df['date'] <= end_d)
    return df.loc[mask].sort_values('date').reset_index(drop=True)


def _build_summary(models_df: pd.DataFrame) -> pd.DataFrame:
    active = models_df[models_df[COL_IS_ACTIVE] == True].copy()
    active[COL_VALIDATED] = active[COL_VALIDATED].fillna(False).astype(bool)

    def sign_bucket(x):
        try:
            xf = float(x)
            if xf < 0:
                return "neg"
            if xf > 0:
                return "pos"
            return "zero"
        except Exception:
            return "nan"

    active["_bucket"] = active[COL_PCT].apply(sign_bucket)

    groups = []
    for (token, horizon), g in active.groupby([COL_TOKEN, COL_HORIZON], dropna=False):
        n_all = len(g)
        succ_all = int(g[COL_VALIDATED].sum()) if n_all > 0 else 0
        rate_all = (succ_all / n_all) if n_all > 0 else None

        g_neg = g[g["_bucket"] == "neg"]
        n_neg = len(g_neg)
        succ_neg = int(g_neg[COL_VALIDATED].sum()) if n_neg > 0 else 0
        rate_neg = (succ_neg / n_neg) if n_neg > 0 else None

        g_pos = g[g["_bucket"] == "pos"]
        n_pos = len(g_pos)
        succ_pos = int(g_pos[COL_VALIDATED].sum()) if n_pos > 0 else 0
        rate_pos = (succ_pos / n_pos) if n_pos > 0 else None

        groups.append({
            COL_TOKEN: token,
            COL_HORIZON: horizon,
            "n_all_active": n_all,
            "success_rate_all": rate_all,
            "n_neg": n_neg,
            "success_rate_neg": rate_neg,
            "n_pos": n_pos,
            "success_rate_pos": rate_pos,
            "last_run_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

    summary_df = pd.DataFrame(groups, columns=[
        COL_TOKEN, COL_HORIZON, "n_all_active", "success_rate_all",
        "n_neg", "success_rate_neg", "n_pos", "success_rate_pos", "last_run_utc"
    ])

    return summary_df.sort_values([COL_TOKEN, COL_HORIZON]).reset_index(drop=True)


# =============================
# Azure Function Entry Point
# =============================

def main(myTimer: func.TimerRequest) -> None:
    start_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logging.info(f"[CoinDesk Validation] Function started at {start_ts}")

    try:
        logging.info("Creating BlobServiceClient…")
        service = _get_blob_service()
        container_models = service.get_container_client(CONTAINER_MODELS)
        container_market = service.get_container_client(CONTAINER_MARKET)
        logging.info(f"Using containers -> models: '{CONTAINER_MODELS}', market: '{CONTAINER_MARKET}'")

        # 1) Load models CSV
        logging.info(f"Loading models CSV '{MODEL_BLOB_NAME}'…")
        models_df = _download_csv_as_df(container_models, MODEL_BLOB_NAME)
        if models_df is None or models_df.empty:
            logging.error("CoinDeskModels.csv is missing or empty. Nothing to process.")
            return
        logging.info(f"Models columns: {list(models_df.columns)} | rows={len(models_df)}")

        required_cols = [COL_TOKEN, COL_HORIZON, COL_PCT, COL_PRICE, COL_START, COL_END, COL_IS_ACTIVE]
        missing = [c for c in required_cols if c not in models_df.columns]
        if missing:
            logging.error(f"Missing required columns in models: {missing}")
            raise RuntimeError(f"CoinDeskModels.csv is missing required columns: {missing}")

        # Ensure derived columns exist
        models_df = _ensure_columns(models_df, {
            COL_VALIDATED: False,
            COL_VALIDATED_ON: None,
            COL_HIT_TYPE: None,
            COL_HIT_PRICE: None,
            COL_MIN_LOW: None,
            COL_MIN_LOW_DATE: None,
            COL_MAX_HIGH: None,
            COL_MAX_HIGH_DATE: None,
            COL_VALIDATION_CLOSED: False,
        })

        # Normalize date columns
        models_df[COL_START] = models_df[COL_START].apply(lambda v: _parse_date_safe(v).isoformat() if _parse_date_safe(v) else None)
        models_df[COL_END] = models_df[COL_END].apply(lambda v: _parse_date_safe(v).isoformat() if _parse_date_safe(v) else None)

        # 2) Only active
        active_mask = models_df[COL_IS_ACTIVE] == True
        active_rows = models_df[active_mask].copy()
        logging.info(f"Active predictions: {len(active_rows)}")
        if active_rows.empty:
            logging.info("No active predictions. Writing summary and exiting.")
            empty_summary = _build_summary(models_df.iloc[0:0])
            _upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, empty_summary)
            _upload_df_as_csv(container_models, MODEL_BLOB_NAME, models_df)
            return

        token_to_daily_df: Dict[str, pd.DataFrame] = {}
        evaluation_rows = []

        for idx, row in active_rows.iterrows():
            token_raw = str(row.get(COL_TOKEN)).strip() if pd.notna(row.get(COL_TOKEN)) else None
            if not token_raw:
                logging.warning(f"Row {idx} has empty symbol; skipping.")
                continue
            token_uc = token_raw.upper()
            market_symbol = token_uc if token_uc.endswith('USDC') else f"{token_uc}USDC"
            token_blob_name = f"1D/{market_symbol}.csv"
            logging.info(f"Loading daily CSV for '{token_uc}' -> '{token_blob_name}'")

            daily_df = token_to_daily_df.get(market_symbol)
            if daily_df is None:
                if not _check_blob_exists(container_market, token_blob_name):
                    logging.warning(f"Daily blob not found: '{token_blob_name}'. Listing candidates with prefix '1D/{token_uc}'…")
                    _list_blobs_with_prefix(container_market, prefix=f"1D/{token_uc}")
                _tmp_df = _download_csv_as_df(container_market, token_blob_name)
                daily_df = _tmp_df if _tmp_df is not None else pd.DataFrame()
                token_to_daily_df[market_symbol] = daily_df

            start_d = _parse_date_safe(row.get(COL_START))
            end_d = _parse_date_safe(row.get(COL_END))
            logging.info(f"Row {idx} | symbol={token_uc} | interval={start_d}..{end_d} | price={row.get(COL_PRICE)} | pct={row.get(COL_PCT)}")

            interval_df = _filter_interval(daily_df, start_d, end_d)
            logging.info(f"Row {idx} | interval rows={len(interval_df)}")

            vals = _validate_row(row, interval_df)
            for k, v in vals.items():
                models_df.at[idx, k] = v

            evaluation_rows.append({
                COL_TOKEN: token_uc,
                COL_HORIZON: row.get(COL_HORIZON),
                COL_START: start_d.isoformat() if start_d else None,
                COL_END: end_d.isoformat() if end_d else None,
                COL_PRICE: float(row.get(COL_PRICE)) if pd.notna(row.get(COL_PRICE)) else None,
                COL_PCT: float(row.get(COL_PCT)) if pd.notna(row.get(COL_PCT)) else None,
                COL_HIT_TYPE: vals.get(COL_HIT_TYPE),
                COL_VALIDATED: vals.get(COL_VALIDATED),
                COL_VALIDATED_ON: vals.get(COL_VALIDATED_ON),
                COL_HIT_PRICE: vals.get(COL_HIT_PRICE),
                COL_MIN_LOW: vals.get(COL_MIN_LOW),
                COL_MIN_LOW_DATE: vals.get(COL_MIN_LOW_DATE),
                COL_MAX_HIGH: vals.get(COL_MAX_HIGH),
                COL_MAX_HIGH_DATE: vals.get(COL_MAX_HIGH_DATE),
                COL_VALIDATION_CLOSED: vals.get(COL_VALIDATION_CLOSED),
                'last_run_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            })

        # 3) Upload updated models CSV (overwrite)
        _upload_df_as_csv(container_models, MODEL_BLOB_NAME, models_df)

        # 3b) Upload evaluation CSV (overwrite)
        evaluation_df = pd.DataFrame(evaluation_rows, columns=[
            COL_TOKEN, COL_HORIZON, COL_START, COL_END, COL_PRICE, COL_PCT,
            COL_HIT_TYPE, COL_VALIDATED, COL_VALIDATED_ON, COL_HIT_PRICE,
            COL_MIN_LOW, COL_MIN_LOW_DATE, COL_MAX_HIGH, COL_MAX_HIGH_DATE,
            COL_VALIDATION_CLOSED, 'last_run_utc'
        ])
        _upload_df_as_csv(container_models, EVALUATION_BLOB_NAME, evaluation_df)

        # 4) Build & upload summary CSV (overwrite)
        summary_df = _build_summary(models_df)
        _upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, summary_df)

        # 4) Build & upload summary CSV (overwrite)
        summary_df = _build_summary(models_df)
        _upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, summary_df)

        logging.info("[CoinDesk Validation] Completed successfully.")

    except Exception as e:
        logging.exception(f"[CoinDesk Validation] Failed with error: {e}")
        raise


# =============================
# function.json (place in the same function folder)
# Folder name should be: CoinDeskModelsEvaluation
# =============================
# {
#   "scriptFile": "__init__.py",
#   "entryPoint": "main",
#   "bindings": [
#     {
#       "name": "myTimer",
#       "type": "timerTrigger",
#       "direction": "in",
#       "schedule": "0 0 15 * * *"  
#     }
#   ]
# }

# =============================
# host.json (root of the Function App)
# =============================
# {
#   "version": "2.0",
#   "logging": {
#     "applicationInsights": {
#       "samplingSettings": {
#         "isEnabled": true
#       }
#     }
#   }
# }

# =============================
# requirements.txt (root or function folder)
# =============================
# azure-functions
# azure-storage-blob>=12.19.0
# pandas>=2.0.0
# pytz>=2023.3  # optional; used only if zoneinfo unavailable

# =============================
# local.settings.json (for local run; do NOT deploy to production)
# =============================
# {
#   "IsEncrypted": false,
#   "Values": {
#     "AzureWebJobsStorage": "<YourConnectionString>",
#     "FUNCTIONS_WORKER_RUNTIME": "python"
#   }
# }
