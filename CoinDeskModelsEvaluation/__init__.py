import logging
import os
from io import BytesIO
from datetime import datetime, date
from typing import Optional, Dict

import pandas as pd
from azure.storage.blob import BlobServiceClient
import azure.functions as func

# =============================
# Constants / Configuration
# =============================
CONTAINER_MODELS = "models-recalc"
CONTAINER_MARKET = "market-data"
MODEL_BLOB_NAME = "CoinDeskModels.csv"
SUMMARY_BLOB_NAME = "CoinDeskModels_Summary.csv"
EVALUATION_BLOB_NAME = "CoinDeskModelsEvaluation.csv"

# Timezone setup — try zoneinfo, else fallback to UTC-only dates
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TIMEZONE = ZoneInfo("Europe/Prague")
except Exception:
    TIMEZONE = None

# Column names in CoinDeskModels
COL_TOKEN = "symbol"
COL_HORIZON = "horizon"
COL_PCT = "predicted_change_pct"
COL_PRICE = "predicted_price"
COL_START = "scrape_date"
COL_END = "model_to"
COL_IS_ACTIVE = "is_active"
COL_CURRENT_PRICE = "current_price"  # <-- new: propagate to evaluation

# New/derived columns (validation outputs)
COL_VALIDATED = "validated"
COL_VALIDATED_ON = "validated_on"
COL_HIT_TYPE = "hit_type"
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
    return BlobServiceClient.from_connection_string(conn_str)


def _download_csv_as_df(container_client, blob_name: str) -> Optional[pd.DataFrame]:
    logging.info(f"Downloading CSV: container='{container_client.container_name}', blob='{blob_name}'")
    try:
        blob_client = container_client.get_blob_client(blob_name)
        stream = blob_client.download_blob(max_concurrency=1).readall()
        try:
            df = pd.read_csv(BytesIO(stream), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(stream), encoding="utf-8")
        return df
    except Exception as e:
        logging.warning(f"Could not download or parse CSV '{blob_name}': {e}")
        return None


def _upload_df_as_csv(container_client, blob_name: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    container_client.upload_blob(name=blob_name, data=csv_bytes, overwrite=True)
    logging.info(f"Uploaded CSV -> {blob_name} ({len(df)} rows)")


def _parse_date_safe(val) -> Optional[date]:
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val, utc=False).date()
    except Exception:
        return None


def _ensure_interval_and_ohlc(df: pd.DataFrame, start_d: Optional[date], end_d: Optional[date]) -> pd.DataFrame:
    """Return DF filtered to inclusive [start_d, end_d] with robust 'date' derivation and numeric OHLC."""
    if start_d is None or end_d is None or df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])  # empty

    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # Derive date
    if "date" not in df.columns:
        if "closetimeiso" in cols_lower:
            df["date"] = pd.to_datetime(df[cols_lower["closetimeiso"]]).dt.date
        elif "opentime" in cols_lower:
            df["date"] = pd.to_datetime(df[cols_lower["opentime"]], unit="ms").dt.date
        elif "closetime" in cols_lower:
            df["date"] = pd.to_datetime(df[cols_lower["closetime"]], unit="ms").dt.date
        else:
            # try any time-like column
            for c in df.columns:
                if "time" in c.lower() or "date" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce").dt.date
                    break
    df["date"] = df["date"].apply(_parse_date_safe)

    # Ensure OHLC and numeric
    for k in ["open", "high", "low", "close"]:
        src = cols_lower.get(k)
        if src and src != k and k not in df.columns:
            df[k] = df[src]
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    mask = (df["date"] >= start_d) & (df["date"] <= end_d)
    out = df.loc[mask, [c for c in ["date","open","high","low","close"] if c in df.columns]].dropna(subset=["date"])\
            .sort_values("date").reset_index(drop=True)
    return out


def _validate_row(row: pd.Series, daily_df: pd.DataFrame) -> Dict[str, object]:
    pred_price = float(row.get(COL_PRICE)) if pd.notna(row.get(COL_PRICE)) else None
    pct = row.get(COL_PCT)

    # Determine which side to test
    hit_type = None
    if pd.notna(pct):
        try:
            p = float(pct)
            if p < 0:
                hit_type = "low"
            elif p > 0:
                hit_type = "high"
        except Exception:
            pass

    # Min/Max over interval
    min_low = float(daily_df["low"].min()) if "low" in daily_df.columns and not daily_df.empty else None
    max_high = float(daily_df["high"].max()) if "high" in daily_df.columns and not daily_df.empty else None
    min_low_date = daily_df.loc[daily_df["low"].idxmin(), "date"] if "low" in daily_df.columns and not daily_df.empty else None
    max_high_date = daily_df.loc[daily_df["high"].idxmax(), "date"] if "high" in daily_df.columns and not daily_df.empty else None

    validated = False
    validated_on = None
    hit_price = None

    if hit_type == "low" and pred_price is not None and not daily_df.empty:
        cond = daily_df["low"] <= pred_price  # exact <= as requested
        if cond.any():
            first_idx = cond.index[cond.argmax()]  # first True by index order
            validated = True
            validated_on = daily_df.loc[first_idx, "date"]
            hit_price = float(daily_df.loc[first_idx, "low"]) if "low" in daily_df.columns else None

    elif hit_type == "high" and pred_price is not None and not daily_df.empty:
        cond = daily_df["high"] >= pred_price  # exact >= as requested
        if cond.any():
            first_idx = cond.index[cond.argmax()]
            validated = True
            validated_on = daily_df.loc[first_idx, "date"]
            hit_price = float(daily_df.loc[first_idx, "high"]) if "high" in daily_df.columns else None

    # validation_closed: today (Europe/Prague) > end_date
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
        COL_MIN_LOW_DATE: min_low_date.isoformat() if isinstance(min_low_date, date) else None,
        COL_MAX_HIGH: max_high,
        COL_MAX_HIGH_DATE: max_high_date.isoformat() if isinstance(max_high_date, date) else None,
        COL_VALIDATION_CLOSED: validation_closed,
    }


# =============================
# Function entry
# =============================

def main(myTimer: func.TimerRequest) -> None:
    logging.info("[CoinDesk Validation] Function starting — entering main()")
    try:
        # Storage
        service = _get_blob_service()
        container_models = service.get_container_client(CONTAINER_MODELS)
        container_market = service.get_container_client(CONTAINER_MARKET)

        # Load models and keep only active
        models_df = _download_csv_as_df(container_models, MODEL_BLOB_NAME)
        if models_df is None or models_df.empty:
            logging.error("CoinDeskModels.csv is missing or empty.")
            return

        required = [COL_TOKEN, COL_HORIZON, COL_PCT, COL_PRICE, COL_START, COL_END, COL_IS_ACTIVE]
        miss = [c for c in required if c not in models_df.columns]
        if miss:
            logging.error(f"CoinDeskModels.csv missing columns: {miss}")
            return

        active_df = models_df[models_df[COL_IS_ACTIVE] == True].copy()
        if active_df.empty:
            logging.info("No active rows to process.")
            # Still write empty evaluation & summary for transparency
            _upload_df_as_csv(container_models, EVALUATION_BLOB_NAME, pd.DataFrame())
            _upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, pd.DataFrame())
            return

        # Normalize date columns in models (keep as dates for interval)
        active_df[COL_START] = active_df[COL_START].apply(_parse_date_safe)
        active_df[COL_END] = active_df[COL_END].apply(_parse_date_safe)

        # Build evaluation rows
        eval_rows = []
        cache_daily: Dict[str, pd.DataFrame] = {}

        for _, row in active_df.iterrows():
            token = str(row[COL_TOKEN]).strip().upper()
            market_symbol = token if token.endswith("USDC") else f"{token}USDC"
            blob_name = f"1D/{market_symbol}.csv"

            # Load or reuse daily CSV
            if market_symbol not in cache_daily:
                cache_daily[market_symbol] = _download_csv_as_df(container_market, blob_name) or pd.DataFrame()
            daily_df = cache_daily[market_symbol]

            start_d = _parse_date_safe(row[COL_START])
            end_d = _parse_date_safe(row[COL_END])
            interval_df = _ensure_interval_and_ohlc(daily_df, start_d, end_d)

            vals = _validate_row(row, interval_df)

            record = {
                COL_TOKEN: token,
                COL_HORIZON: row.get(COL_HORIZON),
                COL_START: start_d,
                COL_END: end_d,
                COL_PRICE: row.get(COL_PRICE),
                COL_PCT: row.get(COL_PCT),
                COL_CURRENT_PRICE: row.get(COL_CURRENT_PRICE) if COL_CURRENT_PRICE in active_df.columns else None,
            }
            record.update(vals)
            eval_rows.append(record)

        evaluation_df = pd.DataFrame(eval_rows)

        # Reorder columns for clarity
        ordered_cols = [
            COL_TOKEN, COL_HORIZON, COL_START, COL_END,
            COL_PRICE, COL_PCT, COL_CURRENT_PRICE,
            COL_HIT_TYPE, COL_VALIDATED, COL_VALIDATED_ON, COL_HIT_PRICE,
            COL_MIN_LOW, COL_MIN_LOW_DATE, COL_MAX_HIGH, COL_MAX_HIGH_DATE,
            COL_VALIDATION_CLOSED
        ]
        for c in ordered_cols:
            if c not in evaluation_df.columns:
                evaluation_df[c] = None
        evaluation_df = evaluation_df[ordered_cols]

        _upload_df_as_csv(container_models, EVALUATION_BLOB_NAME, evaluation_df)
        logging.info(f"Created {EVALUATION_BLOB_NAME} with {len(evaluation_df)} rows.")

        # =============================
        # Create SUMMARY strictly from VALIDATED = TRUE in Evaluation
        # =============================
        if not evaluation_df.empty:
            df = evaluation_df.copy()
            df[COL_VALIDATED] = df[COL_VALIDATED].fillna(False).astype(bool)
            df = df[df[COL_VALIDATED] == True]

            if df.empty:
                logging.warning("No validated rows in evaluation — writing empty summary.")
                _upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, pd.DataFrame(columns=[
                    COL_TOKEN, COL_HORIZON, "n_all_active", "success_rate_all",
                    "n_neg", "success_rate_neg", "n_pos", "success_rate_pos", "last_run_utc"
                ]))
            else:
                # bucket by sign (still computed on validated subset)
                def bucket(v):
                    try:
                        v = float(v)
                        if v < 0: return 'neg'
                        if v > 0: return 'pos'
                        return 'zero'
                    except: return 'nan'
                df['_bucket'] = df[COL_PCT].apply(bucket)

                groups = []
                for (token, horizon), g in df.groupby([COL_TOKEN, COL_HORIZON], dropna=False):
                    n_all = len(g)  # only validated rows
                    succ_all = int(g[COL_VALIDATED].sum())  # equals n_all
                    rate_all = succ_all / n_all if n_all else None  # will be 1.0

                    g_neg = g[g['_bucket'] == 'neg']; n_neg = len(g_neg)
                    g_pos = g[g['_bucket'] == 'pos']; n_pos = len(g_pos)
                    succ_neg = int(g_neg[COL_VALIDATED].sum()) if n_neg else 0
                    succ_pos = int(g_pos[COL_VALIDATED].sum()) if n_pos else 0
                    rate_neg = succ_neg / n_neg if n_neg else None
                    rate_pos = succ_pos / n_pos if n_pos else None

                    groups.append({
                        COL_TOKEN: token,
                        COL_HORIZON: horizon,
                        'n_all_active': n_all,
                        'success_rate_all': rate_all,
                        'n_neg': n_neg,
                        'success_rate_neg': rate_neg,
                        'n_pos': n_pos,
                        'success_rate_pos': rate_pos,
                        'last_run_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z'
                    })

                summary_df = pd.DataFrame(groups).sort_values([COL_TOKEN, COL_HORIZON]).reset_index(drop=True)
                _upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, summary_df)
                logging.info(f"Created {SUMMARY_BLOB_NAME} with {len(summary_df)} rows.")
        else:
            logging.warning("Evaluation DataFrame is empty — skipping summary.")

    except Exception as e:
        logging.exception(f"[CoinDesk Validation] Failed with error: {e}")
        raise
