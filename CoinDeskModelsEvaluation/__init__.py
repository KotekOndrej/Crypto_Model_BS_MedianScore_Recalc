import logging
import os
from io import BytesIO
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional, Tuple


import pandas as pd
from azure.storage.blob import BlobServiceClient
import azure.functions as func


# =============================
# Constants / Configuration
# =============================
CONTAINER_MODELS = "models-recalc" # per user requirement: hardcode container for models
CONTAINER_MARKET = "market-data" # container with daily OHLC data
MODEL_BLOB_NAME = "CoinDeskModels.csv"
SUMMARY_BLOB_NAME = "CoinDeskModels_Summary.csv"
TIMEZONE = ZoneInfo("Europe/Prague")


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
COL_HIT_TYPE = "hit_type" # "low" | "high" (based on sign of predicted_change_pct)
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
raise RuntimeError("AzureWebJobsStorage env var is not set.")
return BlobServiceClient.from_connection_string(conn_str)




def _download_csv_as_df(container_client, blob_name: str) -> Optional[pd.DataFrame]:
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




raise
