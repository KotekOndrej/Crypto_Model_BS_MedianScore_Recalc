import logging
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


token_to_daily_df: dict[str, pd.DataFrame] = {}


for idx, row in active_rows.iterrows():
token = str(row.get(COL_TOKEN)).strip() if pd.notna(row.get(COL_TOKEN)) else None
if not token:
logging.warning(f"Row {idx} has empty token; skipping.")
continue


if token not in token_to_daily_df:
token_blob_name = f"1D/{token}USDC.csv"
logging.info(f"Preparing to load daily CSV for token '{token}' -> '{token_blob_name}'")
if not _check_blob_exists(container_market, token_blob_name):
logging.warning(f"Daily blob not found: '{token_blob_name}'. Listing candidates with prefix '1D/{token}'…")
_list_blobs_with_prefix(container_market, prefix=f"1D/{token}")
daily_df = _download_csv_as_df(container_market, token_blob_name)
if daily_df is None:
logging.warning(f"Daily CSV for token '{token}' not found or unreadable. Using empty frame.")
token_to_daily_df[token] = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
else:
# normalize columns (case-insensitive)
cols_map = {c.lower(): c for c in daily_df.columns}
for expected in ["date", "open", "high", "low", "close"]:
if expected not in cols_map:
logging.warning(f"Daily CSV '{token_blob_name}' missing column '{expected}'. Creating empty column.")
daily_df[expected] = pd.NA
else:
src = cols_map[expected]
if src != expected:
daily_df.rename(columns={src: expected}, inplace=True)
keep = ["date", "open", "high", "low", "close"]
daily_df = daily_df[keep]
logging.info(f"Loaded daily for {token}: rows={len(daily_df)}")
token_to_daily_df[token] = daily_df


daily_df = token_to_daily_df[token]


start_d = _parse_date_safe(row.get(COL_START))
end_d = _parse_date_safe(row.get(COL_END))
logging.info(f"Row {idx} | token={token} | interval={start_d}..{end_d} | price={row.get(COL_PRICE)} | pct={row.get(COL_PCT)}")


interval_df = _filter_interval(daily_df, start_d, end_d)
logging.info(f"Row {idx} | interval rows={len(interval_df)}")


vals = _validate_row(row, interval_df)
for k, v in vals.items():
models_df.at[idx, k] = v


# 3) Upload updated models CSV (overwrite)
_upload_df_as_csv(container_models, MODEL_BLOB_NAME, models_df)


# 4) Build & upload summary CSV (overwrite)
summary_df = _build_summary(models_df)
_upload_df_as_csv(container_models, SUMMARY_BLOB_NAME, summary_df)


logging.info("[CoinDesk Validation] Completed successfully.")


except Exception as e:
logging.exception(f"[CoinDesk Validation] Failed with error: {e}")
raise
