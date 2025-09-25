import datetime as dt
import logging
import os
import re
import traceback
from decimal import Decimal, InvalidOperation
from typing import Tuple, List, Dict, Optional

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# -------------------- Konfigurace --------------------
COINCIDEX_BASE_URL = "https://coincodex.com/predictions/"
USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/2.1)")
TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Prague")  # informativní
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")  # storage Function Appu
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")  # výchozí název CSV

# DŮLEŽITÉ: CoinCodex serverově vrací jen 1. stránku; další jsou JS. Nezkoušej nic >1.
MAX_PAGES = int(os.getenv("MAX_PAGES", "1"))

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Mapování sloupců na label + funkci výpočtu cílového data
HORIZON_MAP = {
    "5D Prediction": ("5D", lambda d: d + relativedelta(days=5)),
    "1M Prediction": ("1M", lambda d: d + relativedelta(months=1)),
    "3M Prediction": ("3M", lambda d: d + relativedelta(months=3)),
    "6M Prediction": ("6M", lambda d: d + relativedelta(months=6)),
    "1Y Prediction": ("1Y", lambda d: d + relativedelta(years=1)),
}

# CSV hlavička – current_price, load_ts, is_active, validation
CSV_HEADER = (
    "scrape_date,load_ts,symbol,token_name,current_price,horizon,model_to,"
    "predicted_price,predicted_change_pct,is_active,validation\n"
)

# -------------------- Parser helpery --------------------
def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z buňky typu '$ 4,660.39 11.49%' vytáhne cenu (Decimal) a % změnu (Decimal)."""
    if not text:
        return None, None
    m_price = re.search(r"[-]?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    price_dec: Optional[Decimal] = None
    if m_price:
        num = m_price.group(1).replace(",", "")
        try:
            price_dec = Decimal(num)
        except InvalidOperation:
            price_dec = None
    m_pct = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", text)
    pct_dec: Optional[Decimal] = None
    if m_pct:
        try:
            pct_dec = Decimal(m_pct.group(1))
        except InvalidOperation:
            pct_dec = None
    return price_dec, pct_dec


def extract_table_rows(html: str) -> List[Dict]:
    """
    Vrací list dictů:
      {
        symbol, token_name, current_price,
        pred_5d, chg_5d, pred_1m, chg_1m, pred_3m, chg_3m, pred_6m, chg_6m, pred_1y, chg_1y
      }
    """
    soup = BeautifulSoup(html, "html.parser")

    table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if any(("5D" in h and "Prediction" in h) for h in headers):
            table = t
            break
    if table is None:
        return []

    header_texts = [th.get_text(strip=True) for th in table.find_all("th")]
    col_idx = {name: i for i, name in enumerate(header_texts)}

    # Price sloupec (může být různě pojmenován, hledáme 'Price')
    price_col_name = None
    if "Price" in col_idx:
        price_col_name = "Price"
    else:
        for h in header_texts:
            if "Price" in h and h not in HORIZON_MAP:
                price_col_name = h
                break

    required = ["Name", "5D Prediction", "1M Prediction", "3M Prediction", "6M Prediction", "1Y Prediction"]
    if not all(r in col_idx for r in required):
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    rows: List[Dict] = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < len(header_texts):
            continue

        # Name: např. "ETH Ethereum"
        name_cell = tds[col_idx["Name"]]
        name_text = name_cell.get_text(" ", strip=True)
        symbol = None
        token_name = None
        if name_text:
            parts = name_text.split()
            if len(parts) >= 2:
                symbol = parts[0]
                token_name = " ".join(parts[1:])
            else:
                token_name = name_text

        if not symbol:
            a = name_cell.find("a")
            if a and a.get_text(strip=True):
                atext = a.get_text(" ", strip=True)
                parts = atext.split()
                if len(parts) >= 2:
                    symbol = parts[0]
                    token_name = " ".join(parts[1:])
                else:
                    token_name = atext

        # Aktuální cena
        current_price: Optional[Decimal] = None
        if price_col_name:
            try:
                price_cell_text = tds[col_idx[price_col_name]].get_text(" ", strip=True)
                current_price, _ = parse_price_and_change(price_cell_text)
            except Exception:
                current_price = None

        preds: Dict[str, Optional[Decimal]] = {}
        chgs: Dict[str, Optional[Decimal]] = {}
        for header in required[1:]:
            cell_text = tds[col_idx[header]].get_text(" ", strip=True)
            price, pct = parse_price_and_change(cell_text)
            preds[header] = price
            chgs[header] = pct

        if symbol and any(preds.values()):
            rows.append({
                "symbol": symbol,
                "token_name": token_name or "",
                "current_price": current_price,
                "pred_5d": preds.get("5D Prediction"),
                "chg_5d": chgs.get("5D Prediction"),
                "pred_1m": preds.get("1M Prediction"),
                "chg_1m": chgs.get("1M Prediction"),
                "pred_3m": preds.get("3M Prediction"),
                "chg_3m": chgs.get("3M Prediction"),
                "pred_6m": preds.get("6M Prediction"),
                "chg_6m": chgs.get("6M Prediction"),
                "pred_1y": preds.get("1Y Prediction"),
                "chg_1y": chgs.get("1Y Prediction"),
            })
    return rows


def fetch_predictions_page(page: Optional[int] = None) -> Optional[str]:
    # Reálně používáme jen page=1; další stránky přes JS nejsou serverově k dispozici.
    url = COINCIDEX_BASE_URL
    try:
        resp = requests.get(url, headers=HEADERS, timeout=45)
        logging.info("[fetch] status=%s len=%s", resp.status_code, len(resp.text))
        if resp.status_code == 200:
            return resp.text
        logging.warning("[fetch] Non-200 status for %s: %s", url, resp.status_code)
    except Exception as e:
        logging.warning("[fetch] Error on %s: %s", url, e)
    return None


def iter_all_pages(max_pages: int = MAX_PAGES):
    """
    Vědomě stahujeme jen 1. stránku – server vrací pouze top 10 záznamů.
    """
    html = fetch_predictions_page(1)
    if not html:
        logging.info("[pager] No HTML for page 1.")
        return
    rows = extract_table_rows(html)
    logging.info("[pager] page=1 extracted_rows=%s", len(rows))
    for r in rows:
        yield r


# -------------------- Dedup --------------------
def dedup_items_by_symbol(items: List[Dict]) -> List[Dict]:
    """
    Odstraní duplicitní tokeny podle 'symbol'. Poslední výskyt vyhrává.
    """
    uniq: Dict[str, Dict] = {}
    for it in items:
        sym = it.get("symbol")
        if not sym:
            continue
        uniq[sym] = it
    return list(uniq.values())


# -------------------- CSV řádky --------------------
def build_active_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[str]:
    """
    Aktivní záznamy (is_active=True) – 1 řádek na (symbol, horizont) s aktuální predikcí.
    Poslední sloupec 'validation' je zatím prázdný.
    """
    rows: List[str] = []
    for it in items:
        symbol = it["symbol"]
        token_name = it.get("token_name", "")
        current_price = it.get("current_price")
        current_price_str = "" if current_price is None else str(current_price)

        pairs = [
            ("5D", it.get("pred_5d"), it.get("chg_5d"), "5D Prediction"),
            ("1M", it.get("pred_1m"), it.get("chg_1m"), "1M Prediction"),
            ("3M", it.get("pred_3m"), it.get("chg_3m"), "3M Prediction"),
            ("6M", it.get("pred_6m"), it.get("chg_6m"), "6M Prediction"),
            ("1Y", it.get("pred_1y"), it.get("chg_1y"), "1Y Prediction"),
        ]
        for label, price, pct, header_name in pairs:
            if price is None:
                continue
            _, to_fn = HORIZON_MAP[header_name]
            model_to = to_fn(scrape_date)
            pct_str = "" if pct is None else str(pct)
            rows.append(
                f"{scrape_date.isoformat()},{load_ts},{symbol},{token_name},{current_price_str},"
                f"{label},{model_to.isoformat()},{price},{pct_str},True,\n"
            )
    return rows


def build_invalidation_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[str]:
    """
    Invalidační „tombstony“ (is_active=False) – max 1× na (symbol, horizon) v rámci běhu.
    Neznáme předchozí predicted_price -> hodnoty necháme prázdné.
    """
    rows: List[str] = []
    seen = set()  # (symbol, label)
    for it in items:
        symbol = it["symbol"]
        token_name = it.get("token_name", "")
        for header_name, (label, to_fn) in HORIZON_MAP.items():
            # invaliduj jen horizonty, pro které dnes máme predikci
            col = {
                "5D Prediction": "pred_5d",
                "1M Prediction": "pred_1m",
                "3M Prediction": "pred_3m",
                "6M Prediction": "pred_6m",
                "1Y Prediction": "pred_1y",
            }[header_name]
            if it.get(col) is None:
                continue
            key = (symbol, label)
            if key in seen:
                continue
            seen.add(key)
            model_to = to_fn(scrape_date)
            rows.append(
                f"{scrape_date.isoformat()},{load_ts},{symbol},{token_name},,"
                f"{label},{model_to.isoformat()},,,False,\n"
            )
    return rows


# -------------------- Blob zápis helpery --------------------
def _append_blockblob_fallback(blob_client, header: str, lines: List[str]) -> None:
    """
    Fallback pro starší SDK bez AppendBlobClient:
    - pokud blob neexistuje -> vytvoř s hlavičkou + data
    - pokud existuje -> stáhni obsah, připoj nové řádky a nahraj s overwrite=True
    """
    from azure.core.exceptions import ResourceNotFoundError

    new_payload = "".join(lines).encode("utf-8")
    if not new_payload:
        logging.warning("[blob-fallback] No rows to append.")
        return

    try:
        try:
            stream = blob_client.download_blob()
            existing = stream.readall()
            logging.info("[blob-fallback] existing_size=%s", len(existing))
        except ResourceNotFoundError:
            existing = b""
            logging.info("[blob-fallback] blob does not exist, will create.")

        if not existing:
            payload = header.encode("utf-8") + new_payload
        else:
            if not existing.startswith(header.encode("utf-8")):
                payload = header.encode("utf-8") + existing + new_payload
            else:
                payload = existing + new_payload

        blob_client.upload_blob(payload, overwrite=True)
        logging.info("[blob-fallback] upload completed. final_size=%s", len(payload))
    except Exception as e:
        logging.error("[blob-fallback] upload error: %s", e)
        logging.error(traceback.format_exc())


def _append_in_chunks_appendblob(append_client, lines: List[str], max_chunk_bytes: int = 3_900_000) -> None:
    buf = []
    size = 0
    for line in lines:
        b = line.encode("utf-8")
        if size + len(b) > max_chunk_bytes and buf:
            append_client.append_block(b"".join(buf))
            buf, size = [], 0
        buf.append(b)
        size += len(b)
    if buf:
        append_client.append_block(b"".join(buf))


# -------------------- Azure Function entrypoint --------------------
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()  # UTC čas zápisu
    logging.info("[CoinDesk_Prediciction] Start %s", scrape_date.isoformat())
    logging.info("[env] OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s MAX_PAGES=%s", OUTPUT_CONTAINER, AZURE_BLOB_NAME, MAX_PAGES)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting.")
        return

    try:
        raw_items = list(iter_all_pages())
        items = dedup_items_by_symbol(raw_items)
        logging.info("[extract] total_items=%s unique_symbols=%s", len(raw_items), len(items))

        # 1) Tombstony (zneplatnit dnešní starší běhy) -> False (max 1× na (symbol, horizon))
        invalidate_rows = build_invalidation_rows(scrape_date, load_ts, items)
        # 2) Aktivní záznamy -> True
        active_rows = build_active_rows(scrape_date, load_ts, items)
        # 3) Pořadí: invalidace -> nové aktivní
        csv_lines = invalidate_rows + active_rows
        logging.info(
            "[csv] invalidate_rows=%s active_rows=%s total_to_append=%s",
            len(invalidate_rows), len(active_rows), len(csv_lines)
        )

        # Blob klient – bez tvrdé závislosti na AppendBlobClient
        try:
            from azure.storage.blob import BlobServiceClient
        except Exception as e:
            logging.error("[blob] import error BlobServiceClient: %s", e)
            logging.error(traceback.format_exc())
            return

        try:
            blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
            try:
                container_client.create_container()
                logging.info("[blob] container created: %s", OUTPUT_CONTAINER)
            except Exception:
                logging.info("[blob] container exists: %s", OUTPUT_CONTAINER)

            # Zkusíme AppendBlobClient, pokud je k dispozici
            try:
                from azure.storage.blob import AppendBlobClient
                append_client = AppendBlobClient.from_connection_string(
                    STORAGE_CONNECTION_STRING,
                    container_name=OUTPUT_CONTAINER,
                    blob_name=AZURE_BLOB_NAME
                )
                if not append_client.exists():
                    append_client.create_blob()
                    append_client.append_block(CSV_HEADER.encode("utf-8"))
                    logging.info("[blob] created append blob + header written")

                if csv_lines:
                    _append_in_chunks_appendblob(append_client, csv_lines)
                    logging.info("[blob] append completed via AppendBlobClient: %s rows", len(csv_lines))
                else:
                    logging.warning("[csv] No rows to append.")
                return
            except Exception as e:
                logging.warning("[blob] AppendBlobClient not available or failed (%s). Falling back to Block Blob.", e)

            # Fallback: Block Blob emulace appendu
            blob_client = container_client.get_blob_client(AZURE_BLOB_NAME)
            _append_blockblob_fallback(blob_client, CSV_HEADER, csv_lines)

        except Exception as e:
            logging.error("[blob] I/O error: %s", e)
            logging.error(traceback.format_exc())
            return

    except Exception as e:
        logging.error("[fatal] Unhandled exception in CoinDesk_Prediciction: %s", e)
        logging.error(traceback.format_exc())
        return
