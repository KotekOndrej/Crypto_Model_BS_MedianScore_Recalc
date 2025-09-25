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
USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/1.6)")
TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Prague")  # informativní
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")  # použij storage Function Appu
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "stgbinancedata")  # default požadovaný
MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))

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

# CSV hlavička – append-only
CSV_HEADER = "scrape_date,symbol,token_name,horizon,model_to,predicted_price,predicted_change_pct\n"


# -------------------- Parser helpery --------------------
def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
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
    url = COINCIDEX_BASE_URL if not page or page == 1 else f"{COINCIDEX_BASE_URL}?page={page}"
    for attempt in range(1, 3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
            logging.info("[fetch] page=%s status=%s len=%s attempt=%s", page or 1, resp.status_code, len(resp.text), attempt)
            if resp.status_code == 200:
                return resp.text
            else:
                logging.warning("[fetch] Non-200 status for %s: %s", url, resp.status_code)
        except Exception as e:
            logging.warning("[fetch] Error on %s (attempt %s): %s", url, attempt, e)
    return None


def iter_all_pages(max_pages: int = MAX_PAGES):
    for p in range(1, max_pages + 1):
        html = fetch_predictions_page(p)
        if not html:
            logging.info("[pager] No HTML for page %s, stop.", p)
            break
        rows = extract_table_rows(html)
        logging.info("[pager] page=%s extracted_rows=%s", p, len(rows))
        if not rows:
            logging.info("[pager] Empty rows on page %s, stop.", p)
            break
        yield from rows


def build_csv_rows(scrape_date: dt.date, items: List[Dict]) -> List[str]:
    rows: List[str] = []
    for it in items:
        symbol = it["symbol"]
        token_name = it.get("token_name", "")
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
                f"{scrape_date.isoformat()},{symbol},{token_name},{label},{model_to.isoformat()},{price},{pct_str}\n"
            )
    return rows


def _append_blockblob_fallback(blob_client, header: str, lines: List[str]) -> None:
    """
    Fallback pro starší SDK bez AppendBlobClient:
    - pokud blob neexistuje -> vytvoř s hlavičkou + data
    - pokud existuje -> stáhni obsah, připoj nové řádky a nahraj s overwrite=True
    Pozn.: pro běžné denní dávky je to naprosto OK.
    """
    from azure.core.exceptions import ResourceNotFoundError

    new_payload = "".join(lines).encode("utf-8")
    if not new_payload:
        logging.warning("[blob-fallback] No rows to append.")
        return

    try:
        # stáhni existující obsah (pokud je)
        try:
            stream = blob_client.download_blob()
            existing = stream.readall()
            logging.info("[blob-fallback] existing_size=%s", len(existing))
        except ResourceNotFoundError:
            existing = b""
            logging.info("[blob-fallback] blob does not exist, will create.")

        if not existing:
            # vytvořit s hlavičkou
            payload = header.encode("utf-8") + new_payload
        else:
            # zkontroluj, zda už má hlavičku
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
    logging.info("[CoinDesk_Prediciction] Start %s", scrape_date.isoformat())
    logging.info("[env] OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s MAX_PAGES=%s", OUTPUT_CONTAINER, AZURE_BLOB_NAME, MAX_PAGES)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting.")
        return

    try:
        all_items = list(iter_all_pages())
        logging.info("[extract] total_items=%s", len(all_items))

        csv_lines = build_csv_rows(scrape_date, all_items)
        logging.info("[csv] lines_to_append=%s", len(csv_lines))

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

            # Zkusíme použít AppendBlobClient, pokud je k dispozici
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
