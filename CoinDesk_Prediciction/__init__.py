import datetime as dt
import logging
import os
import re
import traceback
from decimal import Decimal, InvalidOperation
from typing import Tuple, List, Dict, Optional
import io
import csv
import time
import hashlib

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# -------------------- Konfigurace --------------------
COINCIDEX_BASE_URL = "https://coincodex.com/predictions/"
USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/3.1)")
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")  # storage Function Appu
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")  # výchozí název CSV

# Stránkování: zkusíme page=1..MAX_PAGES; zastavíme, když nic nepřibývá
MAX_PAGES = int(os.getenv("MAX_PAGES", "20"))
PAGE_SLEEP_MS = int(os.getenv("PAGE_SLEEP_MS", "0"))  # např. 200–500 ms pokud by web škrtal

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

# CSV hlavička & pořadí polí
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]
CSV_HEADER = ",".join(CSV_FIELDS) + "\n"

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


def fetch_predictions_page(page: int) -> Optional[str]:
    """
    Zkusí načíst stránku s parametrem ?page=N. Pro page=1 použij základní URL.
    Některé verze webu vrací stále první stránku — ošetříme v iterátoru.
    """
    url = COINCIDEX_BASE_URL if page == 1 else f"{COINCIDEX_BASE_URL}?page={page}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=45)
        logging.info("[fetch] page=%s status=%s len=%s", page, resp.status_code, len(resp.text))
        if resp.status_code == 200:
            return resp.text
        logging.warning("[fetch] Non-200 status for %s: %s", url, resp.status_code)
    except Exception as e:
        logging.warning("[fetch] Error on %s: %s", url, e)
    return None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def iter_all_items() -> List[Dict]:
    """
    Iteruje stránkami ?page=1..MAX_PAGES, agreguje a deduplikuje tokeny.
    Zastaví se, pokud:
      - extract_table_rows vrátí 0 řádků, nebo
      - hash HTML je stejný jako u předchozí stránky (pravděpodobně pořád 1. stránka), nebo
      - nově přidané tokeny = 0 (už nic nepřibývá).
    """
    all_items: List[Dict] = []
    seen_symbols = set()
    prev_hash = None

    for page in range(1, MAX_PAGES + 1):
        html = fetch_predictions_page(page)
        if not html:
            logging.info("[pager] empty html, stop at page %s", page)
            break

        h = hash_text(html)
        if prev_hash is not None and h == prev_hash:
            logging.info("[pager] html identical to previous page (%s==%s), stop.", page, page-1)
            break
        prev_hash = h

        rows = extract_table_rows(html)
        logging.info("[pager] page=%s extracted_rows=%s", page, len(rows))
        if not rows:
            logging.info("[pager] no rows on page %s, stop.", page)
            break

        added = 0
        for it in rows:
            sym = it.get("symbol")
            if not sym or sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            all_items.append(it)
            added += 1

        logging.info("[pager] page=%s newly_added_symbols=%s total_symbols=%s", page, added, len(all_items))
        if added == 0:
            logging.info("[pager] no new symbols, stop.")
            break

        if PAGE_SLEEP_MS > 0:
            time.sleep(PAGE_SLEEP_MS / 1000.0)

    return all_items


def dedup_items_by_symbol(items: List[Dict]) -> List[Dict]:
    """Bezpečnostní deduplikace podle 'symbol' (poslední výskyt vyhrává)."""
    uniq: Dict[str, Dict] = {}
    for it in items:
        sym = it.get("symbol")
        if not sym:
            continue
        uniq[sym] = it
    return list(uniq.values())


# -------------------- CSV I/O --------------------
def load_csv_rows(container_client, blob_name: str) -> List[Dict]:
    """Načte CSV z Blobu do list(dict) se stejným pořadím polí jako CSV_FIELDS. Když neexistuje, vrátí []."""
    from azure.core.exceptions import ResourceNotFoundError
    blob_client = container_client.get_blob_client(blob_name)
    try:
        stream = blob_client.download_blob()
        content = stream.readall().decode("utf-8", errors="ignore")
    except ResourceNotFoundError:
        return []
    except Exception as e:
        logging.warning("[csv-read] Failed to read existing blob: %s", e)
        return []

    rows: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {k: r.get(k, "") for k in CSV_FIELDS}
            rows.append(out)
    logging.info("[csv-read] loaded rows=%s", len(rows))
    return rows


def write_csv_rows(container_client, blob_name: str, rows: List[Dict]) -> None:
    """Zapíše celý CSV (s hlavičkou) s overwrite=True."""
    blob_client = container_client.get_blob_client(blob_name)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        for k in CSV_FIELDS:
            r.setdefault(k, "")
        writer.writerow(r)
    data = buf.getvalue().encode("utf-8")
    blob_client.upload_blob(data, overwrite=True)
    logging.info("[csv-write] uploaded rows=%s size=%s", len(rows), len(data))


# -------------------- CSV tvorba dnešních řádků --------------------
def build_active_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
    """Aktivní dnešní záznamy (is_active=True) – 1 řádek na (symbol, horizont) s aktuální predikcí."""
    rows: List[Dict] = []
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
            rows.append({
                "scrape_date": scrape_date.isoformat(),
                "load_ts": load_ts,
                "symbol": symbol,
                "token_name": token_name,
                "current_price": current_price_str,
                "horizon": label,
                "model_to": model_to.isoformat(),
                "predicted_price": str(price),
                "predicted_change_pct": "" if pct is None else str(pct),
                "is_active": "True",
                "validation": ""
            })
    return rows


def deactivate_todays_rows(existing: List[Dict], today_iso: str) -> int:
    """V existujícím CSV nastaví všem dnešním řádkům is_active=False (bez přidávání nových řádků)."""
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"
            changed += 1
    return changed


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
        # 1) Extrakce z více stránek (+ dedup)
        items = iter_all_items()
        items = dedup_items_by_symbol(items)
        logging.info("[extract] unique_symbols=%s", len(items))

        # 2) Blob klient
        try:
            from azure.storage.blob import BlobServiceClient
        except Exception as e:
            logging.error("[blob] import error BlobServiceClient: %s", e)
            logging.error(traceback.format_exc())
            return

        blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
        try:
            container_client.create_container()
            logging.info("[blob] container created: %s", OUTPUT_CONTAINER)
        except Exception:
            logging.info("[blob] container exists: %s", OUTPUT_CONTAINER)

        # 3) Načti existující CSV, deaktivuj dnešní True a přidej nové aktivní
        existing_rows = load_csv_rows(container_client, AZURE_BLOB_NAME)
        deactivated = deactivate_todays_rows(existing_rows, scrape_date.isoformat())
        new_active_rows = build_active_rows(scrape_date, load_ts, items)
        all_rows = existing_rows + new_active_rows
        logging.info("[csv] deactivated_today=%s newly_active=%s final_rows=%s", deactivated, len(new_active_rows), len(all_rows))

        # 4) Zapiš celý CSV s overwrite=True
        write_csv_rows(container_client, AZURE_BLOB_NAME, all_rows)
        logging.info("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception in CoinDesk_Prediciction: %s", e)
        logging.error(traceback.format_exc())
        return
