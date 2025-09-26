import os
import io
import csv
import re
import time
import logging
import traceback
import datetime as dt
from typing import Dict, List, Optional

import azure.functions as func
import requests
from bs4 import BeautifulSoup  # jen pro minifikaci (volitelné)

# ===================== Konfigurace =====================
VERSION = "6.0-raw-no-parse"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Vstupní seznam tokenů z blobu
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")   # sloupce: symbol,slug,token_name

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/6.0)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

# Limit syrového HTML v CSV (aby soubor nebyl obří)
MAX_RAW_LEN = int(os.getenv("MAX_RAW_LEN", "20000"))

# CSV schema (raw výstup)
CSV_FIELDS = [
    "scrape_date", "load_ts",
    "symbol", "token_name", "slug", "source_url",
    "raw_html_minified", "raw_truncated",
    "is_active", "validation"
]

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Blob CSV I/O =====================
def load_csv_rows(container_client, blob_name: str) -> List[Dict]:
    from azure.core.exceptions import ResourceNotFoundError
    blob_client = container_client.get_blob_client(blob_name)
    try:
        content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
    except ResourceNotFoundError:
        return []
    except Exception as e:
        logging.warning("[csv-read] Failed to read existing blob: %s", e)
        return []

    rows: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        for r in reader:
            # zarovnej klíče k aktuálnímu schématu
            row = {k: r.get(k, "") for k in CSV_FIELDS}
            rows.append(row)
    dlog("[csv-read] loaded rows=%s", len(rows))
    return rows

def write_csv_rows(container_client, blob_name: str, rows: List[Dict]) -> None:
    blob_client = container_client.get_blob_client(blob_name)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for r in rows:
        for k in CSV_FIELDS:
            r.setdefault(k, "")
        writer.writerow(r)
    data = buf.getvalue().encode("utf-8")
    blob_client.upload_blob(data, overwrite=True)
    dlog("[csv-write] uploaded rows=%s size=%s", len(rows), len(data))

def deactivate_todays_rows(existing: List[Dict], today_iso: str) -> int:
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"
            changed += 1
    return changed

# ===================== CoinList =====================
def load_coinlist_from_blob(container_client, blob_name: str) -> List[Dict]:
    """Načte CoinList.csv (symbol,slug,token_name) z OUTPUT_CONTAINER."""
    blob_client = container_client.get_blob_client(blob_name)
    content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
    out: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        cols = [c.strip().lower() for c in (reader.fieldnames or [])]
        required = {"symbol", "slug", "token_name"}
        if not required.issubset(set(cols)):
            raise ValueError(f"CoinList.csv must have columns: symbol, slug, token_name (got: {cols})")
        for row in reader:
            sym = (row.get("symbol") or "").strip().upper()
            slug = (row.get("slug") or "").strip().lower()
            name = (row.get("token_name") or "").strip()
            if not sym or not slug:
                continue
            out.append({"symbol": sym, "slug": slug, "token_name": name})
    dlog("[coinlist] loaded items=%s", len(out))
    return out

# ===================== HTTP: fetch detail (bez parsování) =====================
def fetch_prediction_detail(session: requests.Session, slug: str) -> Optional[str]:
    url = DETAIL_TMPL.format(slug=slug)
    try:
        resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        dlog("[detail] slug=%s status=%s len=%s", slug, resp.status_code, len(resp.text))
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        dlog("[detail] error slug=%s: %s", slug, e)
    return None

# ===================== Minifikace HTML (jen zmenšení, bez parsování hodnot) =====================
WS_RX = re.compile(r"\s+")

def minify_html(html: str) -> str:
    """
    Lehce zmenší HTML: odmaže přebytečné whitespace a nové řádky,
    ale nechá obsah beze změny (žádné parsování hodnot).
    """
    if not html:
        return ""
    # Volitelně můžeme odstranit <script> a <style>, aby CSV nebyl přerostlý:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = str(soup)
    except Exception:
        # fallback: bez BeautifulSoup
        text = html
    # kolaps whitespace
    text = WS_RX.sub(" ", text)
    return text.strip()

# ===================== Sběr RAW dat =====================
def collect_raw_pages(coinlist: List[Dict]) -> List[Dict]:
    session = requests.Session()
    session.headers.update(HEADERS)

    out: List[Dict] = []
    for coin in coinlist:
        slug = coin["slug"]
        symbol = coin["symbol"]
        token_name = coin.get("token_name", "")
        url = DETAIL_TMPL.format(slug=slug)

        html = fetch_prediction_detail(session, slug)
        if not html:
            continue

        mini = minify_html(html)
        truncated = "False"
        if len(mini) > MAX_RAW_LEN:
            mini = mini[:MAX_RAW_LEN]
            truncated = "True"

        out.append({
            "symbol": symbol,
            "token_name": token_name,
            "slug": slug,
            "source_url": url,
            "raw_html_minified": mini,
            "raw_truncated": truncated,
        })

        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)

    dlog("[raw] collected_items=%s (from slugs=%s)", len(out), len(coinlist))
    return out

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.date.today()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s COINLIST_BLOB=%s MAX_RAW_LEN=%s",
         VERSION, OUTPUT_CONTAINER, AZURE_BLOB_NAME, COINLIST_BLOB, MAX_RAW_LEN)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting.")
        return

    try:
        from azure.storage.blob import BlobServiceClient
        bsc = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        cc = bsc.get_container_client(OUTPUT_CONTAINER)
        try:
            cc.create_container()
            dlog("[blob] container created: %s", OUTPUT_CONTAINER)
        except Exception:
            dlog("[blob] container exists: %s", OUTPUT_CONTAINER)

        # 1) načti coin list
        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return

        # 2) stáhni RAW stránky
        items = collect_raw_pages(coinlist)

        # 3) dedup dle symbolu (poslední výhra, pro jistotu)
        uniq: Dict[str, Dict] = {}
        for it in items:
            s = it.get("symbol")
            if s:
                uniq[s] = it
        items = list(uniq.values())
        dlog("[raw] unique_items=%s", len(items))

        # 4) načti existující CSV a zneaktivni dnešní řádky
        existing = load_csv_rows(cc, AZURE_BLOB_NAME)
        deact = deactivate_todays_rows(existing, scrape_date.isoformat())

        # 5) postav nové aktivní řádky (RAW)
        new_rows: List[Dict] = []
        for it in items:
            new_rows.append({
                "scrape_date": scrape_date.isoformat(),
                "load_ts": load_ts,
                "symbol": it["symbol"],
                "token_name": it.get("token_name", ""),
                "slug": it.get("slug", ""),
                "source_url": it.get("source_url", ""),
                "raw_html_minified": it.get("raw_html_minified", ""),
                "raw_truncated": it.get("raw_truncated", "False"),
                "is_active": "True",
                "validation": ""
            })

        all_rows = existing + new_rows
        dlog("[csv] deactivated_today=%s newly_active=%s final_rows=%s", deact, len(new_rows), len(all_rows))

        # 6) zapiš CSV (overwrite)
        write_csv_rows(cc, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception: %s", e)
        logging.error(traceback.format_exc())
        return
