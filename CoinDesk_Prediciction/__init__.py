import os
import io
import re
import csv
import json
import time
import logging
import traceback
import datetime as dt
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace =====================
VERSION = "9.0-coincodex-table+faqjson"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")  # povinné
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Coin list (musí mít sloupce: symbol, slug, token_name)
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/9.0)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

# CSV schema
CSV_FIELDS = [
    "scrape_date", "load_ts",
    "symbol", "slug", "token_name",
    "current_price",
    "horizon", "model_to",
    "predicted_price", "predicted_change_pct",
    "page_url", "html_len", "html_blob",
    "is_active", "validation",
]

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Utility: čísla/normalizace =====================
_MOJIBAKE = ("Ã¢Â€Â¯", "Ã‚Â ", "Ã¢â‚¬â€", "Ã¢â‚¬â„¢", "â€“", "â€”")
_WS_CHARS = ["\u00A0", "\u202F", "\u2009", "\u2007", "\u200A"]

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for bad in _MOJIBAKE:
        s = s.replace(bad, " ")
    for ch in _WS_CHARS:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

_MONEY_ANY = re.compile(r'[\d\.,]+')
_PCT_RE    = re.compile(r'([+\-]?\d+(?:\.\d+)?)\s*%')
_USD_RE    = re.compile(r'\$\s*([\d\.,]+)')
_DATE_RE   = re.compile(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})')

def to_decimal(num: Optional[str | float]) -> Optional[Decimal]:
    if num is None:
        return None
    if isinstance(num, float):
        try:
            return Decimal(str(num))
        except InvalidOperation:
            return None
    s = str(num)
    s = normalize_text(s)
    s = s.replace(",", "").replace(" ", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def clean_money_from_text(s: str) -> Optional[Decimal]:
    if not s:
        return None
    s = normalize_text(s)
    m = _MONEY_ANY.findall(s)
    if not m:
        return None
    return to_decimal(m[-1])

def pct_from_prices(pred: Optional[Decimal], base: Optional[Decimal]) -> Optional[Decimal]:
    try:
        if pred is None or base is None or base == 0:
            return None
        return (pred - base) * Decimal(100) / base
    except Exception:
        return None

# ===================== HTTP =====================
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

# ===================== Upload full HTML =====================
def upload_full_html(container_client, slug: str, scrape_date: dt.date, load_ts: str, html: str) -> str:
    """
    Uloží celé HTML do blobu: html/{slug}/{YYYY-MM-DD}/{HHMMSS}.html
    Vrací blob path (bez SAS).
    """
    safe_slug = re.sub(r"[^a-z0-9\-]", "-", slug.lower())
    ts = load_ts.replace(":", "").replace("-", "")
    hhmmss = ts.split("T")[1].split(".")[0] if "T" in ts else ts[-6:]
    blob_path = f"html/{safe_slug}/{scrape_date.isoformat()}/{hhmmss}.html"
    bc = container_client.get_blob_client(blob_path)
    bc.upload_blob(html.encode("utf-8"), overwrite=True)
    dlog("[html] saved -> %s", blob_path)
    return blob_path

# ===================== Parsování: tabulka =====================
def parse_table_prices(html: str) -> Dict[str, Optional[Decimal]]:
    """
    Z tabulky table.table-grid.prediction-data-table vytáhne:
      - current_price (tr.data-current-price > td)
      - predicted_price (tr.data-predicted-price > td)
    """
    soup = BeautifulSoup(html, "html.parser")
    out = {"current_price": None, "predicted_price": None}
    tbl = soup.select_one("table.table-grid.prediction-data-table")
    if not tbl:
        return out

    curr_td = tbl.select_one("tr.data-current-price > td")
    pred_td = tbl.select_one("tr.data-predicted-price > td")

    if curr_td:
        out["current_price"] = clean_money_from_text(curr_td.get_text(" ", strip=True))
    if pred_td:
        out["predicted_price"] = clean_money_from_text(pred_td.get_text(" ", strip=True))

    return out

# ===================== Parsování: FAQ JSON-LD (1M/6M/1Y) =====================
def _match_first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text) if text else None
    return m.group(1) if m else None

def parse_faq_periods(html: str) -> Dict[str, Optional[Decimal | str]]:
    """
    Z <script id="faq" type="application/ld+json"> přečte textové odpovědi a
    vytáhne predikce pro 1M, 6M, 1Y (cena, % a cílové datum, pokud je uveden).
    Klíče: 1M_predicted_price, 1M_predicted_change_pct, 1M_target_date, atd.
    """
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one('script#faq[type="application/ld+json"]')
    out: Dict[str, Optional[Decimal | str]] = {}
    if not tag or not tag.string:
        return out

    try:
        data = json.loads(tag.string)
    except Exception:
        return out

    qas = data.get("mainEntity", [])
    if not isinstance(qas, list):
        return out

    def fill(prefix: str, text: str):
        usd = _match_first(_USD_RE, text)
        pct = _match_first(_PCT_RE, text)
        dat = _match_first(_DATE_RE, text)
        if usd:
            out[f"{prefix}_predicted_price"] = to_decimal(usd)
        if pct is not None:
            try:
                out[f"{prefix}_predicted_change_pct"] = to_decimal(pct)
            except Exception:
                pass
        if dat:
            out[f"{prefix}_target_date"] = dat

    for qa in qas:
        txt = qa.get("acceptedAnswer", {}).get("text", "") or ""
        tnorm = txt.lower()

        # 1M
        if ("next month" in tnorm) or ("the next month" in tnorm) or ("one month" in tnorm):
            fill("1M", txt)

        # 6M
        if ("six months" in tnorm) or ("the next six months" in tnorm) or ("6 months" in tnorm):
            fill("6M", txt)

        # 1Y
        if ("one year" in tnorm) or ("in one year" in tnorm) or ("12 months" in tnorm):
            fill("1Y", txt)

    return out

# ===================== CSV I/O =====================
def load_csv_rows(container_client, blob_name: str) -> List[Dict]:
    from azure.core.exceptions import ResourceNotFoundError
    bc = container_client.get_blob_client(blob_name)
    try:
        content = bc.download_blob().readall().decode("utf-8", errors="ignore")
    except ResourceNotFoundError:
        return []
    except Exception as e:
        logging.warning("[csv-read] Failed to read existing blob: %s", e)
        return []

    rows: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {k: r.get(k, "") for k in CSV_FIELDS}
            rows.append(row)
    dlog("[csv-read] loaded rows=%s", len(rows))
    return rows

def write_csv_rows(container_client, blob_name: str, rows: List[Dict]) -> None:
    bc = container_client.get_blob_client(blob_name)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        for k in CSV_FIELDS:
            r.setdefault(k, "")
        writer.writerow(r)
    data = buf.getvalue().encode("utf-8")
    bc.upload_blob(data, overwrite=True)
    dlog("[csv-write] uploaded rows=%s size=%s", len(rows), len(data))

# ===================== CoinList (se slugy) =====================
def load_coinlist_from_blob(container_client, blob_name: str) -> List[Dict]:
    """Načte CoinList.csv (sloupce: symbol, slug, token_name) z OUTPUT_CONTAINER."""
    bc = container_client.get_blob_client(blob_name)
    content = bc.download_blob().readall().decode("utf-8", errors="ignore")
    out: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        cols = [c.strip().lower() for c in (reader.fieldnames or [])]
        required = {"symbol", "slug", "token_name"}
        if not required.issubset(set(cols)):
            raise ValueError(f"CoinList.csv must have columns: symbol, slug, token_name (got: {cols})")
        for row in reader:
            sym  = (row.get("symbol") or "").strip().upper()
            slug = (row.get("slug") or "").strip().lower()
            name = (row.get("token_name") or "").strip()
            if not sym or not slug:
                continue
            out.append({"symbol": sym, "slug": slug, "token_name": name})
    dlog("[coinlist] loaded items=%s", len(out))
    return out

# ===================== Sběr dat přes slug detail =====================
def collect_predictions_by_slug(coinlist: List[Dict], container_client, scrape_date: dt.date, load_ts: str) -> List[Dict]:
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

        # uložit celé HTML
        blob_path = upload_full_html(container_client, slug, scrape_date, load_ts, html)

        # tabulka (current + “table predicted”)
        table_vals = parse_table_prices(html)
        current_price = table_vals.get("current_price")
        table_pred = table_vals.get("predicted_price")

        # JSON-LD (1M/6M/1Y)
        faq_vals = parse_faq_periods(html)

        item = {
            "symbol": symbol,
            "slug": slug,
            "token_name": token_name,
            "page_url": url,
            "html_len": len(html),
            "html_blob": blob_path,
            "current_price": current_price,
            "table_predicted_price": table_pred,
            "faq": faq_vals,
        }
        out.append(item)

        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)

    dlog("[detail] collected_items=%s (from slugs=%s)", len(out), len(coinlist))
    return out

# ===================== Build rows =====================
def build_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
    """
    Vytvoří řádky:
      - TABLE (predikce z tabulky)
      - 1M / 6M / 1Y (pokud jsou ve FAQ)
    """
    rows: List[Dict] = []

    def mkrow(symbol, slug, name, current_price, horizon, model_to, pred_price, pred_pct, page_url, html_len, html_blob):
        return {
            "scrape_date": scrape_date.isoformat(),
            "load_ts": load_ts,
            "symbol": symbol,
            "slug": slug,
            "token_name": name,
            "current_price": "" if current_price is None else str(current_price),
            "horizon": horizon,
            "model_to": model_to or "",
            "predicted_price": "" if pred_price is None else str(pred_price),
            "predicted_change_pct": "" if pred_pct is None else str(pred_pct),
            "page_url": page_url,
            "html_len": str(html_len),
            "html_blob": html_blob,
            "is_active": "True",
            "validation": ""
        }

    for it in items:
        sym  = it["symbol"]
        slug = it["slug"]
        name = it.get("token_name", "")
        url  = it.get("page_url", "")
        hlen = it.get("html_len", 0)
        hkey = it.get("html_blob", "")

        curr = to_decimal(it.get("current_price"))
        table_pred = to_decimal(it.get("table_predicted_price"))

        # 1) TABLE horizon (pokud existuje aspoň predikce)
        if table_pred is not None:
            pct = pct_from_prices(table_pred, curr)
            rows.append(mkrow(sym, slug, name, curr, "TABLE", "", table_pred, pct, url, hlen, hkey))

        # 2) FAQ horizons (1M/6M/1Y)
        faq = it.get("faq", {}) or {}

        def put(prefix: str, months: int):
            price = to_decimal(faq.get(f"{prefix}_predicted_price"))
            pct   = to_decimal(faq.get(f"{prefix}_predicted_change_pct"))
            target_date = (faq.get(f"{prefix}_target_date") or "").strip()
            model_to = ""
            if target_date:
                # zkus převést "Oct 27, 2025" -> ISO
                try:
                    model_to = dt.datetime.strptime(target_date, "%b %d, %Y").date().isoformat()
                except Exception:
                    model_to = ""
            if not model_to:
                # fallback: scrape_date + months
                model_to = (scrape_date + relativedelta(months=months)).isoformat()
            if price is None and pct is None:
                return
            # pokud chybí % ale máme obě ceny, dopočítej
            if pct is None:
                pct = pct_from_prices(price, curr)
            rows.append(mkrow(sym, slug, name, curr, prefix, model_to, price, pct, url, hlen, hkey))

        if "1M_predicted_price" in faq or "1M_predicted_change_pct" in faq:
            put("1M", 1)
        if "6M_predicted_price" in faq or "6M_predicted_change_pct" in faq:
            put("6M", 6)
        if "1Y_predicted_price" in faq or "1Y_predicted_change_pct" in faq:
            put("1Y", 12)

    return rows

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.date.today()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s COINLIST_BLOB=%s",
         VERSION, OUTPUT_CONTAINER, AZURE_BLOB_NAME, COINLIST_BLOB)

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

        # 1) načti CoinList (se slugy)
        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return

        # 2) stáhni a ulož detail + full HTML
        items = collect_predictions_by_slug(coinlist, cc, scrape_date, load_ts)

        # 3) sestav řádky
        new_rows = build_rows(scrape_date, load_ts, items)

        # 4) načti dosavadní CSV a připiš nové řádky (bez deaktivace – chceme historii)
        existing = load_csv_rows(cc, AZURE_BLOB_NAME)
        all_rows = existing + new_rows
        dlog("[csv] newly_added=%s final_rows=%s", len(new_rows), len(all_rows))
        write_csv_rows(cc, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception: %s", e)
        logging.error(traceback.format_exc())
        return
