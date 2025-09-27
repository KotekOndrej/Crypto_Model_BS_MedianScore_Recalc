# -*- coding: utf-8 -*-
import os
import io
import re
import csv
import time
import logging
import traceback
import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple, Set

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace =====================
VERSION = "7.0-index-parser-5D-1M-3M"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
PAGE_SLEEP_MS = int(os.getenv("PAGE_SLEEP_MS", "250"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/7.0)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Index s tabulkou predikcí
PREDICTIONS_INDEX = "https://coincodex.com/predictions/"
MAX_INDEX_PAGES = int(os.getenv("MAX_INDEX_PAGES", "10"))

# Pouze 5D / 1M / 3M
HORIZON_MAP = {
    "5D": ("5D Prediction", lambda d: d + relativedelta(days=5)),
    "1M": ("1M Prediction", lambda d: d + relativedelta(months=1)),
    "3M": ("3M Prediction", lambda d: d + relativedelta(months=3)),
}

# CSV schema (beze změn)
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Regexy pro čísla =====================
_RX_PRICE = re.compile(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)")
_RX_PCT   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")

def _to_dec(s: Optional[str]) -> Optional[Decimal]:
    if s is None:
        return None
    try:
        return Decimal(s.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None

def _parse_price_and_pct(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    if not text:
        return None, None
    mp = _RX_PRICE.search(text)
    mc = _RX_PCT.search(text)
    price = _to_dec(mp.group(1)) if mp else None
    pct   = _to_dec(mc.group(1)) if mc else None
    return price, pct

# ===================== Parsování /predictions/ =====================
def _find_table_with_predictions(soup: BeautifulSoup):
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if headers and any("Prediction" in h for h in headers):
            return t, headers
    return None, None

def _find_col(headers: List[str], header_idx: Dict[str, int], candidates: List[object]) -> Optional[int]:
    # kandidáti mohou být řetězce (exact) nebo regex patterny
    for c in candidates:
        if isinstance(c, str) and c in header_idx:
            return header_idx[c]
    # regex fallback
    for i, h in enumerate(headers):
        for c in candidates:
            if hasattr(c, "pattern") and re.search(c, h, re.I):
                return i
        # string fallback case-insensitive
        for c in candidates:
            if isinstance(c, str) and h.lower() == c.lower():
                return i
    return None

def parse_predictions_index(html: str, wanted_symbols: Set[str]) -> Dict[str, Dict]:
    """
    Vrátí {symbol -> {token_name, current_price, pred_5d, chg_5d, pred_1m, chg_1m, pred_3m, chg_3m}}
    pouze pro tickery z wanted_symbols.
    """
    soup = BeautifulSoup(html, "html.parser")
    table, headers = _find_table_with_predictions(soup)
    if not table:
        dlog("[index-parse] table not found")
        return {}

    header_idx = {name: i for i, name in enumerate(headers)}

    name_col  = _find_col(headers, header_idx, ["Name", re.compile(r"\bName\b", re.I), re.compile(r"\bCoin\b", re.I)])
    price_col = _find_col(headers, header_idx, ["Price", "Current Price", re.compile(r"\bPrice\b", re.I)])

    col_5d = _find_col(headers, header_idx, ["5D Prediction", re.compile(r"\b5\s*D\b", re.I), re.compile(r"\b5\s*[- ]*Day", re.I)])
    col_1m = _find_col(headers, header_idx, ["1M Prediction", re.compile(r"\b1\s*M\b", re.I), re.compile(r"\b1\s*[- ]*Month", re.I)])
    col_3m = _find_col(headers, header_idx, ["3M Prediction", re.compile(r"\b3\s*M\b", re.I), re.compile(r"\b3\s*[- ]*Month", re.I)])

    tbody = table.find("tbody") or table
    found: Dict[str, Dict] = {}
    want = {w.upper() for w in wanted_symbols}

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # symbol + token_name z buňky Name
        symbol = None
        token_name = ""
        if name_col is not None and name_col < len(tds):
            name_text = tds[name_col].get_text(" ", strip=True)
            m = re.search(r"\b[A-Z0-9]{2,12}\b", name_text)
            if m:
                symbol = m.group(0).upper()
                token_name = name_text.replace(symbol, "", 1).strip(" -–—·,")

        if not symbol or symbol not in want:
            continue

        # current price
        current_price = None
        if price_col is not None and price_col < len(tds):
            cp_text = tds[price_col].get_text(" ", strip=True)
            current_price, _ = _parse_price_and_pct(cp_text)

        # horizonty
        def cell(ix: Optional[int]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
            if ix is None or ix >= len(tds):
                return None, None
            return _parse_price_and_pct(tds[ix].get_text(" ", strip=True))

        p5d, c5d = cell(col_5d)
        p1m, c1m = cell(col_1m)
        p3m, c3m = cell(col_3m)

        found[symbol] = {
            "symbol": symbol,
            "token_name": token_name,
            "current_price": current_price,
            "pred_5d": p5d, "chg_5d": c5d,
            "pred_1m": p1m, "chg_1m": c1m,
            "pred_3m": p3m, "chg_3m": c3m,
        }

    dlog("[index-parse] matched=%s", len(found))
    return found

# ===================== CSV I/O =====================
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
            rows.append({k: r.get(k, "") for k in CSV_FIELDS})
    dlog("[csv-read] loaded rows=%s", len(rows))
    return rows

def write_csv_rows(container_client, blob_name: str, rows: List[Dict]) -> None:
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
    dlog("[csv-write] uploaded rows=%s size=%s", len(rows), len(data))

def deactivate_todays_rows(existing: List[Dict], today_iso: str) -> int:
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"
            changed += 1
    return changed

# ===================== Build rows (jen 5D/1M/3M) =====================
def build_active_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for it in items:
        symbol = it["symbol"]
        token_name = it.get("token_name", "")
        current_price = it.get("current_price")
        current_price_str = "" if current_price is None else str(current_price)

        pairs = [
            ("5D", it.get("pred_5d"), it.get("chg_5d")),
            ("1M", it.get("pred_1m"), it.get("chg_1m")),
            ("3M", it.get("pred_3m"), it.get("chg_3m")),
        ]
        for short, price, pct in pairs:
            if price is None:
                continue
            _, to_fn = HORIZON_MAP[short]
            model_to = to_fn(scrape_date)
            rows.append({
                "scrape_date": scrape_date.isoformat(),
                "load_ts": load_ts,
                "symbol": symbol,
                "token_name": token_name,
                "current_price": current_price_str,
                "horizon": short,
                "model_to": model_to.isoformat(),
                "predicted_price": str(price),
                "predicted_change_pct": "" if pct is None else str(pct),
                "is_active": "True",
                "validation": ""
            })
    return rows

# ===================== CoinList =====================
def load_coinlist_from_blob(container_client, blob_name: str) -> List[Dict]:
    """Čte CoinList.csv (symbol,slug,token_name) z OUTPUT_CONTAINER (slug se tu nepoužívá, ale necháváme jej v CSV)."""
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
            name = (row.get("token_name") or "").strip()
            if not sym:
                continue
            out.append({"symbol": sym, "token_name": name})
    dlog("[coinlist] loaded symbols=%s", len(out))
    return out

# ===================== Fetch index pages =====================
def fetch_index_page(session: requests.Session, page: int) -> Optional[str]:
    url = PREDICTIONS_INDEX if page == 1 else f"{PREDICTIONS_INDEX}?page={page}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        dlog("[index] page=%s status=%s len=%s", page, resp.status_code, len(resp.text))
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        dlog("[index] error page=%s: %s", page, e)
    return None

def collect_from_index_for_targets(target_symbols: Set[str]) -> List[Dict]:
    """
    Projde 1..MAX_INDEX_PAGES a vrátí list položek pro požadované symboly.
    Jakmile najdeme všechny, končíme.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    pending = {s.upper() for s in target_symbols}
    found: Dict[str, Dict] = {}

    for page in range(1, MAX_INDEX_PAGES + 1):
        html = fetch_index_page(session, page)
        if not html:
            break
        page_found = parse_predictions_index(html, wanted_symbols=pending)
        for sym, item in page_found.items():
            if sym not in found:
                found[sym] = item
        pending -= set(page_found.keys())
        dlog("[index] page=%s matched_now=%s remaining=%s", page, len(page_found), len(pending))
        if not pending:
            break
        if PAGE_SLEEP_MS > 0:
            time.sleep(PAGE_SLEEP_MS / 1000.0)

    return list(found.values())

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.date.today()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s COINLIST_BLOB=%s MAX_INDEX_PAGES=%s",
         VERSION, OUTPUT_CONTAINER, AZURE_BLOB_NAME, COINLIST_BLOB, MAX_INDEX_PAGES)

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

        # 1) načti cílové symboly (z CoinList)
        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return
        target_symbols = {c["symbol"] for c in coinlist}
        name_lookup = {c["symbol"]: c.get("token_name","") for c in coinlist}

        # 2) nasbírej data z indexu /predictions/
        items = collect_from_index_for_targets(target_symbols)

        # 3) doplň token_name ze seznamu
        for it in items:
            it["token_name"] = name_lookup.get(it["symbol"], it.get("token_name",""))

        # 4) dedup (poslední výhra)
        uniq: Dict[str, Dict] = {}
        for it in items:
            s = it.get("symbol")
            if s: uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_items=%s", len(items))

        # 5) CSV overwrite workflow
        existing = load_csv_rows(cc, AZURE_BLOB_NAME)
        deact = deactivate_todays_rows(existing, scrape_date.isoformat())
        new_rows = build_active_rows(scrape_date, load_ts, items)
        all_rows = existing + new_rows
        dlog("[csv] deactivated_today=%s newly_active=%s final_rows=%s", deact, len(new_rows), len(all_rows))
        write_csv_rows(cc, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception: %s", e)
        logging.error(traceback.format_exc())
        return
