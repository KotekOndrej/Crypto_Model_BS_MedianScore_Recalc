import os
import io
import csv
import re
import time
import logging
import traceback
import datetime as dt
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace =====================
VERSION = "5.1-static-coinlist"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Statický seznam tokenů
SYMBOL_SOURCE = os.getenv("SYMBOL_SOURCE", "STATIC").upper()
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")   # musí obsahovat: symbol,slug,token_name

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/5.1)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

# Horizonty -> výpočet cílového data (model_to)
HORIZON_MAP = {
    "5D": ("5D Prediction", lambda d: d + relativedelta(days=5)),
    "1M": ("1M Prediction", lambda d: d + relativedelta(months=1)),
    "3M": ("3M Prediction", lambda d: d + relativedelta(months=3)),
    "6M": ("6M Prediction", lambda d: d + relativedelta(months=6)),
    "1Y": ("1Y Prediction", lambda d: d + relativedelta(years=1)),
}

# CSV schema (finální výstup)
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Parsování a helpery =====================
def parse_decimal_safe(s: str) -> Optional[Decimal]:
    try:
        return Decimal(s)
    except Exception:
        return None

def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z textu vytáhne cenu ($...) a změnu v % (±..%)."""
    if not text:
        return None, None
    # cena
    m_price = re.search(r"[-]?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    price_dec = parse_decimal_safe(m_price.group(1).replace(",", "")) if m_price else None
    # procenta
    m_pct = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", text)
    pct_dec = parse_decimal_safe(m_pct.group(1)) if m_pct else None
    return price_dec, pct_dec

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

def parse_prediction_detail(html: str) -> Dict[str, Optional[Decimal]]:
    """
    Z detailu predikcí vytáhne current_price a hodnoty pro 5D/1M/3M/6M/1Y + %.
    Symbol/jméno teď bereme ze statického CSV (neparsujeme je tady).
    """
    soup = BeautifulSoup(html, "html.parser")

    # current price (volitelné)
    current_price = None
    price_candidate = soup.find(string=re.compile(r"\bPrice\b", re.I))
    if price_candidate:
        section = price_candidate.parent.get_text(" ", strip=True) if hasattr(price_candidate, "parent") else str(price_candidate)
        cp, _ = parse_price_and_change(section)
        current_price = cp or current_price

    def find_row(label_regex: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        node = soup.find(string=re.compile(label_regex, re.I)) or soup.find("th", string=re.compile(label_regex, re.I))
        if not node:
            return None, None
        container = node.parent if hasattr(node, "parent") else None
        text = container.get_text(" ", strip=True) if container else str(node)
        return parse_price_and_change(text)

    mapping = [
        ("5D", "5[-\\s]?Day"),
        ("1M", "1[-\\s]?Month"),
        ("3M", "3[-\\s]?Month"),
        ("6M", "6[-\\s]?Month"),
        ("1Y", "1[-\\s]?Year"),
    ]
    result: Dict[str, Optional[Decimal]] = {
        "current_price": current_price,
        "pred_5d": None, "chg_5d": None,
        "pred_1m": None, "chg_1m": None,
        "pred_3m": None, "chg_3m": None,
        "pred_6m": None, "chg_6m": None,
        "pred_1y": None, "chg_1y": None,
    }
    label_to_key = {
        "5D": ("pred_5d", "chg_5d"),
        "1M": ("pred_1m", "chg_1m"),
        "3M": ("pred_3m", "chg_3m"),
        "6M": ("pred_6m", "chg_6m"),
        "1Y": ("pred_1y", "chg_1y"),
    }
    for short, rx in mapping:
        price, pct = find_row(rx)
        k_price, k_chg = label_to_key[short]
        if price is not None:
            result[k_price] = price
        if pct is not None:
            result[k_chg] = pct
    return result

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

# ===================== Build rows =====================
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
            ("6M", it.get("pred_6m"), it.get("chg_6m")),
            ("1Y", it.get("pred_1y"), it.get("chg_1y")),
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

def deactivate_todays_rows(existing: List[Dict], today_iso: str) -> int:
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"
            changed += 1
    return changed

# ===================== Statický seznam =====================
def load_coinlist_from_blob(container_client, blob_name: str) -> List[Dict]:
    """Čte CoinList.csv (symbol,slug,token_name) z kontejneru."""
    blob_client = container_client.get_blob_client(blob_name)
    content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
    out: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        cols = [c.strip().lower() for c in reader.fieldnames or []]
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

def collect_predictions_static(coinlist: List[Dict]) -> List[Dict]:
    """Stáhne detaily predikcí pro slugs z CoinList.csv a vrátí položky pro build_active_rows."""
    session = requests.Session()
    session.headers.update(HEADERS)

    out: List[Dict] = []
    for idx, coin in enumerate(coinlist, start=1):
        slug = coin["slug"]
        symbol = coin["symbol"]
        token_name = coin.get("token_name", "")
        html = fetch_prediction_detail(session, slug)
        if not html:
            continue
        parsed = parse_prediction_detail(html)
        has_any = any(parsed.get(k) is not None for k in ["pred_5d", "pred_1m", "pred_3m", "pred_6m", "pred_1y"])
        if not has_any:
            continue
        out.append({
            "symbol": symbol,
            "token_name": token_name,
            "current_price": parsed.get("current_price"),
            "pred_5d": parsed.get("pred_5d"), "chg_5d": parsed.get("chg_5d"),
            "pred_1m": parsed.get("pred_1m"), "chg_1m": parsed.get("chg_1m"),
            "pred_3m": parsed.get("pred_3m"), "chg_3m": parsed.get("chg_3m"),
            "pred_6m": parsed.get("pred_6m"), "chg_6m": parsed.get("chg_6m"),
            "pred_1y": parsed.get("pred_1y"), "chg_1y": parsed.get("chg_1y"),
        })
        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)
    dlog("[detail] collected_items=%s (static list size=%s)", len(out), len(coinlist))
    return out

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s SYMBOL_SOURCE=%s OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s COINLIST_BLOB=%s",
         VERSION, SYMBOL_SOURCE, OUTPUT_CONTAINER, AZURE_BLOB_NAME, COINLIST_BLOB)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting.")
        return

    try:
        from azure.storage.blob import BlobServiceClient
        bsc = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        cc = bsc.get_container_client(OUTPUT_CONTAINER)
        # create_container může hodit ResourceExistsError (409) – ošetříme
        try:
            cc.create_container()
            dlog("[blob] container created: %s", OUTPUT_CONTAINER)
        except Exception:
            dlog("[blob] container exists: %s", OUTPUT_CONTAINER)

        # --- STATIC režim: načti CoinList.csv a stáhni predikce pro každý slug ---
        if SYMBOL_SOURCE != "STATIC":
            dlog("[warn] SYMBOL_SOURCE=%s != STATIC -> pokračuji stejně jako STATIC.", SYMBOL_SOURCE)

        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return

        items = collect_predictions_static(coinlist)

        # dedup dle symbolu (poslední výhra)
        uniq: Dict[str, Dict] = {}
        for it in items:
            s = it.get("symbol")
            if s:
                uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_items=%s", len(items))

        # --- CSV overwrite workflow ---
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
