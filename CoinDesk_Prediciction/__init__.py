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
VERSION = "5.2-static-parsefix"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Statický seznam tokenů
SYMBOL_SOURCE = os.getenv("SYMBOL_SOURCE", "STATIC").upper()
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")   # columns: symbol,slug,token_name

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/5.2)"),
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
_PRICE_RX = re.compile(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)")
_PCT_RX   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")

def _to_dec(s: str) -> Optional[Decimal]:
    try:
        return Decimal(s.replace(",", ""))
    except Exception:
        return None

def parse_price(text: str) -> Optional[Decimal]:
    if not text:
        return None
    m = _PRICE_RX.search(text)
    return _to_dec(m.group(1)) if m else None

def parse_pct(text: str) -> Optional[Decimal]:
    if not text:
        return None
    m = _PCT_RX.search(text)
    return _to_dec(m.group(1)) if m else None

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

def _find_current_price(soup: BeautifulSoup) -> Optional[Decimal]:
    """
    Hledej current price jen v kontextech, kde je explicitně 'Current price'/'Live price'/'price is'.
    Tím se vyhneme chycení '2025' v nadpise 'Price Prediction 2025'.
    """
    anchors = [
        re.compile(r"\bcurrent\s+price\b", re.I),
        re.compile(r"\blive\s+price\b", re.I),
        re.compile(r"\bprice\s+is\b", re.I),
    ]
    for rx in anchors:
        node = soup.find(string=rx)
        if not node:
            continue
        # preferuj nejbližší "řádek" nebo sekci
        container = node.find_parent(["tr","div","section","p"]) or node.parent
        if not container:
            container = node if hasattr(node, "parent") else None
        text = container.get_text(" ", strip=True) if container else str(node)
        price = parse_price(text)
        if price is not None:
            return price
    # fallback: najdi první text, kde je $ a zároveň neobsahuje 'Prediction'/'2025'
    for cand in soup.find_all(string=_PRICE_RX):
        ctxt = cand.strip()
        up = (cand.parent.get_text(" ", strip=True) if hasattr(cand, "parent") else ctxt)
        if re.search(r"prediction|202\d", up, re.I):
            continue
        price = parse_price(up)
        if price is not None:
            return price
    return None

def _extract_from_row(tr) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z jednoho <tr> zjisti cenu ($...) a procenta (%...). Ber poslední buňku s $ a první s %."""
    if tr is None:
        return None, None
    tds = tr.find_all(["td","th"])
    price_val, pct_val = None, None
    # procenta: první buňka, kde je procento
    for cell in tds:
        txt = cell.get_text(" ", strip=True)
        p = parse_pct(txt)
        if p is not None:
            pct_val = p
            break
    # cena: POSLEDNÍ buňka, kde je $
    for cell in reversed(tds):
        txt = cell.get_text(" ", strip=True)
        p = parse_price(txt)
        if p is not None:
            price_val = p
            break
    return price_val, pct_val

def _find_prediction_pair(soup: BeautifulSoup, label_regex: str) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    """
    Najdi řádek s daným label (5-Day, 1-Month, ...) a vytáhni z něj ($, %).
    Vrací (price, pct, debug_info).
    """
    # 1) najdi přesný label jako text
    node = soup.find(string=re.compile(label_regex, re.I))
    debug = "label:miss"
    if node:
        tr = node.find_parent("tr")
        if tr:
            price, pct = _extract_from_row(tr)
            debug = "row"
            if price is not None or pct is not None:
                return price, pct, debug
        # fallback: vezmi rodiče a hledej v něm
        cont = node.parent if hasattr(node, "parent") else None
        if cont:
            price = parse_price(cont.get_text(" ", strip=True))
            pct   = parse_pct(cont.get_text(" ", strip=True))
            debug = "parent"
            if price is not None or pct is not None:
                return price, pct, debug

    # 2) projdi všechny tr a hledej ten, který obsahuje label
    for tr in soup.find_all("tr"):
        t = tr.get_text(" ", strip=True)
        if re.search(label_regex, t, re.I):
            price, pct = _extract_from_row(tr)
            debug = "scan-tr"
            if price is not None or pct is not None:
                return price, pct, debug

    # 3) fallback: často jsou mini-karty/divy – najdi div/section s label a v něm $ a %
    for box in soup.find_all(["div","section","li"]):
        t = box.get_text(" ", strip=True)
        if re.search(label_regex, t, re.I):
            price = parse_price(t)
            pct   = parse_pct(t)
            debug = "scan-div"
            if price is not None or pct is not None:
                return price, pct, debug

    return None, None, debug

def parse_prediction_detail(html: str) -> Dict[str, Optional[Decimal]]:
    """
    Vytáhne current_price a predikce 5D/1M/3M/6M/1Y z detailu. Striktně vyžaduje $, % v kontextu labelu.
    """
    soup = BeautifulSoup(html, "html.parser")

    current_price = _find_current_price(soup)

    mapping = [
        ("5D", r"\b5\s*[-\s]?day\b"),
        ("1M", r"\b1\s*[-\s]?month\b"),
        ("3M", r"\b3\s*[-\s]?month\b"),
        ("6M", r"\b6\s*[-\s]?month\b"),
        ("1Y", r"\b1\s*[-\s]?year\b"),
    ]

    result: Dict[str, Optional[Decimal]] = {
        "current_price": current_price,
        "pred_5d": None, "chg_5d": None,
        "pred_1m": None, "chg_1m": None,
        "pred_3m": None, "chg_3m": None,
        "pred_6m": None, "chg_6m": None,
        "pred_1y": None, "chg_1y": None,
    }

    for short, rx in mapping:
        price, pct, dbg = _find_prediction_pair(soup, rx)
        dlog("[parse] horizon=%s dbg=%s price=%s pct=%s", short, dbg, price, pct)
        if price is not None:
            result[{"5D":"pred_5d","1M":"pred_1m","3M":"pred_3m","6M":"pred_6m","1Y":"pred_1y"}[short]] = price
        if pct is not None:
            result[{"5D":"chg_5d","1M":"chg_1m","3M":"chg_3m","6M":"chg_6m","1Y":"chg_1y"}[short]] = pct

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
