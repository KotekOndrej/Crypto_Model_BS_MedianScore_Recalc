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
VERSION = "5.4-static-targeted-parsers"

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
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/5.4)"),
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

# ===================== Regexy a helpery =====================
PRICE_RX = re.compile(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)")
PCT_RX   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")
PAIR_RX  = re.compile(
    r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)(?:(?!\$).){0,160}?([+\-]?\d+(?:\.\d+)?)\s*%",
    re.S,
)

SAFE_LABEL_BLOCKS = ("tr","li","div","section","article","p")

def _to_dec(num_str: str) -> Optional[Decimal]:
    try:
        if num_str is None:
            return None
        return Decimal(str(num_str).replace(",", ""))
    except Exception:
        return None

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el is not None else ""

def parse_price(text: str) -> Optional[Decimal]:
    if not text:
        return None
    m = PRICE_RX.search(text)
    return _to_dec(m.group(1)) if m else None

def parse_pct(text: str) -> Optional[Decimal]:
    if not text:
        return None
    m = PCT_RX.search(text)
    return _to_dec(m.group(1)) if m else None

def pct_from_prices(pred: Optional[Decimal], base: Optional[Decimal]) -> Optional[Decimal]:
    if pred is None or base is None:
        return None
    try:
        if base == 0:
            return None
        return (pred - base) * Decimal(100) / base
    except Exception:
        return None

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

def _nearest_label_block(node):
    """Najdi nejbližší 'řádek/box' obsahující čísla pro daný label."""
    if not node:
        return None
    # preferuj <tr>
    for anc in getattr(node, "parents", []):
        if getattr(anc, "name", "").lower() == "tr":
            return anc
    # další „boxy“
    for anc in getattr(node, "parents", []):
        if getattr(anc, "name", "").lower() in SAFE_LABEL_BLOCKS:
            t = _text(anc)
            if "$" in t or "%" in t:
                return anc
    return node.parent if hasattr(node, "parent") else None

def extract_current_price(soup: BeautifulSoup) -> Optional[Decimal]:
    """Hledej 'Current price' apod., ignoruj bloky s 'prediction/forecast/20xx'."""
    anchors = [
        re.compile(r"\bcurrent\s+price\b", re.I),
        re.compile(r"\blive\s+price\b", re.I),
        re.compile(r"\bprice\s+is\b", re.I),
        re.compile(r"\bprice\b", re.I),
    ]
    for rx in anchors:
        for n in soup.find_all(string=rx):
            blk = _nearest_label_block(n)
            if not blk:
                continue
            t = _text(blk)
            if re.search(r"prediction|forecast|20\d{2}", t, re.I):
                continue
            price = parse_price(t)
            if price is not None:
                dlog("[price] current from %s -> %s", rx.pattern, price)
                return price
    # fallback: header „BTC: $ …“
    hdr = soup.find(string=re.compile(r":\s*\$\s*", re.I))
    if hdr:
        t = _text(hdr.parent)
        price = parse_price(t)
        if price is not None:
            dlog("[price] current from header -> %s", price)
            return price
    return None

def price_after_label(soup: BeautifulSoup, label_rx: str) -> Optional[Decimal]:
    n = soup.find(string=re.compile(label_rx, re.I))
    if not n:
        return None
    blk = _nearest_label_block(n)
    if not blk:
        return None
    t = _text(blk)
    price = parse_price(t)
    return price

def extract_pair_from_block_text(txt: str) -> Tuple[Optional[Decimal], Optional[Decimal], bool]:
    m = PAIR_RX.search(txt)
    if not m:
        return None, None, False
    return _to_dec(m.group(1)), _to_dec(m.group(2)), True

def extract_horizon_pair(soup: BeautifulSoup, label_rx: str, pct_hint_rx: Optional[str] = None) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    """
    Najdi blok s daným labelem a vytáhni (price, pct).
      1) spárovaný vzor '$... ... %' v rámci bloku
      2) samostatně $ a % v rámci bloku
      3) pokud pct_hint_rx je zadán, najdi v samostatné větě s tímto hintem %
    """
    # 1) přímý label
    n = soup.find(string=re.compile(label_rx, re.I))
    if n:
        blk = _nearest_label_block(n)
        if blk:
            txt = _text(blk)
            price, pct, paired = extract_pair_from_block_text(txt)
            if paired:
                return price, pct, "pair"
            price2 = parse_price(txt)
            pct2 = parse_pct(txt)
            if price2 is not None or pct2 is not None:
                return price2, pct2, "block"
    # 2) scan všech boxů s labelem
    for box in soup.find_all(SAFE_LABEL_BLOCKS):
        t = _text(box)
        if not t or not re.search(label_rx, t, re.I):
            continue
        if "$" not in t and "%" not in t:
            continue
        price, pct, paired = extract_pair_from_block_text(t)
        if paired:
            return price, pct, "scan-pair"
        price2 = parse_price(t)
        pct2 = parse_pct(t)
        if price2 is not None or pct2 is not None:
            return price2, pct2, "scan-block"
    # 3) doplňkové procento podle hintu (např. věta „Over the next five days …“)
    if pct_hint_rx:
        node = soup.find(string=re.compile(pct_hint_rx, re.I))
        if node:
            t = _text(_nearest_label_block(node) or node.parent)
            pct3 = parse_pct(t)
            if pct3 is not None:
                return None, pct3, "hint-pct"
    return None, None, "miss"

def extract_5d_price_and_pct(soup: BeautifulSoup) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    # label i běžná varianta bez pomlčky
    return extract_horizon_pair(
        soup,
        r"\b5\s*-\s*Day\s*Prediction\b|\b5\s*Day\s*Prediction\b",
        pct_hint_rx=r"Over the next five days",
    )

def extract_1m_price_and_pct(soup: BeautifulSoup) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    # 1M: často je % v řádku „Price Prediction  $ …  (x%)“
    price = price_after_label(soup, r"\b1\s*-\s*Month\s*Prediction\b|\b1\s*Month\s*Prediction\b")
    pct = None
    node = soup.find(string=re.compile(r"\bPrice\s+Prediction\b", re.I))
    if node:
        t = _text(node.find_parent(["div","section","p"]) or node.parent)
        pct = parse_pct(t)
        dbg = "label+priceprediction"
    else:
        # fallback na generické hledání v bloku
        price2, pct2, dbg2 = extract_horizon_pair(soup, r"\b1\s*-\s*Month\s*Prediction\b|\b1\s*Month\s*Prediction\b")
        price = price or price2
        pct = pct or pct2
        dbg = dbg2
    return price, pct, "1m-" + ("pp" if node else dbg)

def extract_3m_price_and_pct(soup: BeautifulSoup) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    return extract_horizon_pair(
        soup,
        r"\b3\s*-\s*Month\s*Prediction\b|\b3\s*Month\s*Prediction\b"
    )

def extract_6m_price_and_pct(soup: BeautifulSoup) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    return extract_horizon_pair(
        soup,
        r"\b6\s*-\s*Month\s*Prediction\b|\b6\s*Month\s*Prediction\b"
    )

def extract_1y_price_and_pct(soup: BeautifulSoup) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    return extract_horizon_pair(
        soup,
        r"\b1\s*-\s*Year\s*Prediction\b|\b1\s*Year\s*Prediction\b"
    )

def parse_prediction_detail(html: str) -> Dict[str, Optional[Decimal]]:
    """
    Cílené parsování: current price + 5D/1M/3M/6M/1Y.
    Pokud procento pro horizont chybí, dopočítá se z current_price.
    """
    soup = BeautifulSoup(html, "html.parser")

    current_price = extract_current_price(soup)

    p5, c5, d5 = extract_5d_price_and_pct(soup)
    dlog("[parse] 5D dbg=%s price=%s pct=%s", d5, p5, c5)

    p1m, c1m, d1m = extract_1m_price_and_pct(soup)
    dlog("[parse] 1M dbg=%s price=%s pct=%s", d1m, p1m, c1m)

    p3m, c3m, d3m = extract_3m_price_and_pct(soup)
    dlog("[parse] 3M dbg=%s price=%s pct=%s", d3m, p3m, c3m)

    p6m, c6m, d6m = extract_6m_price_and_pct(soup)
    dlog("[parse] 6M dbg=%s price=%s pct=%s", d6m, p6m, c6m)

    p1y, c1y, d1y = extract_1y_price_and_pct(soup)
    dlog("[parse] 1Y dbg=%s price=%s pct=%s", d1y, p1y, c1y)

    # dopočty chybějících procent
    c5  = c5  if c5  is not None else pct_from_prices(p5,  current_price)
    c1m = c1m if c1m is not None else pct_from_prices(p1m, current_price)
    c3m = c3m if c3m is not None else pct_from_prices(p3m, current_price)
    c6m = c6m if c6m is not None else pct_from_prices(p6m, current_price)
    c1y = c1y if c1y is not None else pct_from_prices(p1y, current_price)

    return {
        "current_price": current_price,
        "pred_5d": p5,  "chg_5d": c5,
        "pred_1m": p1m, "chg_1m": c1m,
        "pred_3m": p3m, "chg_3m": c3m,
        "pred_6m": p6m, "chg_6m": c6m,
        "pred_1y": p1y, "chg_1y": c1y,
    }

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
        if not has_any and parsed.get("current_price") is None:
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

        # --- STATIC režim ---
        if SYMBOL_SOURCE != "STATIC":
            dlog("[warn] SYMBOL_SOURCE=%s != STATIC -> pokračuji jako STATIC.", SYMBOL_SOURCE)

        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return

        items = collect_predictions_static(coinlist)

        # dedup dle symbolu
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
