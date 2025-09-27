import os
import io
import re
import csv
import time
import logging
import traceback
import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace =====================
VERSION = "8.1-slug-detail-5D-1M-3M-stable"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Coin list (musí mít sloupce: symbol, slug, token_name)
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/8.1)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

# Horizonty, které ukládáme
HORIZON_MAP = {
    "5D": ("5D Prediction", lambda d: d + relativedelta(days=5)),
    "1M": ("1M Prediction", lambda d: d + relativedelta(months=1)),
    "3M": ("3M Prediction", lambda d: d + relativedelta(months=3)),
}

# CSV schema (výstup)
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]

SAFE_LABEL_BLOCKS = ("tr", "li", "div", "section", "article", "p")

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Regexy a utility =====================
_RX_PRICE = re.compile(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?|[0-9]+(?:\.\d+)?)")
_RX_PCT   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")

def _to_dec(s: Optional[str]) -> Optional[Decimal]:
    if s is None:
        return None
    try:
        return Decimal(s.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None

def parse_price_and_pct(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z textu vytáhne (price, pct). Pokud něco chybí, vrací None."""
    if not text:
        return None, None
    mp = _RX_PRICE.search(text)
    mc = _RX_PCT.search(text)
    price = _to_dec(mp.group(1)) if mp else None
    pct   = _to_dec(mc.group(1)) if mc else None
    return price, pct

def _text(el) -> str:
    return el.get_text(" ", strip=True) if el is not None else ""

def _nearest_label_block(node):
    """
    Najde nejbližší 'řádek/box' pro daný label, který obsahuje $ nebo % (abychom netahali z čistých nadpisů).
    Preferujeme <tr>, pak běžné "box" elementy.
    """
    if not node:
        return None
    # preferuj <tr>
    for anc in getattr(node, "parents", []):
        if getattr(anc, "name", "").lower() == "tr":
            t = _text(anc)
            if "$" in t or "%" in t:
                return anc
    # další boxy
    for anc in getattr(node, "parents", []):
        if getattr(anc, "name", "").lower() in SAFE_LABEL_BLOCKS:
            t = _text(anc)
            if "$" in t or "%" in t:
                return anc
    return node.parent if hasattr(node, "parent") else None

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

# ===================== Parser detail stránky =====================
def extract_current_price(soup: BeautifulSoup) -> Optional[Decimal]:
    """
    Hledej jen v blocích s 'Current price / Live price / price is'.
    Ignoruj texty s 'prediction/forecast/20xx', aby se nepletly jiné tabulky.
    """
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
            price, _ = parse_price_and_pct(t)
            if price is not None:
                dlog("[price] current via %s -> %s", rx.pattern, price)
                return price
    # fallback: řádky typu „BTC: $ …“
    hdr = soup.find(string=re.compile(r":\s*\$\s*", re.I))
    if hdr:
        t = _text(hdr.parent or hdr)
        price, _ = parse_price_and_pct(t)
        if price is not None:
            dlog("[price] current via header -> %s", price)
            return price
    return None

def find_pair_for_horizon(soup: BeautifulSoup, label_regex: str) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
    """
    Najde blok s daným labelem a z jeho textu vytáhne (price, pct).
    Nikdy nebere hodnoty mimo tento blok.
    """
    # 1) přímý label
    node = soup.find(string=re.compile(label_regex, re.I))
    if node:
        blk = _nearest_label_block(node)
        if blk:
            txt = _text(blk)
            price, pct = parse_price_and_pct(txt)
            if price is not None or pct is not None:
                return price, pct, "label-block"
    # 2) scan boxů (fallback)
    for box in soup.find_all(SAFE_LABEL_BLOCKS):
        t = _text(box)
        if not t or not re.search(label_regex, t, re.I):
            continue
        if "$" not in t and "%" not in t:
            continue
        price, pct = parse_price_and_pct(t)
        if price is not None or pct is not None:
            return price, pct, "scan-block"
    return None, None, "miss"

def parse_prediction_detail(html: str) -> Dict[str, Optional[Decimal]]:
    """
    Z detailu vytáhne: current_price + 5D/1M/3M.
    Pokud procento chybí a máme current_price, dopočítá se.
    """
    soup = BeautifulSoup(html, "html.parser")

    current_price = extract_current_price(soup)

    # tolerantní regexy na labely
    rx_5d = r"\b5\s*-\s*day\b|\b5\s*day\b|\b5d\b"
    rx_1m = r"\b1\s*-\s*month\b|\b1\s*month\b|\b1m\b"
    rx_3m = r"\b3\s*-\s*month\b|\b3\s*month\b|\b3m\b"

    p5,  c5,  d5  = find_pair_for_horizon(soup, rx_5d)
    p1m, c1m, d1m = find_pair_for_horizon(soup, rx_1m)
    p3m, c3m, d3m = find_pair_for_horizon(soup, rx_3m)

    dlog("[parse] 5D dbg=%s price=%s pct=%s", d5,  p5,  c5)
    dlog("[parse] 1M dbg=%s price=%s pct=%s", d1m, p1m, c1m)
    dlog("[parse] 3M dbg=%s price=%s pct=%s", d3m, p3m, c3m)

    # dopočet % pokud chybí
    def pct_from_prices(pred: Optional[Decimal], base: Optional[Decimal]) -> Optional[Decimal]:
        try:
            if pred is None or base is None or base == 0:
                return None
            return (pred - base) * Decimal(100) / base
        except Exception:
            return None

    c5  = c5  if c5  is not None else pct_from_prices(p5,  current_price)
    c1m = c1m if c1m is not None else pct_from_prices(p1m, current_price)
    c3m = c3m if c3m is not None else pct_from_prices(p3m, current_price)

    return {
        "current_price": current_price,
        "pred_5d": p5,  "chg_5d": c5,
        "pred_1m": p1m, "chg_1m": c1m,
        "pred_3m": p3m, "chg_3m": c3m,
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

def deactivate_todays_rows(existing: List[Dict], today_iso: str) -> int:
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"
            changed += 1
    return changed

# ===================== CoinList (se slugy) =====================
def load_coinlist_from_blob(container_client, blob_name: str) -> List[Dict]:
    """Načte CoinList.csv (sloupce: symbol, slug, token_name) z OUTPUT_CONTAINER."""
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
            sym  = (row.get("symbol") or "").strip().upper()
            slug = (row.get("slug") or "").strip().lower()
            name = (row.get("token_name") or "").strip()
            if not sym or not slug:
                continue
            out.append({"symbol": sym, "slug": slug, "token_name": name})
    dlog("[coinlist] loaded items=%s", len(out))
    return out

# ===================== Sběr dat přes slug detail =====================
def collect_predictions_by_slug(coinlist: List[Dict]) -> List[Dict]:
    session = requests.Session()
    session.headers.update(HEADERS)

    out: List[Dict] = []
    for coin in coinlist:
        slug = coin["slug"]
        symbol = coin["symbol"]
        token_name = coin.get("token_name", "")

        html = fetch_prediction_detail(session, slug)
        if not html:
            continue

        parsed = parse_prediction_detail(html)
        has_any = any(parsed.get(k) is not None for k in ["pred_5d", "pred_1m", "pred_3m"])
        if not has_any and parsed.get("current_price") is None:
            continue

        out.append({
            "symbol": symbol,
            "token_name": token_name,
            **parsed
        })

        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)

    dlog("[detail] collected_items=%s (from slugs=%s)", len(out), len(coinlist))
    return out

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

        # 2) stáhni a naparsuj detail stránky pro každý slug
        items = collect_predictions_by_slug(coinlist)

        # 3) dedup dle symbolu (poslední výhra)
        uniq: Dict[str, Dict] = {}
        for it in items:
            s = it.get("symbol")
            if s:
                uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_items=%s", len(items))

        # 4) CSV overwrite workflow
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
