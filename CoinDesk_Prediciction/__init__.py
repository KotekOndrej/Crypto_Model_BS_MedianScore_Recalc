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
VERSION = "10.0-slug+ranges+table+paragraph"

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
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/10.0)"),
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

SAFE_BLOCKS = ("div","section","article","li","tr","td","p")

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Utility: text/čísla =====================
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

def to_decimal(num: Optional[str | float | Decimal]) -> Optional[Decimal]:
    if num is None:
        return None
    if isinstance(num, Decimal):
        return num
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

def txt(el) -> str:
    return normalize_text(el.get_text(" ", strip=True)) if el is not None else ""

# ===================== HTTP =====================
def fetch_prediction_detail(session: requests.Session, slug: str) -> Optional[str]:
    url = DETAIL_TMPL.format(slug=slug)
    try:
        resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        # zkus explicitně UTF-8
        resp.encoding = "utf-8"
        html = resp.text
        dlog("[detail] slug=%s status=%s len=%s", slug, resp.status_code, len(html))
        if resp.status_code == 200:
            return html
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

# ===================== Parsování: Current Price + tabulkový predicted =====================
def parse_table_prices(soup: BeautifulSoup) -> Dict[str, Optional[Decimal]]:
    """
    Z tabulky table.table-grid.prediction-data-table vytáhne:
      - current_price (tr.data-current-price > td)
      - table_predicted_price (tr.data-predicted-price > td) – často odpovídá 1M
    """
    out = {"current_price": None, "table_predicted_price": None}
    tbl = soup.select_one("table.table-grid.prediction-data-table")
    if not tbl:
        return out

    curr_td = tbl.select_one("tr.data-current-price > td")
    pred_td = tbl.select_one("tr.data-predicted-price > td")

    if curr_td:
        out["current_price"] = clean_money_from_text(txt(curr_td))
    if pred_td:
        out["table_predicted_price"] = clean_money_from_text(txt(pred_td))

    return out

# ===================== Parsování: prediction ranges (5D/1M/3M) =====================
# Labely v UI: "5-Day Prediction", "1-Month Prediction", "3-Month Prediction"
def parse_prediction_ranges(soup: BeautifulSoup) -> Dict[str, Optional[Decimal]]:
    """
    Hledá bloky: div.prediction-ranges .prediction-range
    Každý range má label (5-Day / 1-Month / 3-Month) a vedle toho $ cena.
    """
    out = {"5D": None, "1M": None, "3M": None}
    ranges = soup.select("div.prediction-ranges .prediction-range")
    if not ranges:
        return out

    for rg in ranges:
        t = txt(rg)
        # Pokus vytáhnout cenu zevnitř, když je struktura hlubší:
        # preferuj poslední <div> s $ … jinak padni na celý text.
        price_text = ""
        price_divs = [d for d in rg.find_all("div") if d and "$" in d.get_text()]
        if price_divs:
            price_text = txt(price_divs[-1])
        else:
            price_text = t

        p = clean_money_from_text(price_text)

        lt = t.lower()
        if lt.startswith("5-day") or "5-day prediction" in lt:
            out["5D"] = p or out["5D"]
        elif lt.startswith("1-month") or "1-month prediction" in lt or "1 month prediction" in lt:
            out["1M"] = p or out["1M"]
        elif lt.startswith("3-month") or "3-month prediction" in lt or "3 month prediction" in lt:
            out["3M"] = p or out["3M"]

    return out

# ===================== Parsování: 5D fallback z odstavce =====================
# "Over the next five days, Bitcoin will reach the highest price of $ 122,373 ... which would represent 11.87% ..."
_RX_5D_SENTENCE = re.compile(
    r"over the next five days.*?\$[\s\u202F]*([\d,]+).*?represent\s*([+\-]?\d+(?:\.\d+)?)\s*%",
    re.I | re.S
)

def parse_5d_from_paragraph(soup: BeautifulSoup) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    body_text = txt(soup)
    m = _RX_5D_SENTENCE.search(body_text)
    if not m:
        return None, None
    price_s, pct_s = m.group(1), m.group(2)
    price = to_decimal(price_s.replace(",", "")) if price_s else None
    pct = to_decimal(pct_s) if pct_s else None
    return price, pct

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

        # soup (HTML parser)
        soup = BeautifulSoup(html, "html.parser")

        # 1) tabulka: current + table predicted
        table_vals = parse_table_prices(soup)
        current_price = table_vals.get("current_price")
        table_pred = table_vals.get("table_predicted_price")

        # 2) prediction ranges: 5D/1M/3M
        ranges = parse_prediction_ranges(soup)
        pred_5d = ranges.get("5D")
        pred_1m = ranges.get("1M")
        pred_3m = ranges.get("3M")

        # 3) fallback pro 5D z odstavce (pokud chybí v ranges)
        chg_5d = None
        if pred_5d is None:
            p5, pct5 = parse_5d_from_paragraph(soup)
            if p5 is not None:
                pred_5d = p5
                chg_5d = pct5

        item = {
            "symbol": symbol,
            "slug": slug,
            "token_name": token_name,
            "page_url": url,
            "html_len": len(html),
            "html_blob": blob_path,
            "current_price": current_price,
            "pred_5d": pred_5d,
            "pred_1m": pred_1m if pred_1m is not None else table_pred,  # fallback 1M z tabulky
            "pred_3m": pred_3m,
            "chg_5d_hint": chg_5d,  # může být None, dopočítáme níže
        }
        out.append(item)

        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)

    dlog("[detail] collected_items=%s (from slugs=%s)", len(out), len(coinlist))
    return out

# ===================== Build rows (jen 5D/1M/3M) =====================
def build_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
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
        # páry (horizon -> months)
        horizons = [
            ("5D", lambda d: d + relativedelta(days=5), it.get("pred_5d"), it.get("chg_5d_hint")),
            ("1M", lambda d: d + relativedelta(months=1), it.get("pred_1m"), None),
            ("3M", lambda d: d + relativedelta(months=3), it.get("pred_3m"), None),
        ]

        for short, fn_to, pred, chg in horizons:
            pred_dec = to_decimal(pred)
            if pred_dec is None:
                continue
            pct = to_decimal(chg) if chg is not None else pct_from_prices(pred_dec, curr)
            model_to = fn_to(scrape_date).isoformat()
            rows.append(mkrow(sym, slug, name, curr, short, model_to, pred_dec, pct, url, hlen, hkey))

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

        # 3) sestav řádky (5D,1M,3M když jsou k dispozici)
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
