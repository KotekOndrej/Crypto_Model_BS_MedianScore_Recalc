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
from urllib.parse import urlparse, parse_qs, urlencode

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace =====================
VERSION = "6.0-static-index"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Statický seznam tokenů (v blobu)
SYMBOL_SOURCE = os.getenv("SYMBOL_SOURCE", "STATIC").upper()
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")   # columns: symbol,slug,token_name

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
PAGE_SLEEP_MS = int(os.getenv("PAGE_SLEEP_MS", "250"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/6.0)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Index s tabulkou predikcí
PREDICTIONS_INDEX = "https://coincodex.com/predictions/"
MAX_INDEX_PAGES = int(os.getenv("MAX_INDEX_PAGES", "15"))  # kolik stran indexu projít max.

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

def to_dec(num_str: Optional[str]) -> Optional[Decimal]:
    try:
        if num_str is None:
            return None
        return Decimal(num_str.replace(",", ""))
    except Exception:
        return None

def parse_price_and_pct(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z jedné buňky vytáhne $cenu a procenta; musí tam být $ nebo % (jinak vrací None)."""
    if not text:
        return None, None
    price = None; pct = None
    m_price = PRICE_RX.search(text)
    if m_price:
        price = to_dec(m_price.group(1))
    m_pct = PCT_RX.search(text)
    if m_pct:
        pct = to_dec(m_pct.group(1))
    return price, pct

def build_url_with_page(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    parsed = urlparse(base_url)
    q = parse_qs(parsed.query)
    q["page"] = [str(page)]
    return parsed._replace(query=urlencode(q, doseq=True)).geturl()

# ===================== Načtení CoinList.csv =====================
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
            name = (row.get("token_name") or "").strip()
            if not sym:
                continue
            out.append({"symbol": sym, "token_name": name})
    dlog("[coinlist] loaded symbols=%s", len(out))
    return out

# ===================== Parsování tabulky z /predictions/ =====================
def extract_rows_from_predictions(html: str) -> List[Dict]:
    """
    Najde hlavní tabulku s '5D Prediction', '1M Prediction' ... a vrátí řádky:
      { symbol, token_name?, current_price, pred_5d/chg_5d, ..., pred_1y/chg_1y }
    """
    soup = BeautifulSoup(html, "html.parser")

    # najdi tabulku dle hlaviček
    table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if any(("5D" in h and "Prediction" in h) for h in headers):
            table = t
            break
    if not table:
        return []

    header_texts = [th.get_text(strip=True) for th in table.find_all("th")]
    idx = {name: i for i, name in enumerate(header_texts)}

    # názvy sloupců (Price může být např. 'Price' nebo 'Current Price')
    price_col = None
    for cand in ["Price", "Current Price"]:
        if cand in idx:
            price_col = idx[cand]
            break
    # fallback: první sloupec, kde se ve vzorku buněk vyskytuje $
    if price_col is None:
        for j, name in enumerate(header_texts):
            # přeskoč jasné predikční sloupce
            if "Prediction" in name:
                continue
            price_col = j
            break

    required = {
        "Name": idx.get("Name"),
        "5D": idx.get("5D Prediction"),
        "1M": idx.get("1M Prediction"),
        "3M": idx.get("3M Prediction"),
        "6M": idx.get("6M Prediction"),
        "1Y": idx.get("1Y Prediction"),
    }
    if any(v is None for v in required.values()):
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    out: List[Dict] = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < len(header_texts):
            continue

        # Name -> symbol + (volitelně) token_name
        name_cell = tds[required["Name"]]
        name_text = name_cell.get_text(" ", strip=True)
        symbol, token_name = None, None
        if name_text:
            parts = name_text.split()
            if len(parts) >= 2:
                symbol, token_name = parts[0].upper(), " ".join(parts[1:])
            else:
                token_name = name_text

        # current price
        cp = None
        if price_col is not None and price_col < len(tds):
            cp_text = tds[price_col].get_text(" ", strip=True)
            cp, _ = parse_price_and_pct(cp_text)

        # Predictions
        preds: Dict[str, Optional[Decimal]] = {}
        chgs: Dict[str, Optional[Decimal]] = {}
        for short, col_idx in [("5D", required["5D"]), ("1M", required["1M"]), ("3M", required["3M"]),
                               ("6M", required["6M"]), ("1Y", required["1Y"])]:
            cell_text = tds[col_idx].get_text(" ", strip=True)
            price, pct = parse_price_and_pct(cell_text)
            preds[short] = price
            chgs[short] = pct

        if symbol:
            out.append({
                "symbol": symbol,
                "token_name": token_name or "",
                "current_price": cp,
                "pred_5d": preds["5D"], "chg_5d": chgs["5D"],
                "pred_1m": preds["1M"], "chg_1m": chgs["1M"],
                "pred_3m": preds["3M"], "chg_3m": chgs["3M"],
                "pred_6m": preds["6M"], "chg_6m": chgs["6M"],
                "pred_1y": preds["1Y"], "chg_1y": chgs["1Y"],
            })
    return out

def collect_from_index_for_targets(target_symbols: List[str]) -> Dict[str, Dict]:
    """
    Projde 1..MAX_INDEX_PAGES stránek indexu a vrátí dict {symbol -> row} jen pro požadované symboly.
    Jakmile najdeme všechny, přestaneme.
    """
    target_set = {s.upper() for s in target_symbols}
    found: Dict[str, Dict] = {}
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, MAX_INDEX_PAGES + 1):
        url = build_url_with_page(PREDICTIONS_INDEX, page)
        try:
            resp = session.get(url, timeout=HTTP_TIMEOUT)
            html = resp.text
            dlog("[index] page=%s status=%s len=%s", page, resp.status_code, len(html))
            if resp.status_code != 200:
                break
        except Exception as e:
            dlog("[index] error page=%s: %s", page, e)
            break

        rows = extract_rows_from_predictions(html)
        # filtruj jen na target symboly
        added = 0
        for r in rows:
            sym = r.get("symbol", "").upper()
            if sym in target_set and sym not in found:
                found[sym] = r
                added += 1
        dlog("[index] page=%s matched=%s total_found=%s", page, added, len(found))

        if len(found) >= len(target_set):
            dlog("[index] all targets found -> stop")
            break

        if PAGE_SLEEP_MS > 0:
            time.sleep(PAGE_SLEEP_MS / 1000.0)

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

# ===================== Build rows =====================
def build_active_rows(scrape_date: dt.date, load_ts: str, items: List[Dict], token_name_lookup: Dict[str, str]) -> List[Dict]:
    rows: List[Dict] = []
    for it in items:
        symbol = it["symbol"].upper()
        token_name = token_name_lookup.get(symbol, it.get("token_name", "")) or ""
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
            if price is None and pct is None:
                continue
            # When price is present but pct missing, dopočítej
            if pct is None and (price is not None) and (current_price is not None) and current_price != 0:
                try:
                    pct = (Decimal(price) - Decimal(current_price)) * Decimal(100) / Decimal(current_price)
                except Exception:
                    pass
            if price is None:
                # bez ceny nemá smysl záznam
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

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s SYMBOL_SOURCE=%s OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s COINLIST_BLOB=%s MAX_INDEX_PAGES=%s",
         VERSION, SYMBOL_SOURCE, OUTPUT_CONTAINER, AZURE_BLOB_NAME, COINLIST_BLOB, MAX_INDEX_PAGES)

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

        if SYMBOL_SOURCE != "STATIC":
            dlog("[warn] SYMBOL_SOURCE=%s != STATIC -> pokračuji jako STATIC.", SYMBOL_SOURCE)

        # --- 1) načti seznam cílových symbolů
        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return
        target_symbols = [c["symbol"] for c in coinlist]
        token_name_lookup = {c["symbol"].upper(): c.get("token_name","") for c in coinlist}

        # --- 2) posbírej data z index tabulky jen pro cílové symboly
        found = collect_from_index_for_targets(target_symbols)
        if not found:
            dlog("[index] nothing matched -> stop")
            return

        # --- 3) připrav nové aktivní řádky
        items = list(found.values())
        # dedup pro jistotu
        uniq: Dict[str, Dict] = {}
        for it in items:
            s = it.get("symbol","").upper()
            if s:
                uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_items=%s", len(items))

        new_rows = build_active_rows(scrape_date, load_ts, items, token_name_lookup)

        # --- 4) CSV overwrite workflow (deaktivace dnešních a připsání nových)
        existing = load_csv_rows(cc, AZURE_BLOB_NAME)
        deact = deactivate_todays_rows(existing, scrape_date.isoformat())
        all_rows = existing + new_rows
        dlog("[csv] deactivated_today=%s newly_active=%s final_rows=%s", deact, len(new_rows), len(all_rows))
        write_csv_rows(cc, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception: %s", e)
        logging.error(traceback.format_exc())
        return
