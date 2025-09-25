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
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# -------------------- Konfigurace --------------------
BASE_URL = "https://coincodex.com/predictions/"
DETAIL_URL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/3.4)")
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))          # tvrdý limit pro jistotu
PAGE_SLEEP_MS = int(os.getenv("PAGE_SLEEP_MS", "250")) # pauza mezi stránkami
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "100"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
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

# -------------------- Helpery --------------------
def dlog(msg, *args): logging.info(msg, *args)
def hash_text(text: str) -> str: return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
def parse_decimal_safe(s: str) -> Optional[Decimal]:
    try: return Decimal(s)
    except: return None

def get_current_page_from_url(url: str) -> int:
    try:
        qs = parse_qs(urlparse(url).query)
        return int(qs.get("page", ["1"])[0]) or 1
    except Exception:
        return 1

def build_url_with_page(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    parsed = urlparse(base_url)
    q = parse_qs(parsed.query)
    q["page"] = [str(page)]
    new_query = urlencode(q, doseq=True)
    return parsed._replace(query=new_query).geturl() or f"{base_url}?page={page}"

# -------------------- Parser tabulky /predictions/ --------------------
def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    if not text: return None, None
    m_price = re.search(r"[-]?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    price_dec = parse_decimal_safe(m_price.group(1).replace(",", "")) if m_price else None
    m_pct = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", text)
    pct_dec = parse_decimal_safe(m_pct.group(1)) if m_pct else None
    return price_dec, pct_dec

def extract_table_rows(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if any(("5D" in h and "Prediction" in h) for h in headers):
            table = t; break
    if table is None: return []

    header_texts = [th.get_text(strip=True) for th in table.find_all("th")]
    col_idx = {name: i for i, name in enumerate(header_texts)}

    price_col_name = "Price" if "Price" in col_idx else next((h for h in header_texts if "Price" in h and h not in HORIZON_MAP), None)
    required = ["Name", "5D Prediction", "1M Prediction", "3M Prediction", "6M Prediction", "1Y Prediction"]
    if not all(r in col_idx for r in required): return []
    tbody = table.find("tbody")
    if not tbody: return []

    rows: List[Dict] = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < len(header_texts): continue

        name_cell = tds[col_idx["Name"]]
        name_text = name_cell.get_text(" ", strip=True)
        symbol, token_name, slug = None, None, None
        a = name_cell.find("a")
        if a and a.get("href"):
            m = re.search(r"/crypto/([^/]+)/?", a.get("href"))
            if m: slug = m.group(1).strip()
        if name_text:
            parts = name_text.split()
            if len(parts) >= 2:
                symbol, token_name = parts[0], " ".join(parts[1:])
            else:
                token_name = name_text
        if not symbol and a and a.get_text(strip=True):
            atext = a.get_text(" ", strip=True)
            parts = atext.split()
            if len(parts) >= 2:
                symbol, token_name = parts[0], " ".join(parts[1:])
            else:
                token_name = atext

        current_price = None
        if price_col_name:
            try:
                price_cell_text = tds[col_idx[price_col_name]].get_text(" ", strip=True)
                current_price, _ = parse_price_and_change(price_cell_text)
            except: current_price = None

        preds, chgs = {}, {}
        for header in required[1:]:
            cell_text = tds[col_idx[header]].get_text(" ", strip=True)
            price, pct = parse_price_and_change(cell_text)
            preds[header], chgs[header] = price, pct

        if symbol and any(preds.values()):
            rows.append({
                "symbol": symbol,
                "token_name": token_name or "",
                "slug": slug,
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

# -------------------- Detekce dostupných čísel stránek --------------------
_PAGE_HREF_RX = re.compile(r"/predictions/\?page=(\d+)", re.I)

def find_available_pages(html: str, current_url: str) -> List[int]:
    """Z HTML vyextrahuje všechna čísla stránek z href, např. ?page=1..10."""
    soup = BeautifulSoup(html, "html.parser")
    pages = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = _PAGE_HREF_RX.search(href)
        if m:
            try:
                pages.add(int(m.group(1)))
            except:
                pass
    # fallback: zkus i textové odkazy s čísly (kdyby href neobsahoval ?page)
    for a in soup.find_all("a"):
        txt = (a.get_text() or "").strip()
        if txt.isdigit():
            try:
                pages.add(int(txt))
            except:
                pass
    pages_list = sorted(pages)
    dlog("[pager] detected_pages=%s", pages_list)
    return pages_list

def compute_next_url(html: str, current_url: str) -> Optional[str]:
    """Najde další URL podle detekovaných ?page=N odkazů."""
    pages = find_available_pages(html, current_url)
    if not pages:
        return None
    curr = get_current_page_from_url(current_url)
    # preferuj nejbližší vyšší číslo
    higher = [p for p in pages if p > curr]
    if higher:
        return build_url_with_page(BASE_URL, min(higher))
    # když nejsou vyšší, ale vidíme max (např. 10), a jsme < max, posuň se o 1
    max_p = max(pages)
    if curr < max_p:
        return build_url_with_page(BASE_URL, curr + 1)
    return None

# -------------------- Detail coinu /price-prediction/ (enrichment) --------------------
def enrich_from_coin_detail(session: requests.Session, item: Dict) -> Dict:
    slug = item.get("slug")
    if not slug: return item
    url = DETAIL_URL_TMPL.format(slug=slug)
    try:
        resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        dlog("[detail] slug=%s status=%s len=%s", slug, resp.status_code, len(resp.text))
        if resp.status_code != 200: return item
        soup = BeautifulSoup(resp.text, "html.parser")

        def find_row(label_regex: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
            row = soup.find(string=re.compile(label_regex, re.I))
            if not row: return None, None
            section = row.parent.get_text(" ", strip=True) if hasattr(row, "parent") else str(row)
            return parse_price_and_change(section)

        mapping = [
            ("5D", "5[-\\s]?Day"),
            ("1M", "1[-\\s]?Month"),
            ("3M", "3[-\\s]?Month"),
            ("6M", "6[-\\s]?Month"),
            ("1Y", "1[-\\s]?Year"),
        ]
        label_to_key = {
            "5D": ("pred_5d", "chg_5d"),
            "1M": ("pred_1m", "chg_1m"),
            "3M": ("pred_3m", "chg_3m"),
            "6M": ("pred_6m", "chg_6m"),
            "1Y": ("pred_1y", "chg_1y"),
        }
        for short, rx in mapping:
            k_price, k_chg = label_to_key[short]
            if item.get(k_price) is None or item.get(k_chg) is None:
                price, pct = find_row(rx)
                if price is not None and item.get(k_price) is None:
                    item[k_price] = price
                if pct is not None and item.get(k_chg) is None:
                    item[k_chg] = pct
        return item
    except Exception:
        return item

# -------------------- Stahování stránek s robustním stránkováním --------------------
def crawl_all_items() -> List[Dict]:
    session = requests.Session()
    session.headers.update(HEADERS)

    all_items: List[Dict] = []
    seen_symbols = set()
    visited_hashes = set()

    url = BASE_URL  # začínáme na stránce 1
    for page_idx in range(1, MAX_PAGES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
            dlog("[fetch] url=%s status=%s len=%s", url, resp.status_code, len(resp.text))
            if resp.status_code != 200:
                dlog("[pager] non-200 -> stop")
                break
            html = resp.text
        except Exception as e:
            dlog("[pager] error fetching %s: %s", url, e)
            break

        h = hash_text(html)
        if h in visited_hashes:
            dlog("[pager] duplicate html hash -> stop")
            break
        visited_hashes.add(h)

        rows = extract_table_rows(html)
        if not rows:
            dlog("[pager] no rows -> stop")
            break

        added = 0
        for it in rows:
            sym = it.get("symbol")
            if not sym or sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            all_items.append(it)
            added += 1
        dlog("[pager] page_idx=%s newly_added=%s total=%s", page_idx, added, len(all_items))

        # NOVÉ: dopočítej další URL z čísel stránek v HTML
        next_url = compute_next_url(html, url)
        if not next_url:
            dlog("[pager] no next link (by href scan) -> stop")
            break

        # sanity: když next_url == url, skonči
        if next_url == url:
            dlog("[pager] next_url equals current -> stop")
            break

        url = next_url

        if added == 0:
            dlog("[pager] nothing new added -> stop")
            break

        if PAGE_SLEEP_MS > 0:
            time.sleep(PAGE_SLEEP_MS / 1000.0)

    # Enrichment z detailů (volitelně doplní chybějící predikce)
    improved = 0
    for idx, it in enumerate(all_items):
        before = (it.get("pred_5d"), it.get("pred_1m"), it.get("pred_3m"), it.get("pred_6m"), it.get("pred_1y"))
        it2 = enrich_from_coin_detail(session, it)
        after = (it2.get("pred_5d"), it2.get("pred_1m"), it2.get("pred_3m"), it2.get("pred_6m"), it2.get("pred_1y"))
        if after != before: improved += 1
        all_items[idx] = it2
        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)
    dlog("[enrich] improved_items=%s of %s", improved, len(all_items))

    return all_items

# -------------------- CSV I/O --------------------
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
        for k in CSV_FIELDS: r.setdefault(k, "")
        writer.writerow(r)
    data = buf.getvalue().encode("utf-8")
    blob_client.upload_blob(data, overwrite=True)
    dlog("[csv-write] uploaded rows=%s size=%s", len(rows), len(data))

# -------------------- CSV tvorba dnešních řádků --------------------
def build_active_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
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
            if price is None: continue
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
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"; changed += 1
    return changed

# -------------------- Azure Function entrypoint --------------------
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[CoinDesk_Prediciction] Start %s", scrape_date.isoformat())
    dlog("[env] OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s", OUTPUT_CONTAINER, AZURE_BLOB_NAME)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting."); return

    try:
        # 1) Crawl: robustní stránkování přes skenování href ?page=N
        items = crawl_all_items()
        # Safety dedup podle symbolu
        uniq = {}
        for it in items:
            s = it.get("symbol")
            if s: uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_symbols=%s", len(items))

        # 2) Blob klient
        try:
            from azure.storage.blob import BlobServiceClient
        except Exception as e:
            logging.error("[blob] import error BlobServiceClient: %s", e)
            logging.error(traceback.format_exc()); return

        blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
        try:
            container_client.create_container(); dlog("[blob] container created: %s", OUTPUT_CONTAINER)
        except Exception:
            dlog("[blob] container exists: %s", OUTPUT_CONTAINER)

        # 3) Načti existující CSV, deaktivuj dnešní True a přidej nové aktivní
        existing_rows = load_csv_rows(container_client, AZURE_BLOB_NAME)
        deactivated = deactivate_todays_rows(existing_rows, scrape_date.isoformat())
        new_active_rows = build_active_rows(scrape_date, load_ts, items)
        all_rows = existing_rows + new_active_rows
        dlog("[csv] deactivated_today=%s newly_active=%s final_rows=%s", deactivated, len(new_active_rows), len(all_rows))

        # 4) Zapiš celý CSV (overwrite)
        write_csv_rows(container_client, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception in CoinDesk_Prediciction: %s", e)
        logging.error(traceback.format_exc()); return
