import datetime as dt
import logging
import os
import re
import traceback
from decimal import Decimal
from typing import Tuple, List, Dict, Optional
import io
import csv
import time
import hashlib
from urllib.parse import urlparse, parse_qs, urlencode, urljoin

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace a konstanty =====================
VERSION = "4.3-router"

# Zdroj scrapu: COIN_LIST | PREDICTIONS | HYBRID
SCRAPE_SOURCE = os.getenv("SCRAPE_SOURCE", "COIN_LIST").upper()

# CoinCodex endpoints
PREDICTIONS_INDEX = "https://coincodex.com/predictions/"
COIN_LIST_BASE   = "https://coincodex.com/crypto/"
DETAIL_TMPL      = "https://coincodex.com/crypto/{slug}/price-prediction/"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Limity a timeouts
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
PAGE_SLEEP_MS = int(os.getenv("PAGE_SLEEP_MS", "250"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))  # pro PREDICTIONS index
COIN_LIST_MAX_PAGES = int(os.getenv("COIN_LIST_MAX_PAGES", "10"))
MAX_COINS = int(os.getenv("MAX_COINS", "500"))

# Fallback manuální seed (čárkami oddělené slugs)
SEED_SLUGS = [s.strip() for s in os.getenv("SEED_SLUGS", "").split(",") if s.strip()]

# Debug uložení HTML (volitelně)
DEBUG_SAVE_HTML = os.getenv("DEBUG_SAVE_HTML", "0") == "1"

HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/4.3)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Horizonty
HORIZON_MAP = {
    "5D": ("5D Prediction", lambda d: d + relativedelta(days=5)),
    "1M": ("1M Prediction", lambda d: d + relativedelta(months=1)),
    "3M": ("3M Prediction", lambda d: d + relativedelta(months=3)),
    "6M": ("6M Prediction", lambda d: d + relativedelta(months=6)),
    "1Y": ("1Y Prediction", lambda d: d + relativedelta(years=1)),
}

# CSV schéma
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]

# ===================== Util a parsování =====================
def dlog(msg, *args): logging.info(msg, *args)

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def parse_decimal_safe(s: str) -> Optional[Decimal]:
    try:
        return Decimal(s)
    except Exception:
        return None

def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    if not text:
        return None, None
    m_price = re.search(r"[-]?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    price_dec = parse_decimal_safe(m_price.group(1).replace(",", "")) if m_price else None
    m_pct = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", text)
    pct_dec = parse_decimal_safe(m_pct.group(1)) if m_pct else None
    return price_dec, pct_dec

def build_url_with_page(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    parsed = urlparse(base_url)
    q = parse_qs(parsed.query)
    q["page"] = [str(page)]
    new_query = urlencode(q, doseq=True)
    return parsed._replace(query=new_query).geturl() or f"{base_url}?page={page}"

# ===================== PREDICTIONS index (primárně 1. stránka, + pokusy o další) =====================
def extract_table_rows_from_predictions(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if any(("5D" in h and "Prediction" in h) for h in headers):
            table = t; break
    if not table:
        return []

    header_texts = [th.get_text(strip=True) for th in table.find_all("th")]
    col_idx = {name: i for i, name in enumerate(header_texts)}
    price_col_name = "Price" if "Price" in col_idx else next((h for h in header_texts if "Price" in h and h not in HORIZON_MAP), None)
    required = ["Name", "5D Prediction", "1M Prediction", "3M Prediction", "6M Prediction", "1Y Prediction"]
    if not all(r in col_idx for r in required):
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []

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
            except Exception:
                current_price = None

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
                "pred_5d": preds.get("5D Prediction"), "chg_5d": chgs.get("5D Prediction"),
                "pred_1m": preds.get("1M Prediction"), "chg_1m": chgs.get("1M Prediction"),
                "pred_3m": preds.get("3M Prediction"), "chg_3m": chgs.get("3M Prediction"),
                "pred_6m": preds.get("6M Prediction"), "chg_6m": chgs.get("6M Prediction"),
                "pred_1y": preds.get("1Y Prediction"), "chg_1y": chgs.get("1Y Prediction"),
            })
    return rows

_PAGE_RX = re.compile(r"/predictions/\?page=(\d+)", re.I)

def crawl_predictions_index() -> List[Dict]:
    session = requests.Session(); session.headers.update(HEADERS)
    all_items: List[Dict] = []; seen = set(); visited = set()
    url = PREDICTIONS_INDEX
    for i in range(1, MAX_PAGES + 1):
        try:
            resp = session.get(url, timeout=HTTP_TIMEOUT); html = resp.text
            dlog("[pred] url=%s status=%s len=%s", url, resp.status_code, len(html))
            if resp.status_code != 200: break
        except Exception as e:
            dlog("[pred] error %s: %s", url, e); break
        h = hash_text(html)
        if h in visited:
            dlog("[pred] duplicate hash -> stop"); break
        visited.add(h)
        rows = extract_table_rows_from_predictions(html)
        if not rows: dlog("[pred] no rows -> stop"); break
        added = 0
        for it in rows:
            s = it.get("symbol")
            if s and s not in seen:
                seen.add(s); all_items.append(it); added += 1
        dlog("[pred] page=%s newly_added=%s total=%s", i, added, len(all_items))
        # zkus najít další stránku v HTML
        soup = BeautifulSoup(html, "html.parser")
        next_href = None
        a_rel = soup.select_one('a[rel="next"]')
        if a_rel and a_rel.get("href"): next_href = a_rel["href"]
        if not next_href:
            # fallback: hledej nejbližší vyšší ?page=
            pages = set()
            for a in soup.find_all("a", href=True):
                m = _PAGE_RX.search(a["href"]); 
                if m:
                    try: pages.add(int(m.group(1)))
                    except: pass
            curr = int(parse_qs(urlparse(url).query).get("page", ["1"])[0])
            higher = sorted([p for p in pages if p > curr])
            if higher: next_href = f"/predictions/?page={higher[0]}"
        if not next_href:
            # forced ?page=2..N
            forced = build_url_with_page(PREDICTIONS_INDEX, i+1)
            test = session.get(forced, timeout=HTTP_TIMEOUT)
            dlog("[pred-forced] %s status=%s len=%s", forced, test.status_code, len(test.text))
            if test.status_code != 200: break
            th = hash_text(test.text)
            if th in visited: break
            if extract_table_rows_from_predictions(test.text):
                url = forced
            else:
                break
        else:
            url = urljoin(PREDICTIONS_INDEX, next_href)
        if PAGE_SLEEP_MS > 0:
            time.sleep(PAGE_SLEEP_MS / 1000.0)
    return all_items

# ===================== COIN_LIST: sebrat slugs z /crypto/?page=N a jet per-coin =====================
_SLUG_RX = re.compile(r"^/crypto/([a-z0-9-]+)/?$", re.I)
_SYMBOL_NAME_RX = re.compile(r"^\s*([A-Z0-9]{2,10})\s+(.+)$")
_TITLE_RX = re.compile(r"^\s*([A-Z0-9]{2,10})\s+(.+?)\s+Price Prediction", re.I)

def fetch_coin_list_page(session: requests.Session, page: int) -> Optional[str]:
    url = build_url_with_page(COIN_LIST_BASE, page)
    try:
        resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        dlog("[list] page=%s status=%s len=%s", page, resp.status_code, len(resp.text))
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        dlog("[list] error page=%s: %s", page, e)
    return None

def extract_slugs_generic(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    slugs = []; seen = set()
    for a in soup.find_all("a", href=True):
        m = _SLUG_RX.match(a["href"])
        if not m: continue
        slug = m.group(1).lower()
        if slug not in seen:
            seen.add(slug); slugs.append(slug)
    return slugs

def collect_coin_slugs(max_pages: int, max_coins: int) -> List[Dict]:
    session = requests.Session(); session.headers.update(HEADERS)
    out: List[Dict] = []; seen = set()
    # manuální seedy jako první
    if SEED_SLUGS:
        for s in SEED_SLUGS:
            if s not in seen:
                seen.add(s); out.append({"slug": s})
    for p in range(1, max_pages + 1):
        if len(out) >= max_coins: break
        html = fetch_coin_list_page(session, p)
        if not html: break
        if DEBUG_SAVE_HTML and p <= 2:
            try:
                from azure.storage.blob import BlobServiceClient
                bsc = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
                cc = bsc.get_container_client(OUTPUT_CONTAINER)
                cc.upload_blob(f"debug_crypto_p{p}.html", html.encode("utf-8"), overwrite=True)
                dlog("[debug] saved /crypto/?page=%s HTML", p)
            except Exception as e:
                dlog("[debug] save html failed: %s", e)
        slugs = extract_slugs_generic(html)
        added = 0
        for s in slugs:
            if s not in seen:
                seen.add(s); out.append({"slug": s}); added += 1
                if len(out) >= max_coins: break
        dlog("[list] page=%s added=%s total=%s", p, added, len(out))
        if added == 0: break
        if PAGE_SLEEP_MS > 0: time.sleep(PAGE_SLEEP_MS / 1000.0)
    return out[:max_coins]

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
    soup = BeautifulSoup(html, "html.parser")
    # symbol & name
    symbol, token_name = None, None
    h = soup.find(["h1","h2"])
    if h:
        t = h.get_text(" ", strip=True)
        m = _SYMBOL_NAME_RX.match(t)
        if m:
            symbol, token_name = m.group(1), m.group(2)
    if not symbol and soup.title:
        m = _TITLE_RX.match(soup.title.get_text(" ", strip=True))
        if m: symbol, token_name = m.group(1), m.group(2)
    if not symbol:
        badge = soup.find(class_=re.compile(r"(cc-symbol|ticker|symbol)", re.I))
        if badge:
            s = (badge.get_text() or "").strip().upper()
            if 2 <= len(s) <= 10: symbol = s
    if not symbol and soup.title:
        first = soup.title.get_text().split()[0].upper()
        if 2 <= len(first) <= 10 and first.isalnum(): symbol = first

    # current price (volitelné)
    current_price = None
    price_candidate = soup.find(string=re.compile(r"\bPrice\b", re.I))
    if price_candidate:
        section = price_candidate.parent.get_text(" ", strip=True) if hasattr(price_candidate, "parent") else str(price_candidate)
        cp, _ = parse_price_and_change(section)
        current_price = cp or current_price

    def find_row(rx: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        node = soup.find(string=re.compile(rx, re.I)) or soup.find("th", string=re.compile(rx, re.I))
        if not node: return None, None
        node_parent = node.parent if hasattr(node, "parent") else None
        text = node_parent.get_text(" ", strip=True) if node_parent else str(node)
        return parse_price_and_change(text)

    mapping = [("5D","5[-\\s]?Day"),("1M","1[-\\s]?Month"),("3M","3[-\\s]?Month"),("6M","6[-\\s]?Month"),("1Y","1[-\\s]?Year")]
    result: Dict[str, Optional[Decimal]] = {
        "symbol": symbol, "token_name": token_name or "", "current_price": current_price,
        "pred_5d": None, "chg_5d": None, "pred_1m": None, "chg_1m": None,
        "pred_3m": None, "chg_3m": None, "pred_6m": None, "chg_6m": None,
        "pred_1y": None, "chg_1y": None,
    }
    label_to_key = {"5D":("pred_5d","chg_5d"), "1M":("pred_1m","chg_1m"),
                    "3M":("pred_3m","chg_3m"), "6M":("pred_6m","chg_6m"), "1Y":("pred_1y","chg_1y")}
    for short, rx in mapping:
        price, pct = find_row(rx)
        k_price, k_chg = label_to_key[short]
        if price is not None: result[k_price] = price
        if pct   is not None: result[k_chg] = pct
    return result

def collect_predictions_from_details(seed: List[Dict]) -> List[Dict]:
    session = requests.Session(); session.headers.update(HEADERS)
    out: List[Dict] = []
    for i, c in enumerate(seed, start=1):
        if i > MAX_COINS:
            dlog("[detail] reached MAX_COINS=%s", MAX_COINS); break
        slug = c.get("slug"); if not slug: continue
        html = fetch_prediction_detail(session, slug)
        if not html: continue
        p = parse_prediction_detail(html)
        has_any = any(p.get(k) is not None for k in ["pred_5d","pred_1m","pred_3m","pred_6m","pred_1y"])
        if p.get("symbol") and has_any:
            out.append({
                "symbol": p["symbol"], "token_name": p.get("token_name",""), "current_price": p.get("current_price"),
                "pred_5d": p.get("pred_5d"), "chg_5d": p.get("chg_5d"),
                "pred_1m": p.get("pred_1m"), "chg_1m": p.get("chg_1m"),
                "pred_3m": p.get("pred_3m"), "chg_3m": p.get("chg_3m"),
                "pred_6m": p.get("pred_6m"), "chg_6m": p.get("chg_6m"),
                "pred_1y": p.get("pred_1y"), "chg_1y": p.get("chg_1y"),
            })
        if DETAIL_SLEEP_MS > 0: time.sleep(DETAIL_SLEEP_MS / 1000.0)
    dlog("[detail] collected_items=%s (from seeds=%s)", len(out), len(seed))
    return out

# ===================== CSV I/O a sestavení řádků =====================
def load_csv_rows(container_client, blob_name: str) -> List[Dict]:
    from azure.core.exceptions import ResourceNotFoundError
    blob = container_client.get_blob_client(blob_name)
    try:
        content = blob.download_blob().readall().decode("utf-8", errors="ignore")
    except ResourceNotFoundError:
        return []
    except Exception as e:
        logging.warning("[csv-read] Failed to read existing blob: %s", e); return []
    rows: List[Dict] = []
    with io.StringIO(content) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: r.get(k, "") for k in CSV_FIELDS})
    dlog("[csv-read] loaded rows=%s", len(rows))
    return rows

def write_csv_rows(container_client, blob_name: str, rows: List[Dict]) -> None:
    blob = container_client.get_blob_client(blob_name)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        for k in CSV_FIELDS: r.setdefault(k, "")
        writer.writerow(r)
    data = buf.getvalue().encode("utf-8")
    blob.upload_blob(data, overwrite=True)
    dlog("[csv-write] uploaded rows=%s size=%s", len(rows), len(data))

def build_active_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for it in items:
        symbol = it["symbol"]; token_name = it.get("token_name","")
        cp = it.get("current_price"); cp_str = "" if cp is None else str(cp)
        pairs = [("5D", it.get("pred_5d"), it.get("chg_5d")),
                 ("1M", it.get("pred_1m"), it.get("chg_1m")),
                 ("3M", it.get("pred_3m"), it.get("chg_3m")),
                 ("6M", it.get("pred_6m"), it.get("chg_6m")),
                 ("1Y", it.get("pred_1y"), it.get("chg_1y"))]
        for short, price, pct in pairs:
            if price is None: continue
            _, to_fn = HORIZON_MAP[short]
            model_to = to_fn(scrape_date)
            rows.append({
                "scrape_date": scrape_date.isoformat(),
                "load_ts": load_ts,
                "symbol": symbol,
                "token_name": token_name,
                "current_price": cp_str,
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
            r["is_active"] = "False"; changed += 1
    return changed

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s SCRAPE_SOURCE=%s MAX_PAGES=%s COIN_LIST_MAX_PAGES=%s MAX_COINS=%s",
         VERSION, SCRAPE_SOURCE, MAX_PAGES, COIN_LIST_MAX_PAGES, MAX_COINS)
    dlog("[env] OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s", OUTPUT_CONTAINER, AZURE_BLOB_NAME)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting."); return

    try:
        from azure.storage.blob import BlobServiceClient
        bsc = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        cc = bsc.get_container_client(OUTPUT_CONTAINER)
        try:
            cc.create_container(); dlog("[blob] container created: %s", OUTPUT_CONTAINER)
        except Exception:
            dlog("[blob] container exists: %s", OUTPUT_CONTAINER)

        # --------- výběr zdroje ----------
        items: List[Dict] = []
        if SCRAPE_SOURCE == "COIN_LIST":
            seed = collect_coin_slugs(COIN_LIST_MAX_PAGES, MAX_COINS)
            dlog("[list] seeds=%s", len(seed))
            if not seed:
                if SEED_SLUGS:
                    dlog("[list] using SEED_SLUGS fallback: %s", len(SEED_SLUGS))
                    seed = [{"slug": s} for s in SEED_SLUGS]
                else:
                    dlog("[list] no seeds -> stop"); return
            items = collect_predictions_from_details(seed)

        elif SCRAPE_SOURCE == "PREDICTIONS":
            items = crawl_predictions_index()
            # enrichment přes detail pro doplnění chybějících hodnot (pokud máme slug)
            session = requests.Session(); session.headers.update(HEADERS)
            improved = 0
            for idx, it in enumerate(items):
                slug = it.get("slug")
                if not slug: continue
                html = fetch_prediction_detail(session, slug)
                if not html: continue
                p = parse_prediction_detail(html)
                changed = False
                for short, (hdr, _) in HORIZON_MAP.items():
                    k_price = {"5D":"pred_5d","1M":"pred_1m","3M":"pred_3m","6M":"pred_6m","1Y":"pred_1y"}[short]
                    k_chg   = {"5D":"chg_5d","1M":"chg_1m","3M":"chg_3m","6M":"chg_6m","1Y":"chg_1y"}[short]
                    if it.get(k_price) is None and p.get(k_price) is not None:
                        it[k_price] = p[k_price]; changed = True
                    if it.get(k_chg) is None and p.get(k_chg) is not None:
                        it[k_chg] = p[k_chg]; changed = True
                if it.get("current_price") is None and p.get("current_price") is not None:
                    it["current_price"] = p["current_price"]; changed = True
                if changed: improved += 1
                if DETAIL_SLEEP_MS > 0: time.sleep(DETAIL_SLEEP_MS / 1000.0)
            dlog("[pred] enrichment improved=%s", improved)

        else:  # HYBRID: zkus index, když málo výsledků, dobij coin-listem
            idx_items = crawl_predictions_index()
            dlog("[hybrid] index_items=%s", len(idx_items))
            items = idx_items
            if len(items) < 50:
                seed = collect_coin_slugs(COIN_LIST_MAX_PAGES, MAX_COINS)
                dlog("[hybrid] coin_list seeds=%s", len(seed))
                if seed:
                    more = collect_predictions_from_details(seed)
                    # merge podle symbolu
                    by_sym = {it.get("symbol"): it for it in items if it.get("symbol")}
                    added = 0
                    for it in more:
                        s = it.get("symbol")
                        if s and s not in by_sym:
                            by_sym[s] = it; added += 1
                    items = list(by_sym.values())
                    dlog("[hybrid] merged add=%s total=%s", added, len(items))

        # dedup dle symbolu (jistota)
        uniq = {}
        for it in items:
            s = it.get("symbol")
            if s: uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_items=%s", len(items))

        # --- CSV: načti staré, deaktivuj dnešní True, přidej nové True, zapiš overwrite ---
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
