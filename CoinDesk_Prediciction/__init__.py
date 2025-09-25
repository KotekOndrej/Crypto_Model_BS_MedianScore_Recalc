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
from urllib.parse import urlparse, parse_qs, urlencode

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# -------------------- Konfigurace --------------------
COIN_LIST_BASE = "https://coincodex.com/crypto/"
PREDICTION_DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/4.0)")
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
PAGE_SLEEP_MS = int(os.getenv("PAGE_SLEEP_MS", "250"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))

# kolik stránek katalogu coinu projít (server-side pagination bývá stabilní)
COIN_LIST_MAX_PAGES = int(os.getenv("COIN_LIST_MAX_PAGES", "10"))
# bezpečnostní limit kolik coinů vyřídit za běh (pro produkci si nastav podle potřeby)
MAX_COINS = int(os.getenv("MAX_COINS", "300"))

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Mapování horizontů + výpočet cílového data
HORIZON_MAP = {
    "5D": ("5D Prediction", lambda d: d + relativedelta(days=5)),
    "1M": ("1M Prediction", lambda d: d + relativedelta(months=1)),
    "3M": ("3M Prediction", lambda d: d + relativedelta(months=3)),
    "6M": ("6M Prediction", lambda d: d + relativedelta(months=6)),
    "1Y": ("1Y Prediction", lambda d: d + relativedelta(years=1)),
}

CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]
CSV_HEADER = ",".join(CSV_FIELDS) + "\n"

# -------------------- Helpery --------------------
def dlog(msg, *args): logging.info(msg, *args)

def parse_decimal_safe(s: str) -> Optional[Decimal]:
    try:
        return Decimal(s)
    except Exception:
        return None

def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z textu typu '$ 4,660.39 11.49%' -> (4660.39, 11.49)"""
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

# -------------------- 1) Seznam coinů: slugs + tickery z /crypto/?page=N --------------------
def fetch_coin_list_page(session: requests.Session, page: int) -> Optional[str]:
    url = build_url_with_page(COIN_LIST_BASE, page)
    try:
        resp = session.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        dlog("[list] page=%s status=%s len=%s", page, resp.status_code, len(resp.text))
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        dlog("[list] error: %s", e)
    return None

def extract_slugs_from_coin_list(html: str) -> List[Dict]:
    """
    Vrátí [{"slug": "ethereum", "symbol": "ETH", "token_name": "Ethereum"}, ...]
    podle tabulky na /crypto/.
    """
    soup = BeautifulSoup(html, "html.parser")
    # heuristika: první tabulka s hlavičkou obsahující "Name" a "Price"
    target_table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if "Name" in headers and "Price" in headers:
            target_table = t
            break
    if not target_table:
        return []

    header_texts = [th.get_text(strip=True) for th in target_table.find_all("th")]
    col_idx = {name: i for i, name in enumerate(header_texts)}
    if "Name" not in col_idx:
        return []
    tbody = target_table.find("tbody")
    if not tbody:
        return []

    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < len(header_texts):
            continue
        name_cell = tds[col_idx["Name"]]
        # link vypadá jako /crypto/ethereum/...
        a = name_cell.find("a", href=True)
        slug = None
        if a and a["href"]:
            m = re.search(r"/crypto/([^/]+)/?", a["href"])
            if m:
                slug = m.group(1).strip()

        # text bývá "ETH Ethereum"
        name_text = name_cell.get_text(" ", strip=True)
        symbol, token_name = None, None
        if name_text:
            parts = name_text.split()
            if len(parts) >= 2:
                symbol, token_name = parts[0], " ".join(parts[1:])
            else:
                token_name = name_text

        if slug and symbol:
            rows.append({"slug": slug, "symbol": symbol, "token_name": token_name or ""})
    return rows

def collect_coin_slugs(max_pages: int, max_coins: int) -> List[Dict]:
    """
    Projde /crypto/?page=1..max_pages a nasbírá slugs + tickery.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    seen = set()
    out: List[Dict] = []
    for p in range(1, max_pages + 1):
        html = fetch_coin_list_page(session, p)
        if not html:
            break
        batch = extract_slugs_from_coin_list(html)
        new = 0
        for it in batch:
            if it["slug"] not in seen:
                seen.add(it["slug"])
                out.append(it)
                new += 1
                if len(out) >= max_coins:
                    dlog("[list] reached MAX_COINS=%s", max_coins)
                    return out
        dlog("[list] page=%s added=%s total=%s", p, new, len(out))
        if new == 0:
            # nic nového – ukonči
            break
        if PAGE_SLEEP_MS > 0:
            time.sleep(PAGE_SLEEP_MS / 1000.0)
    return out

# -------------------- 2) Per-coin: detail predikcí /crypto/{slug}/price-prediction/ --------------------
def fetch_prediction_detail(session: requests.Session, slug: str) -> Optional[str]:
    url = PREDICTION_DETAIL_TMPL.format(slug=slug)
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
    Z detailu predikce vytáhne: current_price (pokud najde), pred_5d/1m/3m/6m/1y a chg_5d/1m/3m/6m/1y.
    Heuristika: hledáme řádky/sekce s textem "5-Day", "1-Month", atd., a z nich vyparsujeme cenu a %.
    """
    soup = BeautifulSoup(html, "html.parser")

    # aktuální cena (volitelně – necháme prázdné, když nenajdeme)
    current_price = None
    # pokus: najdi elementy, kde je "Price" a v okolí dolarová hodnota
    price_candidate = soup.find(string=re.compile(r"Price", re.I))
    if price_candidate:
        section = price_candidate.parent.get_text(" ", strip=True) if hasattr(price_candidate, "parent") else str(price_candidate)
        cp, _ = parse_price_and_change(section)
        current_price = cp or current_price

    # helpers pro horizonty
    def find_row(label_regex: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        row = soup.find(string=re.compile(label_regex, re.I))
        if not row:
            return None, None
        section = row.parent.get_text(" ", strip=True) if hasattr(row, "parent") else str(row)
        return parse_price_and_change(section)

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

def collect_predictions_from_details(coin_seed: List[Dict]) -> List[Dict]:
    """
    Pro každý coin ve seed listu stáhne detail predikce a vrátí normalizované položky:
    {
      symbol, token_name, current_price,
      pred_5d, chg_5d, pred_1m, chg_1m, pred_3m, chg_3m, pred_6m, chg_6m, pred_1y, chg_1y
    }
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    out: List[Dict] = []
    for i, c in enumerate(coin_seed, start=1):
        if i > MAX_COINS:
            dlog("[detail] reached MAX_COINS=%s", MAX_COINS)
            break
        slug = c["slug"]
        symbol = c["symbol"]
        token_name = c.get("token_name", "")

        html = fetch_prediction_detail(session, slug)
        if not html:
            continue
        parsed = parse_prediction_detail(html)
        # aspoň nějaká predikce musí být k dispozici
        if any(parsed.get(k) is not None for k in ["pred_5d","pred_1m","pred_3m","pred_6m","pred_1y"]):
            out.append({
                "symbol": symbol,
                "token_name": token_name,
                "current_price": parsed.get("current_price"),
                "pred_5d": parsed.get("pred_5d"),
                "chg_5d": parsed.get("chg_5d"),
                "pred_1m": parsed.get("pred_1m"),
                "chg_1m": parsed.get("chg_1m"),
                "pred_3m": parsed.get("pred_3m"),
                "chg_3m": parsed.get("chg_3m"),
                "pred_6m": parsed.get("pred_6m"),
                "chg_6m": parsed.get("chg_6m"),
                "pred_1y": parsed.get("pred_1y"),
                "chg_1y": parsed.get("chg_1y"),
            })
        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)

    dlog("[detail] collected_items=%s (from seeds=%s)", len(out), len(coin_seed))
    return out

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
        for k in CSV_FIELDS:
            r.setdefault(k, "")
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
            ("5D", it.get("pred_5d"), it.get("chg_5d")),
            ("1M", it.get("pred_1m"), it.get("chg_1m")),
            ("3M", it.get("pred_3m"), it.get("chg_3m")),
            ("6M", it.get("pred_6m"), it.get("chg_6m")),
            ("1Y", it.get("pred_1y"), it.get("chg_1y")),
        ]
        for short, price, pct in pairs:
            if price is None:
                continue
            header_name, to_fn = HORIZON_MAP[short]
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

# -------------------- Azure Function entrypoint --------------------
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    load_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[CoinDesk_Prediciction] Start %s", scrape_date.isoformat())
    dlog("[env] OUTPUT_CONTAINER=%s AZURE_BLOB_NAME=%s COIN_LIST_MAX_PAGES=%s MAX_COINS=%s",
         OUTPUT_CONTAINER, AZURE_BLOB_NAME, COIN_LIST_MAX_PAGES, MAX_COINS)

    if not STORAGE_CONNECTION_STRING:
        logging.error("[env] AzureWebJobsStorage is NOT set. Exiting.")
        return

    try:
        # 1) Seed slugů/tickerů z katalogu /crypto/
        seed = collect_coin_slugs(COIN_LIST_MAX_PAGES, MAX_COINS)
        dlog("[list] seeds=%s", len(seed))
        if not seed:
            dlog("[list] no seeds -> stop")
            return

        # 2) Per-coin z detailů predikcí
        items = collect_predictions_from_details(seed)
        # dedup dle symbolu (pojistka)
        uniq = {}
        for it in items:
            s = it.get("symbol")
            if s:
                uniq[s] = it
        items = list(uniq.values())
        dlog("[extract] unique_items=%s", len(items))

        # 3) Blob klient
        from azure.storage.blob import BlobServiceClient
        blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
        try:
            container_client.create_container()
            dlog("[blob] container created: %s", OUTPUT_CONTAINER)
        except Exception:
            dlog("[blob] container exists: %s", OUTPUT_CONTAINER)

        # 4) Načti existující CSV, deaktivuj dnešní True a přidej nové aktivní
        existing_rows = load_csv_rows(container_client, AZURE_BLOB_NAME)
        deactivated = deactivate_todays_rows(existing_rows, scrape_date.isoformat())
        new_active_rows = build_active_rows(scrape_date, load_ts, items)
        all_rows = existing_rows + new_active_rows
        dlog("[csv] deactivated_today=%s newly_active=%s final_rows=%s",
             deactivated, len(new_active_rows), len(all_rows))

        # 5) Zapiš celý CSV (overwrite)
        write_csv_rows(container_client, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception in CoinDesk_Prediciction: %s", e)
        logging.error(traceback.format_exc())
        return
