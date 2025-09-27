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

VERSION = "8.4-debug-dump"

# Azure
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

# Coin list
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")

# HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/8.4)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

HORIZON_MAP = {
    "5D": ("5D", lambda d: d + relativedelta(days=5)),
    "1M": ("1M", lambda d: d + relativedelta(months=1)),
    "3M": ("3M", lambda d: d + relativedelta(months=3)),
}

# CSV schema – rozšířeno o debug sloupce
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "slug", "token_name",
    "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "page_url", "html_len",
    "current_block", "pred5d_block", "pred1m_block", "pred3m_block",
    "raw_html_minified",
    "is_active", "validation"
]

SAFE_LABEL_BLOCKS = ("tr","li","div","section","article","p")

def dlog(msg, *args):
    logging.info(msg, *args)

# ========= normalizace / regex =========
_MOJIBAKE = ("Ã¢Â€Â¯", "Ã‚Â ", "Ã¢â‚¬â€", "Ã¢â‚¬â„¢", "â€“", "â€”")
_WS_CHARS = ["\u00A0", "\u202F", "\u2009", "\u2007", "\u200A"]

def normalize_text(s: str) -> str:
    if not s:
        return ""
    for bad in _MOJIBAKE:
        s = s.replace(bad, " ")
    for ch in _WS_CHARS:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

_RX_PRICE = re.compile(r"\$\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)")
_RX_PCT   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")

def _to_dec(num: Optional[str]) -> Optional[Decimal]:
    if num is None: return None
    num = num.replace(",", "").replace(" ", "")
    try: return Decimal(num)
    except (InvalidOperation, AttributeError): return None

def parse_price_and_pct(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    t = normalize_text(text)
    if not t: return None, None
    mp = _RX_PRICE.search(t)
    mc = _RX_PCT.search(t)
    price = _to_dec(mp.group(1)) if mp else None
    pct   = _to_dec(mc.group(1)) if mc else None
    return price, pct

def _txt(el) -> str:
    return normalize_text(el.get_text(" ", strip=True)) if el is not None else ""

# ========= HTTP =========
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

# ========= Parsování (vrací i DEBUG bloky) =========
def extract_current_price_and_block(soup: BeautifulSoup, slug: str) -> Tuple[Optional[Decimal], str]:
    # 1) topbar
    try:
        candidates = soup.select(
            f'ul.market-overview a[href="/crypto/{slug}/"], '
            f'ul.market-overview a[href*="/crypto/{slug}/"]'
        )
        if candidates:
            a = candidates[0]
            li = a.find_parent("li") or a.parent
            if li:
                val = li.select_one(".value") or li
                block_txt = _txt(val if val else li)
                price, _ = parse_price_and_pct(block_txt)
                if price is not None:
                    dlog("[price] current via topbar -> %s", price)
                    return price, block_txt
    except Exception:
        pass

    # 2) labely
    anchors = [
        re.compile(r"\bcurrent\s+price\b", re.I),
        re.compile(r"\blive\s+price\b", re.I),
        re.compile(r"\bprice\s+is\b", re.I),
    ]
    for rx in anchors:
        for n in soup.find_all(string=rx):
            blk = None
            for anc in getattr(n, "parents", []):
                if getattr(anc, "name", "").lower() in SAFE_LABEL_BLOCKS:
                    t = _txt(anc)
                    if "$" in t:
                        blk = anc
                        break
            if not blk: 
                continue
            block_txt = _txt(blk)
            price, _ = parse_price_and_pct(block_txt)
            if price is not None:
                dlog("[price] current via label '%s' -> %s", rx.pattern, price)
                return price, block_txt

    return None, ""

RX_5D = [re.compile(r"\b5\s*D\b", re.I), re.compile(r"\b5\s*-\s*Day\b", re.I), re.compile(r"\b5\s*Day\b", re.I)]
RX_1M = [re.compile(r"\b1\s*M\b", re.I), re.compile(r"\b1\s*-\s*Month\b", re.I), re.compile(r"\b1\s*Month\b", re.I)]
RX_3M = [re.compile(r"\b3\s*M\b", re.I), re.compile(r"\b3\s*-\s*Month\b", re.I), re.compile(r"\b3\s*Month\b", re.I)]

def parse_predictions_blocks(soup: BeautifulSoup) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Vrátí bloky textu pro 5D/1M/3M (pro debug) + z nich spočítané ceny/%. 
    {'5D': {'block': '...','price': Decimal|None,'pct': Decimal|None}, ...}
    """
    out = {"5D": {"block":"", "price":None, "pct":None},
           "1M": {"block":"", "price":None, "pct":None},
           "3M": {"block":"", "price":None, "pct":None}}
    LABELS = {
        "5D": re.compile(r"\b5\s*[- ]*Day\s+Price\s+Prediction\b|\b5D\s+Prediction\b|\b5\s*Day\b", re.I),
        "1M": re.compile(r"\b1\s*[- ]*Month\s+Price\s+Prediction\b|\b1M\s+Prediction\b|\b1\s*Month\b", re.I),
        "3M": re.compile(r"\b3\s*[- ]*Month\s+Price\s+Prediction\b|\b3M\s+Prediction\b|\b3\s*Month\b", re.I),
    }

    def nearest_box_with_symbols(node):
        for anc in node.parents:
            if getattr(anc, "name","").lower() in ("div","section","article","li","tr","td"):
                t = _txt(anc)
                if "$" in t or "%" in t:
                    return anc
        return None

    for key, rx in LABELS.items():
        node = soup.find(string=rx)
        box = None
        if node:
            box = nearest_box_with_symbols(node)
        if not box:
            # fallback: projdi boxy a hledej label v textu
            for b in soup.find_all(["div","section","article","li","tr"]):
                t = _txt(b)
                if t and rx.search(t) and ("$" in t or "%" in t):
                    box = b
                    break
        if box:
            t = _txt(box)
            out[key]["block"] = t
            price, pct = parse_price_and_pct(t)
            out[key]["price"] = price
            out[key]["pct"]   = pct

    return out

def parse_prediction_detail(html: str, slug: str) -> Dict[str, Optional[Decimal]]:
    soup = BeautifulSoup(html, "html.parser")

    current_price, current_block = extract_current_price_and_block(soup, slug)
    blocks = parse_predictions_blocks(soup)

    # sestav návrat + debug texty
    result = {
        "current_price": current_price,
        "current_block": current_block,
        "pred_5d": blocks["5D"]["price"], "chg_5d": blocks["5D"]["pct"], "pred5d_block": blocks["5D"]["block"],
        "pred_1m": blocks["1M"]["price"], "chg_1m": blocks["1M"]["pct"], "pred1m_block": blocks["1M"]["block"],
        "pred_3m": blocks["3M"]["price"], "chg_3m": blocks["3M"]["pct"], "pred3m_block": blocks["3M"]["block"],
    }

    # dopočet % z current_price pokud chybí
    def pct_from_prices(pred: Optional[Decimal], base: Optional[Decimal]) -> Optional[Decimal]:
        try:
            if pred is None or base is None or base == 0:
                return None
            return (pred - base) * Decimal(100) / base
        except Exception:
            return None

    if result["chg_5d"] is None:
        result["chg_5d"] = pct_from_prices(result["pred_5d"], current_price)
    if result["chg_1m"] is None:
        result["chg_1m"] = pct_from_prices(result["pred_1m"], current_price)
    if result["chg_3m"] is None:
        result["chg_3m"] = pct_from_prices(result["pred_3m"], current_price)

    return result

# ========= CSV I/O =========
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
            rows.append({k: r.get(k, "") for k in CSV_FIELDS})
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

def deactivate_todays_rows(existing: List[Dict], today_iso: str) -> int:
    changed = 0
    for r in existing:
        if r.get("scrape_date") == today_iso and str(r.get("is_active")).strip().lower() == "true":
            r["is_active"] = "False"
            changed += 1
    return changed

# ========= CoinList =========
def load_coinlist_from_blob(container_client, blob_name: str) -> List[Dict]:
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

# ========= Sběr =========
def collect_predictions_by_slug(coinlist: List[Dict]) -> List[Dict]:
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
            # i tak založíme RAW řádek s minimem info
            out.append({
                "symbol": symbol, "slug": slug, "token_name": token_name,
                "page_url": url, "html_len": 0, "raw_html_minified": "",
                "current_price": None,
                "current_block": "", "pred5d_block": "", "pred1m_block": "", "pred3m_block": "",
                "pred_5d": None, "chg_5d": None, "pred_1m": None, "chg_1m": None, "pred_3m": None, "chg_3m": None,
            })
            continue

        parsed = parse_prediction_detail(html, slug)
        raw_min = normalize_text(html)
        if len(raw_min) > 5000:
            raw_min = raw_min[:5000]

        item = {
            "symbol": symbol, "slug": slug, "token_name": token_name,
            "page_url": url, "html_len": len(html), "raw_html_minified": raw_min,
            **parsed
        }
        out.append(item)

        if DETAIL_SLEEP_MS > 0:
            time.sleep(DETAIL_SLEEP_MS / 1000.0)

    dlog("[detail] collected_items=%s (from slugs=%s)", len(out), len(coinlist))
    return out

# ========= Build rows =========
def build_rows(scrape_date: dt.date, load_ts: str, items: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for it in items:
        symbol = it["symbol"]
        slug   = it.get("slug","")
        token_name = it.get("token_name","")
        curr = it.get("current_price")
        curr_s = "" if curr is None else str(curr)

        # 1) vždy ulož RAW řádek
        rows.append({
            "scrape_date": scrape_date.isoformat(),
            "load_ts": load_ts,
            "symbol": symbol,
            "slug": slug,
            "token_name": token_name,
            "current_price": curr_s,
            "horizon": "RAW",
            "model_to": "",
            "predicted_price": "",
            "predicted_change_pct": "",
            "page_url": it.get("page_url",""),
            "html_len": str(it.get("html_len","")),
            "current_block": it.get("current_block",""),
            "pred5d_block": it.get("pred5d_block",""),
            "pred1m_block": it.get("pred1m_block",""),
            "pred3m_block": it.get("pred3m_block",""),
            "raw_html_minified": it.get("raw_html_minified",""),
            "is_active": "True",
            "validation": ""
        })

        # 2) pokud jsou predikce, přidej standardní 5D/1M/3M řádky
        pairs = [
            ("5D", it.get("pred_5d"), it.get("chg_5d")),
            ("1M", it.get("pred_1m"), it.get("chg_1m")),
            ("3M", it.get("pred_3m"), it.get("chg_3m")),
        ]
        for short, price, pct in pairs:
            if price is None and pct is None:
                continue
            _, to_fn = HORIZON_MAP[short]
            model_to = to_fn(scrape_date)
            rows.append({
                "scrape_date": scrape_date.isoformat(),
                "load_ts": load_ts,
                "symbol": symbol,
                "slug": slug,
                "token_name": token_name,
                "current_price": curr_s,
                "horizon": short,
                "model_to": model_to.isoformat(),
                "predicted_price": "" if price is None else str(price),
                "predicted_change_pct": "" if pct is None else str(pct),
                "page_url": it.get("page_url",""),
                "html_len": str(it.get("html_len","")),
                "current_block": it.get("current_block",""),
                "pred5d_block": it.get("pred5d_block",""),
                "pred1m_block": it.get("pred1m_block",""),
                "pred3m_block": it.get("pred3m_block",""),
                "raw_html_minified": it.get("raw_html_minified",""),
                "is_active": "True",
                "validation": ""
            })
    return rows

# ========= MAIN =========
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

        # 1) CoinList
        coinlist = load_coinlist_from_blob(cc, COINLIST_BLOB)
        if not coinlist:
            dlog("[coinlist] empty -> stop")
            return

        # 2) stáhni a poskládej položky (vždy i RAW)
        items = collect_predictions_by_slug(coinlist)

        # 3) (ne)deduplikuj — chceme vše logovat; když chceš, můžeš ponechat uniq dle symbolu
        # uniq = {}
        # for it in items: uniq[it["symbol"]] = it
        # items = list(uniq.values())

        # 4) CSV overwrite s deaktivací dnešních
        existing = load_csv_rows(cc, AZURE_BLOB_NAME)
        deact = 0
        # pokud nechceš zneaktivňovat RAW, nech to na 0:
        # deact = deactivate_todays_rows(existing, scrape_date.isoformat())

        new_rows = build_rows(scrape_date, load_ts, items)
        all_rows = existing + new_rows
        dlog("[csv] newly_added=%s final_rows=%s", len(new_rows), len(all_rows))
        write_csv_rows(cc, AZURE_BLOB_NAME, all_rows)
        dlog("[done] Overwrite completed.")

    except Exception as e:
        logging.error("[fatal] Unhandled exception: %s", e)
        logging.error(traceback.format_exc())
        return
