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
VERSION = "8.3-slug-detail-strict+cards"

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
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/8.3)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

# Horizonty, které ukládáme
HORIZON_MAP = {
    "5D": ("5D", lambda d: d + relativedelta(days=5)),
    "1M": ("1M", lambda d: d + relativedelta(months=1)),
    "3M": ("3M", lambda d: d + relativedelta(months=3)),
}

# CSV schema (přidán 'slug')
CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "slug", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]

SAFE_LABEL_BLOCKS = ("tr", "li", "div", "section", "article", "p")

# ===================== Log util =====================
def dlog(msg, *args):
    logging.info(msg, *args)

# ===================== Normalizace textu =====================
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

# ===================== Regexy a utility =====================
# Cena MUSÍ mít dolar a povolujeme čárky i mezery jako oddělovače tisíců
_RX_PRICE = re.compile(
    r"\$\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
)
_RX_PCT   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")

def _to_dec(num: Optional[str]) -> Optional[Decimal]:
    if num is None:
        return None
    num = num.replace(",", "").replace(" ", "")
    try:
        return Decimal(num)
    except (InvalidOperation, AttributeError):
        return None

def parse_price_and_pct(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Z textu vytáhne (price s $ , pct). Cena bez $ se nebere."""
    t = normalize_text(text)
    if not t:
        return None, None
    mp = _RX_PRICE.search(t)
    mc = _RX_PCT.search(t)
    price = _to_dec(mp.group(1)) if mp else None
    pct   = _to_dec(mc.group(1)) if mc else None
    return price, pct

def _txt(el) -> str:
    return normalize_text(el.get_text(" ", strip=True)) if el is not None else ""

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

# ===================== Parser: Current Price =====================
def extract_current_price(soup: BeautifulSoup, slug: str) -> Optional[Decimal]:
    """
    1) Top bar: <li><a href="/crypto/{slug}/"> ... <span class="value"> $ … </span>
    2) Sekundárně bloky s "Current price" / "Live price".
    Žádný jiný fallback.
    """
    # 1) topbar (tolerantní href)
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
                price, _ = parse_price_and_pct(_txt(val))
                if price is not None:
                    dlog("[price] current via topbar -> %s", price)
                    return price
    except Exception:
        pass

    # 2) jasné labely
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
                    if "$" in _txt(anc):
                        blk = anc
                        break
            if not blk:
                continue
            price, _ = parse_price_and_pct(_txt(blk))
            if price is not None:
                dlog("[price] current via label '%s' -> %s", rx.pattern, price)
                return price

    # bez dalšího fallbacku
    return None

# ===================== Parser: Predikce 5D/1M/3M (tabulka i karty) =====================
RX_5D = [re.compile(r"\b5\s*D\b", re.I), re.compile(r"\b5\s*-\s*Day\b", re.I), re.compile(r"\b5\s*Day\b", re.I)]
RX_1M = [re.compile(r"\b1\s*M\b", re.I), re.compile(r"\b1\s*-\s*Month\b", re.I), re.compile(r"\b1\s*Month\b", re.I)]
RX_3M = [re.compile(r"\b3\s*M\b", re.I), re.compile(r"\b3\s*-\s*Month\b", re.I), re.compile(r"\b3\s*Month\b", re.I)]

def _col_index(headers: List[str], patterns: List[re.Pattern]) -> Optional[int]:
    if not headers:
        return None
    for i, h in enumerate(headers):
        for rx in patterns:
            if rx.search(h):
                return i
    return None

def find_predictions_table(soup: BeautifulSoup):
    for tbl in soup.find_all("table"):
        headers = [ _txt(th) for th in tbl.find_all("th") ]
        hcat = " ".join(headers)
        if re.search(r"\b5\s*D|5\s*-\s*Day|5\s*Day", hcat, re.I) and \
           re.search(r"\b1\s*M|1\s*-\s*Month|1\s*Month", hcat, re.I) and \
           re.search(r"\b3\s*M|3\s*-\s*Month|3\s*Month", hcat, re.I):
            return tbl, headers
    return None, None

def parse_predictions_5d_1m_3m(soup: BeautifulSoup) -> Dict[str, Optional[Decimal]]:
    # 1) tabulka
    tbl, headers = find_predictions_table(soup)
    if tbl:
        if not headers:
            headers = []
        if not headers:
            first_tr = tbl.find("tr")
            headers = [ _txt(x) for x in (first_tr.find_all(["th","td"]) if first_tr else []) ]
        idx_5d = _col_index(headers, RX_5D)
        idx_1m = _col_index(headers, RX_1M)
        idx_3m = _col_index(headers, RX_3M)

        body = tbl.find("tbody") or tbl
        for tr in body.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            def val(ix: Optional[int]):
                if ix is None or ix >= len(tds):
                    return None, None
                return parse_price_and_pct(_txt(tds[ix]))
            p5, c5 = val(idx_5d)
            p1, c1 = val(idx_1m)
            p3, c3 = val(idx_3m)
            if any(v is not None for v in (p5, p1, p3, c5, c1, c3)):
                return {"pred_5d": p5, "chg_5d": c5, "pred_1m": p1, "chg_1m": c1, "pred_3m": p3, "chg_3m": c3}

    # 2) karty/divy
    LABELS = {
        "5D": re.compile(r"\b5\s*[- ]*Day\s+Price\s+Prediction\b|\b5D\s+Prediction\b", re.I),
        "1M": re.compile(r"\b1\s*[- ]*Month\s+Price\s+Prediction\b|\b1M\s+Prediction\b", re.I),
        "3M": re.compile(r"\b3\s*[- ]*Month\s+Price\s+Prediction\b|\b3M\s+Prediction\b", re.I),
    }
    out = {"pred_5d": None, "chg_5d": None, "pred_1m": None, "chg_1m": None, "pred_3m": None, "chg_3m": None}

    def find_in_cards(rx: re.Pattern) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        node = soup.find(string=rx)
        if not node:
            # fallback: projdi boxy a hledej label v textu
            for box in soup.find_all(["div","section","article","li"]):
                t = _txt(box)
                if t and rx.search(t) and ("$" in t or "%" in t):
                    return parse_price_and_pct(t)
            return None, None
        # vzhůru po rodičích do bloku se $/%
        for anc in node.parents:
            if getattr(anc, "name", "").lower() in ("div","section","article","li","tr"):
                t = _txt(anc)
                if "$" in t or "%" in t:
                    return parse_price_and_pct(t)
        return None, None

    p5, c5 = find_in_cards(LABELS["5D"])
    p1, c1 = find_in_cards(LABELS["1M"])
    p3, c3 = find_in_cards(LABELS["3M"])

    out.update({"pred_5d": p5, "chg_5d": c5, "pred_1m": p1, "chg_1m": c1, "pred_3m": p3, "chg_3m": c3})
    return out

def parse_prediction_detail(html: str, slug: str) -> Dict[str, Optional[Decimal]]:
    soup = BeautifulSoup(html, "html.parser")

    current_price = extract_current_price(soup, slug)
    preds = parse_predictions_5d_1m_3m(soup)

    # dopočet % z current_price (jen když dává smysl)
    def pct_from_prices(pred: Optional[Decimal], base: Optional[Decimal]) -> Optional[Decimal]:
        try:
            if pred is None or base is None or base == 0:
                return None
            return (pred - base) * Decimal(100) / base
        except Exception:
            return None

    if preds["chg_5d"] is None:
        preds["chg_5d"] = pct_from_prices(preds["pred_5d"], current_price)
    if preds["chg_1m"] is None:
        preds["chg_1m"] = pct_from_prices(preds["pred_1m"], current_price)
    if preds["chg_3m"] is None:
        preds["chg_3m"] = pct_from_prices(preds["pred_3m"], current_price)

    return {"current_price": current_price, **preds}

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
            row = {k: r.get(k, "") for k in CSV_FIELDS}
            rows.append(row)
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

        parsed = parse_prediction_detail(html, slug)
        has_any = any(parsed.get(k) is not None for k in ["pred_5d", "pred_1m", "pred_3m"])
        if not has_any and parsed.get("current_price") is None:
            continue

        out.append({
            "symbol": symbol,
            "slug": slug,
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
        slug   = it.get("slug","")
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
                "slug": slug,
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
