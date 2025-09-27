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
VERSION = "8.2-slug-detail-strict"

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
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/8.2)"),
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
_MOJIBAKE = ("Ã¢Â€Â¯", "Ã‚Â ", "Ã¢â‚¬â€")
_WS_CHARS = ["\u00A0", "\u202F", "\u2009", "\u2007", "\u200A"]

def normalize_text(s: str) -> str:
    if not s:
        return ""
    for bad in _MOJIBAKE:
        s = s.replace(bad, " ")
    for ch in _WS_CHARS:
        s = s.replace(ch, " ")
    # zkolabuj whitespace
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

def _text(el) -> str:
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
    2) Sekundárně bloky s "Current price" / "Live price" (v tomtéž boxu).
    Žádný jiný fallback.
    """
    # 1) Top bar s odkazem na /crypto/{slug}/
    try:
        sel = soup.select(f'ul.market-overview a[href="/crypto/{slug}/"]')
        if sel:
            li = sel[0].find_parent("li")
            if li:
                val = li.select_one(".value")
                price, _ = parse_price_and_pct(_text(val or li))
                if price is not None:
                    dlog("[price] current via topbar -> %s", price)
                    return price
    except Exception:
        pass

    # 2) Jasně označený lokální box
    anchors = [
        re.compile(r"\bcurrent\s+price\b", re.I),
        re.compile(r"\blive\s+price\b", re.I),
        re.compile(r"\bprice\s+is\b", re.I),
    ]
    for rx in anchors:
        for n in soup.find_all(string=rx):
            # najdi blízký box se znakem $
            blk = n
            for anc in getattr(n, "parents", []):
                if getattr(anc, "name", "").lower() in SAFE_LABEL_BLOCKS:
                    if "$" in _text(anc):
                        blk = anc
                        break
            price, _ = parse_price_and_pct(_text(blk))
            if price is not None:
                dlog("[price] current via label '%s' -> %s", rx.pattern, price)
                return price

    # žádný další fallback
    return None

# ===================== Parser: Predikční tabulka 5D/1M/3M =====================
def find_predictions_table(soup: BeautifulSoup):
    """
    Najde tabulku/box, který má současně 5D, 1M, 3M ve hlavičce nebo první řádce.
    Vrací (table, headers:list[str]).
    """
    for tbl in soup.find_all("table"):
        headers = [ _text(th) for th in tbl.find_all("th") ]
        hset = " ".join(headers)
        if re.search(r"\b5\s*D|5\s*-\s*Day|5\s*Day", hset, re.I) and \
           re.search(r"\b1\s*M|1\s*-\s*Month|1\s*Month", hset, re.I) and \
           re.search(r"\b3\s*M|3\s*-\s*Month|3\s*Month", hset, re.I):
            return tbl, headers
    # fallback: některé stránky mají hlavičky v prvním <tr> v <tbody>
    for tbl in soup.find_all("table"):
        first_tr = tbl.find("tr")
        if not first_tr:
            continue
        cells = [ _text(x) for x in first_tr.find_all(["th","td"]) ]
        cset = " ".join(cells)
        if re.search(r"\b5\s*D|5\s*-\s*Day|5\s*Day", cset, re.I) and \
           re.search(r"\b1\s*M|1\s*-\s*Month|1\s*Month", cset, re.I) and \
           re.search(r"\b3\s*M|3\s*-\s*Month|3\s*Month", cset, re.I):
            return tbl, cells
    return None, None

def _col_index(headers: List[str], patterns: List[re.Pattern]) -> Optional[int]:
    if not headers:
        return None
    for i, h in enumerate(headers):
        for rx in patterns:
            if rx.search(h):
                return i
    return None

RX_5D = [re.compile(r"\b5\s*D\b", re.I), re.compile(r"\b5\s*-\s*Day\b", re.I), re.compile(r"\b5\s*Day\b", re.I)]
RX_1M = [re.compile(r"\b1\s*M\b", re.I), re.compile(r"\b1\s*-\s*Month\b", re.I), re.compile(r"\b1\s*Month\b", re.I)]
RX_3M = [re.compile(r"\b3\s*M\b", re.I), re.compile(r"\b3\s*-\s*Month\b", re.I), re.compile(r"\b3\s*Month\b", re.I)]

def parse_predictions_5d_1m_3m(soup: BeautifulSoup) -> Dict[str, Optional[Decimal]]:
    """
    Vrátí dict s klíči pred_5d/chg_5d, pred_1m/chg_1m, pred_3m/chg_3m.
    Pokud tabulka není nalezena, vrátí None hodnoty (raději než chytat promo boxy).
    """
    tbl, headers = find_predictions_table(soup)
    if not tbl:
        return {"pred_5d": None, "chg_5d": None, "pred_1m": None, "chg_1m": None, "pred_3m": None, "chg_3m": None}

    # získej hlavičky; pokud nebyly, zkus první řádek
    if not headers or not any(headers):
        first_tr = tbl.find("tr")
        headers = [ _text(x) for x in (first_tr.find_all(["th","td"]) if first_tr else []) ]

    idx_5d = _col_index(headers, RX_5D)
    idx_1m = _col_index(headers, RX_1M)
    idx_3m = _col_index(headers, RX_3M)

    # jdeme skrz řádky a hledáme první „datový“ řádek, který má v uvedených sloupcích $ a/nebo %
    body = tbl.find("tbody") or tbl
    pred_5d = chg_5d = pred_1m = chg_1m = pred_3m = chg_3m = None

    for tr in body.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        def val(ix: Optional[int]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
            if ix is None or ix >= len(tds):
                return None, None
            cell_text = _text(tds[ix])
            # v cellu bereme PRVNÍ $… a PRVNÍ % (nic mimo)
            price, pct = parse_price_and_pct(cell_text)
            return price, pct

        # zkus vyčíst ze stejného řádku
        p5,  c5  = val(idx_5d)
        p1m, c1m = val(idx_1m)
        p3m, c3m = val(idx_3m)

        any_price = any(x is not None for x in (p5, p1m, p3m))
        any_pct   = any(x is not None for x in (c5, c1m, c3m))
        if any_price or any_pct:
            pred_5d, chg_5d = p5, c5
            pred_1m, chg_1m = p1m, c1m
            pred_3m, chg_3m = p3m, c3m
            break

    return {
        "pred_5d": pred_5d, "chg_5d": chg_5d,
        "pred_1m": pred_1m, "chg_1m": chg_1m,
        "pred_3m": pred_3m, "chg_3m": chg_3m,
    }

# ===================== Parser: Detail =====================
def parse_prediction_detail(html: str, slug: str) -> Dict[str, Optional[Decimal]]:
    soup = BeautifulSoup(html, "html.parser")

    current_price = extract_current_price(soup, slug)

    preds = parse_predictions_5d_1m_3m(soup)

    # Pokud % chybí, dopočítej je z current_price (jen pokud dává smysl)
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

    return {
        "current_price": current_price,
        **preds
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
