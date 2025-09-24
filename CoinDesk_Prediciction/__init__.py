import datetime as dt
import logging
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Tuple, List, Dict, Optional

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from azure.storage.blob import BlobServiceClient, ContainerClient, AppendBlobClient

# -------------------- Konfigurace z env proměnných --------------------
COINCIDEX_BASE_URL = "https://coincodex.com/predictions/"
USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/1.2)")
TIMEZONE = os.getenv("APP_TIMEZONE", "Europe/Prague")  # informativní; scrape_date = lokální date.today()

# Použij standardní storage účet Function Appu
STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")

# Nové názvy / defaulty
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "predictions")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "stgbinancedata")  # požadovaný default

# Max. počet stran stránkování
MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Mapování hlaviček tabulky na (zkrácený label, funkci pro výpočet model_to)
HORIZON_MAP = {
    "5D Prediction": ("5D", lambda d: d + relativedelta(days=5)),
    "1M Prediction": ("1M", lambda d: d + relativedelta(months=1)),
    "3M Prediction": ("3M", lambda d: d + relativedelta(months=3)),
    "6M Prediction": ("6M", lambda d: d + relativedelta(months=6)),
    "1Y Prediction": ("1Y", lambda d: d + relativedelta(years=1)),
}

# CSV hlavička – append-only
CSV_HEADER = "scrape_date,symbol,token_name,horizon,model_to,predicted_price,predicted_change_pct\n"


# -------------------- Helpery parsování --------------------
def parse_price_and_change(text: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Z buňky typu: "$ 4,660.39 11.49%" vytáhne:
      - cenu (Decimal) 4660.39
      - procentuální změnu (Decimal) 11.49 (může být záporná)
    Pokud něco chybí, vrací None.
    """
    if not text:
        return None, None

    # cena: první peněžní číslo (povolíme tisícové čárky)
    m_price = re.search(
        r"[-]?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)",
        text
    )
    price_dec = None
    if m_price:
        num = m_price.group(1).replace(",", "")
        try:
            price_dec = Decimal(num)
        except InvalidOperation:
            price_dec = None

    # procento: první výskyt čísla s %
    m_pct = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", text)
    pct_dec = None
    if m_pct:
        try:
            pct_dec = Decimal(m_pct.group(1))
        except InvalidOperation:
            pct_dec = None

    return price_dec, pct_dec


def extract_table_rows(html: str) -> List[Dict]:
    """
    Z HTML /predictions/ vyčte list tokenů s predikcemi a procenty:
    {
      'symbol': 'ETH', 'token_name': 'Ethereum',
      'pred_5d': Decimal, 'chg_5d': Decimal,
      'pred_1m': Decimal, 'chg_1m': Decimal, ...
    }
    """
    soup = BeautifulSoup(html, "lxml")

    table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if any(("5D" in h and "Prediction" in h) for h in headers):
            table = t
            break
    if table is None:
        return []

    header_texts = [th.get_text(strip=True) for th in table.find_all("th")]
    col_idx = {name: i for i, name in enumerate(header_texts)}

    required = ["Name", "5D Prediction", "1M Prediction", "3M Prediction", "6M Prediction", "1Y Prediction"]
    if not all(r in col_idx for r in required):
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    rows: List[Dict] = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < len(header_texts):
            continue

        # Name: typicky "ETH Ethereum"
        name_cell = tds[col_idx["Name"]]
        name_text = name_cell.get_text(" ", strip=True)
        symbol = None
        token_name = None
        if name_text:
            parts = name_text.split()
            if len(parts) >= 2:
                symbol = parts[0]
                token_name = " ".join(parts[1:])
            else:
                token_name = name_text

        if not symbol:
            a = name_cell.find("a")
            if a and a.get_text(strip=True):
                atext = a.get_text(" ", strip=True)
                parts = atext.split()
                if len(parts) >= 2:
                    symbol = parts[0]
                    token_name = " ".join(parts[1:])
                else:
                    token_name = atext

        preds: Dict[str, Optional[Decimal]] = {}
        chgs: Dict[str, Optional[Decimal]] = {}
        for header in required[1:]:
            cell_text = tds[col_idx[header]].get_text(" ", strip=True)
            price, pct = parse_price_and_change(cell_text)
            preds[header] = price
            chgs[header] = pct

        if symbol and any(preds.values()):
            rows.append({
                "symbol": symbol,
                "token_name": token_name or "",
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


def fetch_predictions_page(page: Optional[int] = None) -> Optional[str]:
    url = COINCIDEX_BASE_URL if not page or page == 1 else f"{COINCIDEX_BASE_URL}?page={page}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
    except Exception as e:
        logging.warning("Request error for %s: %s", url, e)
        return None

    if resp.status_code != 200:
        logging.warning("Failed to GET %s -> %s", url, resp.status_code)
        return None
    return resp.text


def iter_all_pages(max_pages: int = MAX_PAGES):
    for p in range(1, max_pages + 1):
        html = fetch_predictions_page(p)
        if not html:
            break
        rows = extract_table_rows(html)
        if not rows:
            break
        for r in rows:
            yield r


# -------------------- Blob helpery --------------------
def ensure_container(client: BlobServiceClient, container_name: str) -> ContainerClient:
    container_client = client.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception:
        # container pravděpodobně existuje
        pass
    return container_client


def append_csv_lines(container_client: ContainerClient, blob_name: str, lines: List[str]) -> None:
    """
    Append-only zapisuje do Append Blobu.
    Pokud blob neexistuje, vytvoří se a zapíše se hlavička.
    """
    append_client: AppendBlobClient = container_client.get_blob_client(blob_name).as_append_blob_client()
    try:
        if not append_client.exists():
            append_client.create_blob()
            append_client.append_block(CSV_HEADER.encode("utf-8"))
    except Exception as e:
        logging.exception("Failed to create/ensure append blob: %s", e)
        raise

    payload = "".join(lines).encode("utf-8")
    if payload:
        append_client.append_block(payload)


def build_csv_rows(scrape_date: dt.date, items: List[Dict]) -> List[str]:
    """
    Vytvoří CSV řádky pro všechny horizonty.
    CSV: scrape_date,symbol,token_name,horizon,model_to,predicted_price,predicted_change_pct
    """
    rows: List[str] = []

    for it in items:
        symbol = it["symbol"]
        token_name = it.get("token_name", "")

        pairs = [
            ("5D", it.get("pred_5d"), it.get("chg_5d"), "5D Prediction"),
            ("1M", it.get("pred_1m"), it.get("chg_1m"), "1M Prediction"),
            ("3M", it.get("pred_3m"), it.get("chg_3m"), "3M Prediction"),
            ("6M", it.get("pred_6m"), it.get("chg_6m"), "6M Prediction"),
            ("1Y", it.get("pred_1y"), it.get("chg_1y"), "1Y Prediction"),
        ]
        for label, price, pct, header_name in pairs:
            if price is None:
                continue
            _, to_fn = HORIZON_MAP[header_name]
            model_to = to_fn(scrape_date)
            pct_str = "" if pct is None else str(pct)  # pokud procento není uvedeno, nech prázdné
            row = f"{scrape_date.isoformat()},{symbol},{token_name},{label},{model_to.isoformat()},{price},{pct_str}\n"
            rows.append(row)

    return rows


# -------------------- Azure Function entrypoint --------------------
def main(mytimer: func.TimerRequest) -> None:
    scrape_date = dt.datetime.now().date()
    logging.info("[CoinDesk_Prediciction] Start scrape for %s", scrape_date.isoformat())

    all_items = list(iter_all_pages())
    logging.info("Extracted %d tokens across pages", len(all_items))

    if not all_items:
        logging.warning("No predictions extracted; nothing to write.")
        return

    csv_lines = build_csv_rows(scrape_date, all_items)
    logging.info("Prepared %d CSV lines to append", len(csv_lines))

    if not STORAGE_CONNECTION_STRING:
        logging.error("AzureWebJobsStorage is not set. Aborting.")
        return

    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    container_client = ensure_container(blob_service, OUTPUT_CONTAINER)
    append_csv_lines(container_client, AZURE_BLOB_NAME, csv_lines)

    logging.info("Append completed -> container=%s blob=%s", OUTPUT_CONTAINER, AZURE_BLOB_NAME)
