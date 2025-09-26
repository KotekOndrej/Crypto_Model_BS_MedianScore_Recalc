import os
import io
import csv
import re
import time
import logging
import traceback
import datetime as dt
from decimal import Decimal
from typing import Dict, List, Optional

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ===================== Konfigurace =====================
VERSION = "5.7-static-3horizons"

STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")

SYMBOL_SOURCE = os.getenv("SYMBOL_SOURCE", "STATIC").upper()
COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "45"))
DETAIL_SLEEP_MS = int(os.getenv("DETAIL_SLEEP_MS", "120"))
HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; CoincodexPredictionsFunc/5.7)"),
}

DETAIL_TMPL = "https://coincodex.com/crypto/{slug}/price-prediction/"

# Pouze 5D,1M,3M
HORIZON_MAP = {
    "5D": ("5D Prediction", lambda d: d + relativedelta(days=5)),
    "1M": ("1M Prediction", lambda d: d + relativedelta(months=1)),
    "3M": ("3M Prediction", lambda d: d + relativedelta(months=3)),
}

CSV_FIELDS = [
    "scrape_date", "load_ts", "symbol", "token_name", "current_price",
    "horizon", "model_to", "predicted_price", "predicted_change_pct",
    "is_active", "validation"
]

# ===================== Log =====================
def dlog(msg, *args): logging.info(msg, *args)

# ===================== Parser =====================
PRICE_RX = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
PCT_RX   = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*%")

def parse_price_and_change(text: str):
    if not text: return None, None
    price, pct = None, None
    mp = PRICE_RX.search(text)
    if mp:
        price = Decimal(mp.group(1).replace(",", ""))
    mpct = PCT_RX.search(text)
    if mpct:
        pct = Decimal(mpct.group(1))
    return price, pct

def fetch_prediction_detail(session, slug: str) -> Optional[str]:
    try:
        r = session.get(DETAIL_TMPL.format(slug=slug), headers=HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        dlog("[detail] error %s: %s", slug, e)
    return None

def parse_prediction_detail(html: str) -> Dict[str, Optional[Decimal]]:
    soup = BeautifulSoup(html, "html.parser")
    result = {"current_price": None,
              "pred_5d": None,"chg_5d": None,
              "pred_1m": None,"chg_1m": None,
              "pred_3m": None,"chg_3m": None}

    # current price
    cur = soup.find(string=re.compile(r"Current Price", re.I))
    if cur:
        t = cur.find_parent().get_text(" ", strip=True)
        p,_ = parse_price_and_change(t)
        result["current_price"] = p

    # horizons
    mapping = {
        "5D": r"\b5[- ]?Day\b|\b5D\b",
        "1M": r"\b1[- ]?Month\b|\b1M\b",
        "3M": r"\b3[- ]?Month\b|\b3M\b",
    }
    for h, rx in mapping.items():
        node = soup.find(string=re.compile(rx, re.I))
        if node:
            t = node.find_parent().get_text(" ", strip=True)
            p,c = parse_price_and_change(t)
            result[f"pred_{h.lower()}"] = p
            result[f"chg_{h.lower()}"] = c
    return result

# ===================== CSV I/O =====================
def load_csv_rows(cc, blob_name):
    from azure.core.exceptions import ResourceNotFoundError
    bc = cc.get_blob_client(blob_name)
    try:
        data = bc.download_blob().readall().decode("utf-8")
    except ResourceNotFoundError:
        return []
    rows=[]
    with io.StringIO(data) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k:r.get(k,"") for k in CSV_FIELDS})
    return rows

def write_csv_rows(cc, blob_name, rows):
    bc = cc.get_blob_client(blob_name)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    bc.upload_blob(buf.getvalue().encode("utf-8"), overwrite=True)

# ===================== Rows =====================
def build_active_rows(scrape_date, load_ts, items):
    rows=[]
    for it in items:
        for short in ["5D","1M","3M"]:
            p = it.get(f"pred_{short.lower()}")
            c = it.get(f"chg_{short.lower()}")
            if p is None: continue
            _,to_fn = HORIZON_MAP[short]
            rows.append({
                "scrape_date": scrape_date.isoformat(),
                "load_ts": load_ts,
                "symbol": it["symbol"],
                "token_name": it["token_name"],
                "current_price": "" if it.get("current_price") is None else str(it["current_price"]),
                "horizon": short,
                "model_to": to_fn(scrape_date).isoformat(),
                "predicted_price": str(p),
                "predicted_change_pct": "" if c is None else str(c),
                "is_active": "True",
                "validation": ""
            })
    return rows

def deactivate_todays_rows(existing, today):
    ch=0
    for r in existing:
        if r["scrape_date"]==today and r["is_active"]=="True":
            r["is_active"]="False"; ch+=1
    return ch

# ===================== CoinList =====================
def load_coinlist_from_blob(cc, blob_name):
    bc=cc.get_blob_client(blob_name)
    content=bc.download_blob().readall().decode("utf-8")
    out=[]
    with io.StringIO(content) as f:
        reader=csv.DictReader(f)
        for row in reader:
            out.append({"symbol":row["symbol"].upper(),
                        "slug":row["slug"].lower(),
                        "token_name":row["token_name"]})
    return out

def collect_predictions_static(coinlist):
    s=requests.Session(); s.headers.update(HEADERS)
    out=[]
    for coin in coinlist:
        html=fetch_prediction_detail(s, coin["slug"])
        if not html: continue
        parsed=parse_prediction_detail(html)
        if not any(parsed.values()): continue
        out.append({"symbol":coin["symbol"],
                    "token_name":coin["token_name"],
                    **parsed})
        if DETAIL_SLEEP_MS>0: time.sleep(DETAIL_SLEEP_MS/1000.0)
    return out

# ===================== MAIN =====================
def main(mytimer: func.TimerRequest) -> None:
    scrape_date=dt.date.today()
    load_ts=dt.datetime.now(dt.timezone.utc).isoformat()
    dlog("[start] version=%s", VERSION)

    from azure.storage.blob import BlobServiceClient
    bsc=BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    cc=bsc.get_container_client(OUTPUT_CONTAINER)
    try: cc.create_container()
    except: pass

    coinlist=load_coinlist_from_blob(cc, COINLIST_BLOB)
    items=collect_predictions_static(coinlist)

    uniq={it["symbol"]:it for it in items}
    items=list(uniq.values())

    existing=load_csv_rows(cc, AZURE_BLOB_NAME)
    deact=deactivate_todays_rows(existing, scrape_date.isoformat())
    new=build_active_rows(scrape_date, load_ts, items)
    all_rows=existing+new
    dlog("[csv] deactivated=%s newly_active=%s total=%s", deact, len(new), len(all_rows))
    write_csv_rows(cc, AZURE_BLOB_NAME, all_rows)
