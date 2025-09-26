import os
import logging
import datetime
import io
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
import re

def dlog(msg, *args):
    logging.info(msg, *args)

def slug_to_url(slug: str) -> str:
    return f"https://coincodex.com/crypto/{slug}/price-prediction/"

def parse_price_and_change(text: str):
    # najde $ číslo a % číslo
    price, change = None, None
    m_price = re.search(r"\$([\d,]+\.?\d*)", text)
    if m_price:
        price = float(m_price.group(1).replace(",", ""))
    m_change = re.search(r"([+-]?\d+\.?\d*)\s*%", text)
    if m_change:
        change = float(m_change.group(1))
    return price, change

def parse_prediction_detail(html: str, symbol: str, token_name: str):
    soup = BeautifulSoup(html, "html.parser")

    current_price = None
    current_node = soup.find(string=re.compile(r"\bPrice\b", re.I))
    if current_node:
        sec = current_node.parent.get_text(" ", strip=True) if hasattr(current_node, "parent") else str(current_node)
        cp, _ = parse_price_and_change(sec)
        current_price = cp

    def find_row(label_regex: str):
        node = soup.find(string=re.compile(label_regex, re.I))
        if not node:
            th = soup.find("th", string=re.compile(label_regex, re.I))
            node = th if th else None
        if not node:
            return None, None
        cont = node.parent.get_text(" ", strip=True) if hasattr(node, "parent") else str(node)
        return parse_price_and_change(cont)

    preds = {}
    for short, rx in [("5D","5[-\\s]?Day"),("1M","1[-\\s]?Month"),("3M","3[-\\s]?Month"),
                      ("6M","6[-\\s]?Month"),("1Y","1[-\\s]?Year")]:
        p, c = find_row(rx)
        preds[f"pred_{short}"] = p
        preds[f"chg_{short}"] = c

    return {
        "symbol": symbol,
        "token_name": token_name,
        "current_price": current_price,
        **preds
    }

def main(mytimer) -> None:
    VERSION = "5.0-static"
    load_date = datetime.date.today().isoformat()
    load_time = datetime.datetime.utcnow().isoformat()

    OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "models-recalc")
    AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "CoinDeskModels.csv")
    COINLIST_BLOB = os.getenv("COINLIST_BLOB", "CoinList.csv")
    conn_str = os.getenv("AzureWebJobsStorage")

    dlog("[start] version=%s source=STATIC blob=%s", VERSION, COINLIST_BLOB)

    # --- blob client
    bsc = BlobServiceClient.from_connection_string(conn_str)
    container_client = bsc.get_container_client(OUTPUT_CONTAINER)
    container_client.create_container(exist_ok=True)

    # --- načti CoinList.csv
    blob_client = container_client.get_blob_client(COINLIST_BLOB)
    csv_data = blob_client.download_blob().readall().decode("utf-8")
    coinlist = pd.read_csv(io.StringIO(csv_data))

    rows = []
    for _, row in coinlist.iterrows():
        sym, slug, tname = row["symbol"], row["slug"], row["token_name"]
        url = slug_to_url(slug)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                dlog("[detail] slug=%s status=%s", slug, resp.status_code)
                continue
            dlog("[detail] slug=%s ok", slug)
            parsed = parse_prediction_detail(resp.text, sym, tname)
            for short, col in [("5D","pred_5D"),("1M","pred_1M"),("3M","pred_3M"),
                               ("6M","pred_6M"),("1Y","pred_1Y")]:
                pred_price = parsed.get(col)
                if pred_price is None:
                    continue
                rows.append({
                    "date": load_date,
                    "load_time": load_time,
                    "symbol": parsed["symbol"],
                    "token_name": parsed["token_name"],
                    "model_to": short,
                    "predicted_price": pred_price,
                    "current_price": parsed["current_price"],
                    "pct_change": parsed.get(f"chg_{short}"),
                    "is_active": True,
                    "validation": ""
                })
        except Exception as e:
            dlog("[error] slug=%s err=%s", slug, e)

    df_new = pd.DataFrame(rows)

    # --- načti starý CSV
    blob_client = container_client.get_blob_client(AZURE_BLOB_NAME)
    try:
        old_csv = blob_client.download_blob().readall().decode("utf-8")
        df_old = pd.read_csv(io.StringIO(old_csv))
    except Exception:
        df_old = pd.DataFrame()

    # --- zneplatni staré řádky stejného dne/symbolu/modelu
    if not df_old.empty:
        mask = (df_old["date"] == load_date) & df_old["is_active"]
        for _, r in df_new.iterrows():
            df_old.loc[
                mask & (df_old["symbol"]==r["symbol"]) & (df_old["model_to"]==r["model_to"]),
                "is_active"
            ] = False
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # --- zapiš zpět
    out_buf = io.StringIO()
    df_all.to_csv(out_buf, index=False)
    blob_client.upload_blob(out_buf.getvalue(), overwrite=True)
    dlog("[done] uploaded rows=%s", len(df_all))
