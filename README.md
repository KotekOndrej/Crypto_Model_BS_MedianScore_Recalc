# BS Function App

Azure Function pro výpočet optimálních Buy/Sell (B,S) úrovní z minutových dat kryptoměn.

## Jak to funguje
- Načítá CSV soubory z blob storage (market-data).
- Z posledních 20 dní určuje nejlepší (B,S) pár pro následující den.
- Výsledek ukládá do výstupního blob storage (market-signals) v CSV formátu.

## Výstupní CSV
Každý řádek obsahuje:
- `pair` – název obchodního páru (např. XRPUSDT)
- `B` – hodnota buy
- `S` – hodnota sell
- `gap_pct` – rozdíl mezi S a B v procentech
- `date` – datum, pro který je nastavení platné
- `model` – použitý model (BS_MedianScore)
