import pandas_datareader.data as web
import pandas as pd
import os
import time
from datetime import datetime

# Create folder
os.makedirs("data", exist_ok=True)

# List of tickers (Stooq supports US stocks without '^', '.' or '-')
tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "BRK.B",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "XOM",
    "UNH",
    "HD",
    "MA",
    "CVX",
    "ABBV",
    "LLY",
    "PEP",
]

# Download data for each
for ticker in tickers:
    print(f"Downloading {ticker} from Stooq...")
    try:
        df = web.DataReader(ticker, "stooq", start="2000-01-01", end=datetime.today())
        df = df[::-1]  # Reverse because Stooq returns newest first
        df.to_csv(f"data/{ticker.replace('.', '-')}.csv")
    except Exception as e:
        print(f"❌ Failed to download {ticker}: {e}")

    time.sleep(1)  # Just in case

print("✅ Finished downloading all available stocks.")
