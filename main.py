from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import yfinance as yf
from datetime import datetime
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

PASSWORD = os.getenv("APP_PASSWORD", "test123")

def get_monthly_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1mo", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        if "Close" not in df.columns or df["Close"].dropna().empty:
            print(f"ERROR: {ticker} data missing 'Close'")
            return None
        return df[["Close"]]
    except Exception as e:
        print(f"ERROR retrieving {ticker}: {e}")
        return None

@app.get("/")
async def root():
    return RedirectResponse("/login")

@app.get("/login")
async def show_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == PASSWORD:
        return RedirectResponse(url="/instructions", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid password"})

@app.get("/instructions")
async def show_instructions(request: Request):
    end_date = datetime.today()
    start_date = end_date - pd.DateOffset(months=13)
    spy = get_monthly_data("SPY", start_date, end_date)
    is_current = False
    if spy is not None:
        latest = spy.index.max()
        if pd.notna(latest):
            is_current = latest.month == end_date.month and latest.year == end_date.year
    return templates.TemplateResponse("instructions.html", {"request": request, "is_current": is_current})

@app.post("/tickers")
async def get_tickers(request: Request):
    tickers = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "UNH", "JNJ", "XOM", "JPM",
        "V", "PG", "LLY", "TSLA", "MA", "AVGO", "HD", "MRK", "PEP", "ABBV"
    ]
    end_date = datetime.today()
    start_date = end_date - pd.DateOffset(months=13)

    spy_data = get_monthly_data("SPY", start_date, end_date)
    if spy_data is None or "Close" not in spy_data.columns:
        print("SPY data missing 'Close'")
        return templates.TemplateResponse("tickers.html", {"request": request, "top_tickers": [{"sector": "Error", "ticker": ""}]})

    spy_returns = spy_data["Close"].pct_change().sub(0.024 / 12).fillna(0)
    results = []

    for ticker in tickers:
        df = get_monthly_data(ticker, start_date, end_date)
        if df is None or len(df) < 13:
            continue
        try:
            df["1R"] = df["Close"].pct_change().fillna(0)
            df["12R"] = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
            weights = [0.01, 0.01, 0.04, 0.04, 0.09, 0.09, 0.16, 0.16, 0.25, 0.25, 0.36, 0.36]
            df["12WR"] = pd.Series(weights, index=df.index[-12:]) @ df["1R"].iloc[-12:]
            df["6V"] = df["1R"].iloc[-6:].std() * (2 ** 0.5)
            beta = df["1R"].iloc[-12:].cov(spy_returns.iloc[-12:]) / spy_returns.iloc[-12:].var()
            df["6B"] = beta
            df["DAM"] = (df["12R"] * df["12WR"]) / (df["6V"] * df["6B"])
            final_score = df["DAM"].iloc[-1]

            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get("sector", "Unknown")

            if sector != "Unknown":
                results.append({"ticker": ticker, "sector": sector, "DAM": final_score})
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    df_results = pd.DataFrame(results)
    top = df_results.sort_values("DAM", ascending=False).dropna(subset=["sector"])
    top_by_sector = top.groupby("sector").first().reset_index()
    top_list = [{"sector": row["sector"], "ticker": row["ticker"]} for _, row in top_by_sector.iterrows()]

    return templates.TemplateResponse("tickers.html", {"request": request, "top_tickers": top_list})
