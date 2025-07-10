
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

PASSWORD = os.getenv("APP_PASSWORD", "finance")

@app.get("/")
async def root():
    return templates.TemplateResponse("login.html", {"request": {}})

@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == PASSWORD:
        return RedirectResponse(url="/instructions", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid password"})

@app.get("/instructions")
async def show_instructions(request: Request):
    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo", auto_adjust=True)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(0)

    if "Close" not in spy.columns or spy.empty:
        print("SPY data missing 'Close'")
        return templates.TemplateResponse("instructions.html", {"request": request, "latest": "Error", "using_current": False})

    latest = spy.index.max()
    using_current = latest.month == datetime.today().month and latest.year == datetime.today().year
    latest_str = latest.strftime("%Y-%m") if not pd.isnull(latest) else "Error"

    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "latest": latest_str,
        "using_current": using_current
    })

@app.post("/tickers")
async def get_tickers(request: Request):
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "JPM", "JNJ", "V", "UNH",
        "NVDA", "HD", "PG", "MA", "DIS", "PEP", "KO", "MRK", "PFE", "T"
    ]
    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    rf_rate = 0.024 / 12

    try:
        spy_data = yf.download("SPY", start=start_date, end=end_date, interval="1mo", auto_adjust=True)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.droplevel(0)
        if "Close" not in spy_data.columns:
            raise ValueError("SPY data missing 'Close'")
        spy_data = spy_data[["Close"]].copy()
        spy_data = spy_data.reset_index()
        spy_data["1R"] = spy_data["Close"].pct_change().fillna(0)
        spy_data["Excess"] = spy_data["1R"] - rf_rate
    except Exception as e:
        print("ERROR: SPY data missing 'Close'")
        return templates.TemplateResponse("tickers.html", {"request": request, "top_tickers": [{"sector": "Error", "ticker": ""}]})

    results = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            if "Close" not in df.columns:
                continue
            df = df[["Close"]].copy().reset_index()
            df["Date"] = pd.to_datetime(df["Date"])
            df["1R"] = df["Close"].pct_change().fillna(0)

            if len(df) < 13:
                continue

            initial = df["Close"].iloc[-13]
            final = df["Close"].iloc[-1]
            R12 = (final - initial) / initial

            weights = np.array([0.01, 0.01, 0.04, 0.04, 0.09, 0.09, 0.16, 0.16, 0.25, 0.25, 0.36, 0.36])
            returns = df["1R"].iloc[-12:].values
            WR12 = np.dot(returns, weights)

            volatility = df["1R"].iloc[-6:].std() * np.sqrt(2)

            merged = pd.merge(df, spy_data, on="Date", how="inner")
            if merged.empty:
                continue
            beta = np.polyfit(merged["Excess"], merged["1R"], 1)[0]

            dam = (R12 * WR12) / (volatility * beta)
            sector = yf.Ticker(ticker).info.get("sector", None)
            if not sector:
                continue
            results.append({"ticker": ticker, "sector": sector, "dam": dam})
        except Exception as e:
            continue

    df = pd.DataFrame(results)
    top = df.sort_values("dam", ascending=False).dropna().groupby("sector").first().reset_index()
    top_tickers = [{"sector": row["sector"], "ticker": row["ticker"]} for _, row in top.iterrows()]

    return templates.TemplateResponse("tickers.html", {"request": request, "top_tickers": top_tickers})
