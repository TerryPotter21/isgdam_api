from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTHORIZED_CODE = "freelunch"

@app.api_route("/", methods=["GET", "POST"], response_class=HTMLResponse)
async def login_page(request: Request):
    if request.method == "POST":
        form = await request.form()
        password = form.get("password")
        if password == AUTHORIZED_CODE:
            return RedirectResponse(url="/instructions", status_code=303)
        else:
            return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid access code"})
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.get("/instructions", response_class=HTMLResponse)
async def show_instructions(request: Request):
    today = datetime.now()
    current_month = today.strftime("%Y-%m")
    start_date = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo", auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(0)

    latest_date = spy.index.max()
    latest_month = latest_date.strftime("%Y-%m") if pd.notna(latest_date) else "Error"
    is_current = latest_month == current_month

    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "current_month": current_month,
        "latest_date": latest_month,
        "is_current": is_current
    })

@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
    tickers = [
        'AAPL', 'MSFT', 'NVDA',  # Tech
        'JNJ', 'PFE', 'ABBV',    # Healthcare
        'XOM', 'CVX', 'COP',     # Energy
        'JPM', 'BAC', 'GS',      # Financials
        'WMT', 'COST', 'TGT',    # Consumer Staples
        'NKE', 'HD', 'LOW'       # Consumer Discretionary
    ]

    today = datetime.now()
    start_date = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    # Get SPY data
    spy_data = yf.download("SPY", start=start_date, end=end_date, interval="1mo", auto_adjust=True)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.droplevel(0)

    if "Close" not in spy_data.columns:
        print("ERROR: SPY data missing 'Close'")
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": {"Error": "SPY data missing 'Close'"}
        })

    spy_returns = spy_data["Close"].pct_change().sub(0.024 / 12).fillna(0)
    spy_returns.name = 'SPY 1R'

    all_data = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)

            if "Close" not in df.columns or df.empty:
                continue

            sector = yf.Ticker(ticker).info.get("sector", "N/A")
            if sector == "N/A":
                continue

            df = df[["Close"]].copy()
            df["Date"] = df.index
            df["Ticker"] = ticker
            df["Sector"] = sector
            df["1R"] = df["Close"].pct_change().sub(0.024 / 12)
            df["SPY 1R"] = df["Date"].map(spy_returns.to_dict())
            df.reset_index(drop=True, inplace=True)
            all_data.append(df)
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

    if not all_data:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": {"Error": "No data retrieved."}
        })

    df = pd.concat(all_data, ignore_index=True).dropna(subset=["1R", "SPY 1R"])
    df["Month"] = df["Date"].dt.to_period("M")

    results = []
    weights = [0.01, 0.01, 0.04, 0.04, 0.09, 0.09, 0.16, 0.16, 0.25, 0.25, 0.36, 0.36]

    for ticker in df["Ticker"].unique():
        tdf = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(tdf) < 13:
            continue

        try:
            # 12R
            r12 = (tdf.loc[12, "Close"] - tdf.loc[0, "Close"]) / tdf.loc[0, "Close"]
            # 12WR
            wr12 = np.dot(tdf.loc[1:12, "1R"], weights)
            # 6V
            v6 = tdf.loc[7:12, "1R"].std(ddof=1) * np.sqrt(2)
            # 6B
            cov = np.cov(tdf.loc[7:12, "1R"], tdf.loc[7:12, "SPY 1R"])[0][1]
            var = np.var(tdf.loc[7:12, "SPY 1R"], ddof=1)
            b6 = cov / var if var != 0 else np.nan
            # DAM
            dam = (r12 * wr12) / (v6 * b6) if all(pd.notna([r12, wr12, v6, b6])) and v6 * b6 != 0 else np.nan
            sector = tdf.loc[0, "Sector"]
            if pd.notna(dam):
                results.append({"Ticker": ticker, "Sector": sector, "DAM": dam})
        except Exception as e:
            print(f"Error in DAM for {ticker}: {e}")
            continue

    result_df = pd.DataFrame(results).dropna()

    top_tickers = (
        result_df.sort_values(by="DAM", ascending=False)
        .groupby("Sector")
        .first()
        .reset_index()
    )

    result = {}
    for _, row in top_tickers.iterrows():
        result[row["Sector"]] = {"top": row["Ticker"]}

    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": result
    })
