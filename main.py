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
    spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo")

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(0)

    if "Adj Close" in spy.columns:
        latest_date = spy.index.max().strftime("%Y-%m") if not spy.empty else "N/A"
        is_current = latest_date == current_month
    else:
        latest_date = "Missing 'Adj Close'"
        is_current = False

    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "current_month": current_month,
        "latest_date": latest_date,
        "is_current": is_current
    })

@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'JNJ', 'PFE', 'ABBV',
        'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'GS',
        'WMT', 'COST', 'TGT', 'NKE', 'HD', 'LOW'
    ]

    today = datetime.now()
    start_date = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    weights = [0.01, 0.01, 0.04, 0.04, 0.09, 0.09, 0.16, 0.16, 0.25, 0.25, 0.36, 0.36]

    spy_data = yf.download("SPY", start=start_date, end=end_date, interval="1mo")
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.droplevel(0)

    if "Adj Close" not in spy_data.columns:
        print("ERROR: SPY data missing 'Adj Close'")
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": {"Error": "SPY data missing 'Adj Close'"}
        })

    spy_returns = spy_data["Adj Close"].pct_change().sub(0.024 / 12).fillna(0)
    spy_returns.name = "SPY 1R"
    spy_dict = spy_returns.to_dict()

    all_data = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            if "Adj Close" not in df.columns:
                print(f"Skipping {ticker}: no 'Adj Close'")
                continue
            df = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
            info = yf.Ticker(ticker).info
            sector = info.get("sector", None)
            if sector is None or pd.isna(sector):
                print(f"Skipping {ticker}: no sector info")
                continue
            df["Ticker"] = ticker
            df["Sector"] = sector
            df["Date"] = df.index
            df["1R"] = df["Close"].pct_change().sub(0.024 / 12)
            df["SPY 1R"] = df["Date"].map(spy_dict)
            df.dropna(subset=["1R", "SPY 1R"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            all_data.append(df)
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

    if not all_data:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": {"Error": "No data could be retrieved."}
        })

    df = pd.concat(all_data)
    df["Month"] = df["Date"].dt.to_period("M")

    results = []
    for ticker in df["Ticker"].unique():
        tdf = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(tdf) < 13:
            continue
        try:
            r12 = (tdf.loc[12, "Close"] - tdf.loc[0, "Close"]) / tdf.loc[0, "Close"]
            wr12 = sum(tdf.loc[i, "1R"] * weights[i] for i in range(12))
            v6 = tdf.loc[:5, "1R"].std(ddof=1) * np.sqrt(2)
            b6 = np.cov(tdf.loc[:5, "1R"], tdf.loc[:5, "SPY 1R"])[0, 1] / np.var(tdf.loc[:5, "SPY 1R"])
            dam = (r12 * wr12) / (v6 * b6) if all(pd.notna([r12, wr12, v6, b6])) and v6 * b6 != 0 else np.nan
            sector = tdf.loc[0, "Sector"]
            if pd.notna(dam) and sector:
                results.append({"Ticker": ticker, "Sector": sector, "DAM": dam})
        except Exception as e:
            print(f"Error calculating DAM for {ticker}: {e}")
            continue

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": {"Error": "No valid DAM results."}
        })

    top_tickers = (
        result_df.sort_values("DAM", ascending=False)
        .groupby("Sector")
        .first()
        .reset_index()
    )

    output = {}
    for _, row in top_tickers.iterrows():
        output[row["Sector"]] = {"top": row["Ticker"]}

    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": output
    })