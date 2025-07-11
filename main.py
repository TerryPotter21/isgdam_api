from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd, numpy as np
import yfinance as yf
from yahooquery import Ticker

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

AUTHORIZED_CODE = "freelunch"

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

@app.api_route("/", methods=["GET", "POST"], response_class=HTMLResponse)
async def login_page(request: Request):
    if request.method == "POST":
        form = await request.form()
        if form.get("password") == AUTHORIZED_CODE:
            return RedirectResponse("/instructions", status_code=303)
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid access code"})
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.get("/instructions", response_class=HTMLResponse)
async def show_instructions(request: Request):
    today = datetime.now()
    cm = today.strftime("%Y-%m")
    sd = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    ed = today.strftime("%Y-%m-%d")
    spy = flatten(yf.download("SPY", start=sd, end=ed, interval="1mo", auto_adjust=True))

    if spy.empty or "Close" not in spy.columns:
        latest = "N/A"
    else:
        spy = spy.reset_index()
        latest = spy["Date"].max().strftime("%Y-%m")

    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "current_month": cm,
        "latest_date": latest,
        "is_current": latest == cm
    })

@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
    tickers = [
        "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN", "WMT", "PEP", "COP", "CVX",
        "JPM", "MS", "JNJ", "ABBV", "UNH", "GE", "CAT", "NVDA", "SO", "DUK",
        "PLD", "O", "LIN", "SHW", "T", "F", "MRK"
    ]
    today = datetime.now()
    sd = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    ed = today.strftime("%Y-%m-%d")

    spy = flatten(yf.download("SPY", start=sd, end=ed, interval="1mo", auto_adjust=True))
    if spy.empty or "Close" not in spy.columns:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": [],
            "weights": [],
            "error": "SPY data missing or incomplete"
        })

    spy = spy.reset_index()
    spy["SPY1R"] = spy["Close"].pct_change().sub(0.024 / 12).fillna(0)
    spy_map = dict(zip(spy["Date"], spy["SPY1R"]))

    dfs = []
    for t in tickers:
        df = flatten(yf.download(t, start=sd, end=ed, interval="1mo", auto_adjust=True))
        if df.empty or "Close" not in df.columns:
            continue
        df = df.reset_index()
        df["Ticker"] = t
        df["Sector"] = yf.Ticker(t).info.get("sector", "N/A")
        df["1R"] = df["Close"].pct_change().sub(0.024 / 12)
        df["SPY1R"] = df["Date"].map(spy_map)
        df = df.dropna(subset=["1R", "SPY1R"])
        if len(df) >= 13 and df["Sector"].iloc[0] != "N/A":
            dfs.append(df)

    if not dfs:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": [],
            "weights": [],
            "error": "No tickers met the minimum criteria"
        })

    df_all = pd.concat(dfs)
    weights = [0.01, 0.01, 0.04, 0.04, 0.09, 0.09, 0.16, 0.16, 0.25, 0.25, 0.36, 0.36]
    results = []

    for t, grp in df_all.groupby("Ticker"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        try:
            r12 = (grp.loc[12, "Close"] - grp.loc[0, "Close"]) / grp.loc[0, "Close"]
            wr12 = sum(grp.loc[i, "1R"] * weights[i] for i in range(12))
            v6 = grp.loc[:5, "1R"].std(ddof=1) * np.sqrt(2)
            cov = np.cov(grp.loc[:5, "1R"], grp.loc[:5, "SPY1R"])[0, 1]
            b6 = cov / np.var(grp.loc[:5, "SPY1R"])
            if v6 == 0 or b6 == 0:
                continue
            dam = (r12 * wr12) / (v6 * b6)
            results.append({"Sector": grp["Sector"].iloc[0], "Ticker": t, "D": dam})
        except Exception:
            continue

    df_result = pd.DataFrame(results).dropna(subset=["D"])
    df_result = df_result.sort_values("D", ascending=False)

    final_rows = []
    for sector, group in df_result.groupby("Sector"):
        top = group.head(2)
        row = {"sector": sector, "ticker": None, "alt_ticker": None}
        if len(top) > 0:
            row["ticker"] = top.iloc[0]["Ticker"]
        if len(top) > 1:
            row["alt_ticker"] = top.iloc[1]["Ticker"]
        final_rows.append(row)

    try:
        spy_info = Ticker("SPY")
        sector_data = spy_info.fund_sector_weightings
        if isinstance(sector_data, dict) and "SPY" in sector_data:
            weight_map = sector_data["SPY"]
        else:
            weight_map = {}
    except Exception:
        weight_map = {}

    weight_table = [{"sector": sec, "weight": f"{wt:.2%}"} for sec, wt in weight_map.items()]
    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": final_rows,
        "weights": weight_table,
        "error": None
    })
