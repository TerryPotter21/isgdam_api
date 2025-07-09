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
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

AUTHORIZED_CODE = "freelunch"

@app.api_route("/", methods=["GET", "POST"], response_class=HTMLResponse)
async def login_page(request: Request):
    if request.method == "POST":
        form = await request.form()
        if form.get("password") == AUTHORIZED_CODE:
            return RedirectResponse("/instructions", status_code=303)
        else:
            return templates.TemplateResponse("login.html", {"request":request, "error":"Invalid access code"})
    return templates.TemplateResponse("login.html", {"request":request, "error":None})

@app.get("/instructions", response_class=HTMLResponse)
async def show_instructions(request: Request):
    today = datetime.now()
    current_month = today.strftime("%Y-%m")
    start_date = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo", auto_adjust=True, multi_index=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    if "Close" not in spy.columns:
        logging.error("SPY data missing 'Close'; columns: %s", spy.columns)
        latest = "Error"
        is_current = False
    else:
        latest = spy.index.max().strftime("%Y-%m")
        is_current = (latest == current_month)

    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "current_month": current_month,
        "latest_date": latest,
        "is_current": is_current
    })

@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
    tickers = ["AAPL","MSFT","NVDA","JNJ","PFE","XOM"]
    today = datetime.now()
    start_date = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    results = []

    spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo", auto_adjust=True, multi_index=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.droplevel(1)
    if "Close" not in spy:
        logging.error("SPY missing")
        return templates.TemplateResponse("tickers.html", {"request":request, "tickers":{"Error":"Can't fetch SPY"}})
    spy_pr = spy["Close"].pct_change().sub(0.024/12).fillna(0).reset_index(drop=True)

    for t in tickers:
        df = yf.download(t, start=start_date, end=end_date, interval="1mo", auto_adjust=True, multi_index=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if "Close" not in df.columns:
            logging.warning("Skip %s missing Close", t); continue
        info = yf.Ticker(t).info
        sec = info.get("sector")
        if not sec:
            logging.warning("Skip %s missing sector", t); continue

        df = df.reset_index(drop=True)
        df["1R"] = df["Close"].pct_change().sub(0.024/12)
        df["SPY"] = spy_pr
        if len(df) < 13: continue

        r12 = (df.loc[12,"Close"] - df.loc[0,"Close"]) / df.loc[0,"Close"]
        weights = [0.01,0.01,0.04,0.04,0.09,0.09,0.16,0.16,0.25,0.25,0.36,0.36]
        wr12 = sum(df.loc[i,"1R"]*weights[i] for i in range(12))
        v6 = df.loc[:5,"1R"].std(ddof=1)*np.sqrt(2)
        b6 = np.cov(df.loc[:5,"1R"], df.loc[:5,"SPY"])[0,1]/np.var(df.loc[:5,"SPY"])
        dam = (r12*wr12)/(v6*b6) if v6*b6 and not np.isnan(v6*b6) else np.nan

        results.append({"Sector":sec, "Ticker":t, "DAM":dam})

    if not results:
        return templates.TemplateResponse("tickers.html", {"request":request, "tickers":{"Error":"No valid data"}})

    resdf = (pd.DataFrame(results).dropna().sort_values("DAM", ascending=False)
             .groupby("Sector").first().reset_index())

    out = {row.Sector:{"top":row.Ticker} for _,row in resdf.iterrows()}
    return templates.TemplateResponse("tickers.html", {"request":request, "tickers":out})