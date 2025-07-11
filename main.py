from fastapi import FastAPI, Request, RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dateutil.relativedelta import relativedelta
from yahooquery import Ticker
import pandas as pd, numpy as np
import yfinance as yf

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
    latest = "N/A"
    if not spy.empty and "Close" in spy.columns:
        latest = spy.reset_index()["Date"].max().strftime("%Y-%m")
    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "current_month": cm,
        "latest_date": latest,
        "is_current": latest == cm
    })

@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
    tickers = ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","PEP","KO","WMT","COST",
               "PFE","JNJ","MRK","ABBV","CVX","XOM","COP","JPM","GS","MS","BAC",
               "NEE","DUK","SO","LMT","GE","BA","CAT","PLD","O","AMT","LIN","APD","SHW"]

    today = datetime.now()
    sd = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    ed = today.strftime("%Y-%m-%d")

    spy = flatten(yf.download("SPY", start=sd, end=ed, interval="1mo", auto_adjust=True))
    if spy.empty or "Close" not in spy.columns:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": [],
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
        df = df.dropna(subset=["1R","SPY1R"])
        if len(df) >= 13 and df["Sector"].iloc[0] != "N/A":
            dfs.append(df)

    if not dfs:
        return templates.TemplateResponse("tickers.html", {
            "request": request,
            "tickers": [],
            "error": "No tickers met the minimum criteria"
        })

    df_all = pd.concat(dfs)
    df_all["Month"] = df_all["Date"].dt.to_period("M")
    weights = [0.01]*2 + [0.04]*2 + [0.09]*2 + [0.16]*2 + [0.25]*2 + [0.36]*2
    results = []
    for t, grp in df_all.groupby("Ticker"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        try:
            r12 = (grp.loc[12,"Close"] - grp.loc[0,"Close"]) / grp.loc[0,"Close"]
            wr12 = sum(grp.loc[i,"1R"]*weights[i] for i in range(12))
            v6 = grp.loc[:5,"1R"].std(ddof=1)*np.sqrt(2)
            cov = np.cov(grp.loc[:5,"1R"], grp.loc[:5,"SPY1R"])[0,1]
            b6 = cov / np.var(grp.loc[:5,"SPY1R"])
            if v6==0 or b6==0:
                continue
            dam = (r12 * wr12)/(v6*b6)
            results.append({"Sector": grp["Sector"].iloc[0], "Ticker": t, "D": dam})
        except:
            continue

    res = pd.DataFrame(results).dropna(subset=["D"])
    top_two = res.sort_values("D",ascending=False).groupby("Sector").head(2).reset_index(drop=True)

    # Fetch SPY weights, handle both formats
    try:
        sw = Ticker("SPY").fund_sector_weightings
        if isinstance(sw, dict) and "SPY" in sw:
            weight_map = sw["SPY"]
        elif hasattr(sw, "columns") and "SPY" in sw.columns:
            weight_map = sw["SPY"].to_dict()
        else:
            print("⚠️ Unexpected SPY fund_sector_weightings structure:", type(sw))
            weight_map = {}
    except Exception as e:
        print("⚠️ Error fetching SPY weights:", e)
        weight_map = {}

    output = []
    for sector, grp in top_two.groupby("Sector"):
        tickers_list = grp.sort_values("D",ascending=False)["Ticker"].tolist()
        output.append({
            "sector": sector,
            "ticker": tickers_list[0] if len(tickers_list)>0 else "N/A",
            "alt_ticker": tickers_list[1] if len(tickers_list)>1 else "N/A",
            "weight": f"{round(weight_map.get(sector, 0)*100,2)}%" if sector in weight_map else "N/A"
        })

    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": output,
        "error": None
    })
