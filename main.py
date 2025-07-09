from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS (just in case)
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
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/instructions", response_class=HTMLResponse)
async def show_instructions(request: Request, code: str = Form(...)):
    if code != AUTHORIZED_CODE:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid access code"})

    # Check data freshness using SPY
    today = datetime.now()
    current_month = today.strftime("%Y-%m")
    start_date = (today - relativedelta(months=13)).replace(day=1).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start_date, end=end_date, interval="1mo")
    latest_date = spy.index.max().strftime("%Y-%m") if not spy.empty else "N/A"
    is_current = latest_date == current_month

    return templates.TemplateResponse("instructions.html", {
        "request": request,
        "current_month": current_month,
        "latest_date": latest_date,
        "is_current": is_current
    })

@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
    # TEMP: Use just a few tickers for now
    tickers = {
        "Tech": ["AAPL", "MSFT"],
        "Health": ["ABBV", "JNJ"],
        "Energy": ["XOM", "CVX"]
    }

    result = {}
    for sector, pair in tickers.items():
        result[sector] = {
            "top": pair[0],
            "alt": pair[1]
        }

    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": result
    })