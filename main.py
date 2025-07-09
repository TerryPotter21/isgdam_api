from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static folder (if needed in the future)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTHORIZED_CODE = "freelunch"

# Login route: GET shows form, POST handles login logic
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

# Instructions route: checks if SPY has latest monthly data
@app.get("/instructions", response_class=HTMLResponse)
async def show_instructions(request: Request):
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

# Ticker output route (dummy data for now)
@app.post("/tickers", response_class=HTMLResponse)
async def get_tickers(request: Request):
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