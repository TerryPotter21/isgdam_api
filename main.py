from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

app = FastAPI()

AUTHORIZED_CODES = ["freelunch"]

@app.post("/tickers")
def get_dam_tickers(code: str = Form(...)):
    if code not in AUTHORIZED_CODES:
        return JSONResponse(content={"error": "Invalid access code."}, status_code=401)
    
    # Dummy data for now (replace with real top tickers later)
    top_tickers = {
        "Technology": {"Ticker": "AAPL", "Alt": "MSFT"},
        "Energy": {"Ticker": "XOM", "Alt": "CVX"},
        "Healthcare": {"Ticker": "JNJ", "Alt": "PFE"}
    }
    return {"tickers": top_tickers}
