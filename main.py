# main.py  ───────────────────────────────────────────────────────────
import os, time, requests, pandas as pd, pandas_ta as pta
from fastapi import FastAPI, Depends, HTTPException, Header
from jose import jwt
from sklearn.datasets import load_boston      # demo data
import xgboost as xgb, onnx, skl2onnx          # ML section
# -------------------------------------------------------------------
BINANCE = "https://api.binance.com/api/v3/klines"
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")

app = FastAPI(title="Edge-Trade Analytics")

# ───────────── JWT dependency ──────────────────────────────────────
def auth(token: str = Header(..., alias="Authorization")):
    try:
        jwt.decode(token.split()[1], JWT_SECRET, algorithms=["HS256"])
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid JWT")

# ───────────── 1️⃣ OHLCV endpoint ──────────────────────────────────
@app.get("/ohlcv", dependencies=[Depends(auth)])
def ohlcv(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 500):
    res = requests.get(BINANCE, params={
        "symbol": symbol, "interval": interval, "limit": limit
    }).json()                                       # Binance docs :contentReference[oaicite:5]{index=5}
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(res, columns=cols).astype(float)
    return df[["open","high","low","close","volume"]].to_dict("list")

# ───────────── 2️⃣ Order-Block detector (very minimal demo) ────────
@app.get("/orderblocks", dependencies=[Depends(auth)])
def order_blocks(symbol: str = "BTCUSDT", interval: str = "1h"):
    df = pd.DataFrame(ohlcv(symbol, interval)["close"])
    # placeholder: find HL > previous-high swing → mark as bullish block
    highs = df["close"].rolling(3).apply(lambda x: x[1]>x[0] and x[1]>x[2])
    blocks = df[highs==1].tail(3).to_dict("records")
    return {"blocks": blocks}
#  Replace with LuxAlgo’s open-source logic for production. :contentReference[oaicite:6]{index=6}

# ───────────── 3️⃣ Volume-flow / CVD ───────────────────────────────
@app.get("/volume", dependencies=[Depends(auth)])
def volume(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500):
    df = pd.DataFrame(ohlcv(symbol, interval, limit))
    df["cvd"] = pta.cvd(df["close"], df["volume"])
    return df["cvd"].tolist()

# ───────────── 4️⃣ Quant-bias (tiny demo model) ────────────────────
X, y = load_boston(return_X_y=True)          # placeholder dataset
booster = xgb.XGBRegressor().fit(X, y)
booster.save_model("quant.json")             # convert to ONNX later :contentReference[oaicite:7]{index=7}

@app.get("/quantbias", dependencies=[Depends(auth)])
def quant_bias(last_price: float):
    pred = booster.predict([[last_price]*X.shape[1]])[0]
    return {"bias": float(pred)}

# ───────────── 5️⃣ Position sizing helper ─────────────────────────
@app.get("/position_size", dependencies=[Depends(auth)])
def position(balance: float, entry: float, stop: float, risk: float = 1.5):
    risk_usd = balance * risk/100
    size = risk_usd / abs(entry - stop)
    return {"size_btc": round(size, 4)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
