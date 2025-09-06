# TrendSpotter-Pro

from fastapi import FastAPI, WebSocket
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import asyncio

# Import core logic from your MVP
from market_whisperer_mvp import (
    simulate_ohlcv,
    simulate_l1,
    compute_fluidity,
    triple_barrier_labels,
    train_model
)

app = FastAPI(title="📈 Market Whisperer API", version="0.2")


# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "Welcome to Market Whisperer API 🚀"}


# ---------- Predict ----------
@app.get("/predict")
def predict(n: int = 500):
    """
    Simulate market data, run model training, and return predictions.
    n = number of OHLCV datapoints to simulate
    """
    ohlcv = simulate_ohlcv(n)
    l1 = simulate_l1(len(ohlcv))

    # Feature engineering
    ohlcv["spread"] = l1["ask"] - l1["bid"]
    ohlcv["imbalance"] = (l1["size_bid"] - l1["size_ask"]) / (l1["size_bid"] + l1["size_ask"] + 1e-6)
    ohlcv["volatility"] = ohlcv["close"].pct_change().rolling(20).std().fillna(0)
    ohlcv["fluidity"] = compute_fluidity(ohlcv, l1)
    ohlcv["label"] = triple_barrier_labels(ohlcv["close"].values)

    # Train + predict
    model = train_model(ohlcv)
    features = ["fluidity", "spread", "imbalance", "volatility"]
    ohlcv["signal"] = model.predict(ohlcv[features])

    return ohlcv[["time", "close", "signal"]].tail(50).to_dict(orient="records")


# ---------- Backtest ----------
@app.get("/backtest")
def backtest(n: int = 1000, stop_loss: float = 0.01, take_profit: float = 0.02):
    """
    Run a simple backtest: enter at signal==1, exit on -1 or TP/SL hit.
    Returns stats: PnL, Sharpe, accuracy, trades.
    """
    ohlcv = simulate_ohlcv(n)
    l1 = simulate_l1(len(ohlcv))

    # Features
    ohlcv["spread"] = l1["ask"] - l1["bid"]
    ohlcv["imbalance"] = (l1["size_bid"] - l1["size_ask"]) / (l1["size_bid"] + l1["size_ask"] + 1e-6)
    ohlcv["volatility"] = ohlcv["close"].pct_change().rolling(20).std().fillna(0)
    ohlcv["fluidity"] = compute_fluidity(ohlcv, l1)
    ohlcv["label"] = triple_barrier_labels(ohlcv["close"].values)

    # Train + predict
    model = train_model(ohlcv)
    features = ["fluidity", "spread", "imbalance", "volatility"]
    ohlcv["signal"] = model.predict(ohlcv[features])

    # Naive backtest
    pnl = []
    position = None
    entry_price = None
    for i, row in ohlcv.iterrows():
        if position is None and row["signal"] == 1:
            position = "long"
            entry_price = row["close"]
        elif position == "long":
            change = (row["close"] - entry_price) / entry_price
            if change <= -stop_loss or change >= take_profit or row["signal"] == -1:
                pnl.append(change)
                position, entry_price = None, None

    stats = {
        "trades": len(pnl),
        "win_rate": np.mean([1 if x > 0 else 0 for x in pnl]) if pnl else 0,
        "avg_return": np.mean(pnl) if pnl else 0,
        "sharpe": (np.mean(pnl) / (np.std(pnl) + 1e-6)) * np.sqrt(252) if pnl else 0,
        "accuracy": accuracy_score(ohlcv["label"], ohlcv["signal"]),
    }
    return stats


# ---------- Websocket Stream ----------
@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        ohlcv = simulate_ohlcv(1)
        l1 = simulate_l1(1)

        # Features
        spread = l1["ask"].iloc[0] - l1["bid"].iloc[0]
        imbalance = (l1["size_bid"].iloc[0] - l1["size_ask"].iloc[0]) / (
            l1["size_bid"].iloc[0] + l1["size_ask"].iloc[0] + 1e-6
        )
        volatility = 0.0  # placeholder
        fluidity = compute_fluidity(ohlcv, l1)

        # Fake model rule for stream (demo only)
        signal = 1 if fluidity > 0.5 else -1

        await websocket.send_json({
            "time": str(ohlcv["time"].iloc[0]),
            "price": float(ohlcv["close"].iloc[0]),
            "signal": int(signal)
        })
        await asyncio.sleep(1)



