import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

cols = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
]

btc = pd.read_parquet("btcusdt_yearly/BTCUSDT_1m_2017.parquet")
eth = pd.read_parquet("ethusdt_yearly/ETHUSDT_1m_2017.parquet")

btc.columns = cols
eth.columns = cols

btc["time"] = pd.to_datetime(btc["open_time"], unit="ms")
eth["time"] = pd.to_datetime(eth["open_time"], unit="ms")

btc = btc.set_index("time").add_prefix("btc_")
eth = eth.set_index("time").add_prefix("eth_")

df = btc.join(eth, how="inner")

df["btc_log_ret"] = np.log(df["btc_close"]).diff()
df["eth_log_ret"] = np.log(df["eth_close"]).diff()

df["btc_ret"] = df["btc_close"].pct_change()
df["eth_ret"] = df["eth_close"].pct_change()


df["btc_vol_1h"] = df["btc_log_ret"].rolling(60).std()
df["btc_vol_4h"] = df["btc_log_ret"].rolling(240).std()

df["eth_vol_1h"] = df["eth_log_ret"].rolling(60).std()
df["eth_vol_4h"] = df["eth_log_ret"].rolling(240).std()


df["btc_taker_ratio"] = df["btc_taker_buy_base_vol"] / df["btc_volume"]
df["eth_taker_ratio"] = df["eth_taker_buy_base_vol"] / df["eth_volume"]

df["btc_range_pct"] = (df["btc_high"] - df["btc_low"]) / df["btc_close"]
df["eth_range_pct"] = (df["eth_high"] - df["eth_low"]) / df["eth_close"]

df["btc_quote_to_base"] = df["btc_quote_asset_volume"] / df["btc_volume"]
df["eth_quote_to_base"] = df["eth_quote_asset_volume"] / df["eth_volume"]


roll = 60

cov = df["eth_log_ret"].rolling(roll).cov(df["btc_log_ret"])
var = df["btc_log_ret"].rolling(roll).var()

df["beta"] = cov / var

df["spread"] = df["eth_log_ret"] - df["beta"] * df["btc_log_ret"]


lags = [1, 2, 3]

for k in lags:
    df[f"btc_leads_eth_{k}m"] = df["btc_log_ret"].shift(k)
    df[f"eth_leads_btc_{k}m"] = df["eth_log_ret"].shift(k)


leak_cols = [
    "btc_range_pct", "eth_range_pct",
    "btc_taker_ratio", "eth_taker_ratio",
    "btc_quote_to_base", "eth_quote_to_base"
]

df[leak_cols] = df[leak_cols].shift(1)
df = df.dropna()

# assume `df` already contains the merged, prefixed, shifted, and cleaned features from prior pipeline
# required feature columns must exist; replace/extend features list as needed
features = [
    "btc_log_ret", "btc_vol_1h", "btc_vol_4h", "btc_taker_ratio", "btc_range_pct", "btc_quote_to_base",
    "eth_log_ret", "eth_vol_1h", "eth_vol_4h", "eth_taker_ratio", "eth_range_pct", "eth_quote_to_base",
    "spread", "btc_leads_eth_1m", "btc_leads_eth_2m", "btc_leads_eth_3m", "eth_leads_btc_1m", "eth_leads_btc_2m", "eth_leads_btc_3m"
]

# target: future 60-minute cumulative simple return of ETH (non-leaky), bucketed into terciles -> 3 regimes
horizon = 60
df["eth_future_ret_60m"] = df["eth_close"].pct_change(periods=horizon).shift(-horizon)  # shift -horizon makes it a future target
df = df.dropna(subset=features + ["eth_future_ret_60m"])

# tercile cut: 0 = bear (bottom tercile), 1 = sideways (middle), 2 = bull (top)
df["regime"] = pd.qcut(df["eth_future_ret_60m"], q=3, labels=[0, 1, 2]).astype(int)

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RegimeNN(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, max_epochs=100, lr=1e-3):

    X = torch.tensor(X_train.values, dtype=torch.float32)
    y = torch.tensor(y_train.values, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for _ in range(max_epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    return model

def predict(model, X):
    X = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        return torch.softmax(logits, dim=1).numpy()

def train_torch(model, train_loader, val_loader, device, max_epochs=100, lr=1e-3, early_stop_patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_state = None
    best_val = np.inf
    patience = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    model.load_state_dict(best_state)
    return model

def walk_forward_training(df, features, target, input_dim,
                          train_size=2000, test_size=500):

    model = RegimeNN(input_dim)                  # persistent model
    model_states = []                            # store weights per window
    predictions = []                             # store window predictions
    window_ranges = []

    start = 0
    end = len(df)

    while start + train_size + test_size <= end:

        train_slice = slice(start, start + train_size)
        test_slice  = slice(start + train_size,
                            start + train_size + test_size)

        X_train = df.iloc[train_slice][features]
        y_train = df.iloc[train_slice][target]

        X_test = df.iloc[test_slice][features]
        y_test = df.iloc[test_slice][target]

        # Standardize with training stats only
        mean = X_train.mean()
        std = X_train.std().replace(0, 1)
        X_train_s = (X_train - mean) / std
        X_test_s = (X_test - mean) / std

        # Train model (incremental)
        model = train_model(model, X_train_s, y_train)

        # Save weights
        model_states.append(model.state_dict())

        # Predict
        probs = predict(model, X_test_s)
        preds = probs.argmax(axis=1)

        predictions.append((preds, y_test.values))
        window_ranges.append((start, start + train_size + test_size))

        start += test_size

    return model_states, predictions, window_ranges

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_train = 2000       # number of rows used for fitting the model in each fold
window_test = 500         # out-of-sample test block length
val_frac = 0.2            # fraction of the training slice reserved for validation
batch_size = 256

results = []
capital = 100_000.0
cap_history = [capital]
trade_log = []

for start in range(0, len(df) - window_train - window_test + 1, window_test):
    train_idx = range(start, start + window_train)
    test_idx = range(start + window_train, start + window_train + window_test)

    X_train = df.iloc[train_idx][features].values
    y_train = df.iloc[train_idx]["regime"].values
    X_test = df.iloc[test_idx][features].values
    y_test = df.iloc[test_idx]["regime"].values

    # split train -> train/val (preserve time order: take last val_frac of training slice as validation to avoid leakage)
    split_point = int(len(X_train) * (1 - val_frac))
    X_tr, X_val = X_train[:split_point], X_train[split_point:]
    y_tr, y_val = y_train[:split_point], y_train[split_point:]

    # scaling: mean/std on X_tr only
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0).clip(min=1e-8)
    X_tr_s = (X_tr - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std

    # datasets and loaders
    train_ds = TabularDataset(X_tr_s, y_tr)
    val_ds = TabularDataset(X_val_s, y_val)
    test_ds = TabularDataset(X_test_s, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # model training
    model = RegimeNN(input_dim=X_tr_s.shape[1], output_dim=3).to(device)
    model = train_torch(model, train_loader, val_loader, device, max_epochs=100, lr=1e-3, early_stop_patience=12)

    # predictions on test set
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    probs = np.vstack(probs)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0,1,2])

    # basic, robust position rule: long 1 (bull), short -1 (bear), 0 for sideways
    # use predicted regime and confidence threshold
    conf = probs.max(axis=1)
    threshold = 0.6
    positions = np.zeros_like(preds, dtype=float)
    positions[(preds == 2) & (conf >= threshold)] = 1.0
    positions[(preds == 0) & (conf >= threshold)] = -1.0

    # position sizing: ETH volatility-based scaling using eth_vol_1h in test slice (use median of training window's vol as base)
    test_vol = df.iloc[test_idx]["eth_vol_1h"].values
    base_vol = np.median(df.iloc[train_idx]["eth_vol_1h"].dropna().values)
    vol_scale = np.clip(base_vol / (test_vol + 1e-9), 0.1, 3.0)  # avoid extreme scaling
    notional_pct = 0.02  # base 2% of capital
    position_sizes = notional_pct * vol_scale  # fraction of capital per position

    # compute next-1m simple returns for ETH as realized pnl driver (non-leaky)
    next_ret = df.iloc[test_idx]["eth_ret"].shift(-0).values  # eth_ret is the return from t-1 to t; alignment chosen earlier avoids leakage

    # iterate test period and update capital
    for i in range(len(preds)):
        p = positions[i]
        size = position_sizes[i]
        ret = next_ret[i]
        strat_pnl = p * ret * size * capital
        capital += strat_pnl
        trade_log.append({
            "timestamp": df.iloc[test_idx].index[i],
            "predicted": int(preds[i]),
            "confidence": float(conf[i]),
            "position": float(p),
            "position_size_pct": float(size),
            "market_return": float(ret),
            "pnl": float(strat_pnl),
            "capital": float(capital)
        })

    cap_history.append(capital)
    results.append({
        "window": f"{start}:{start+window_train+window_test}",
        "accuracy": acc,
        "confusion_matrix": cm
    })

    # optional early stop of whole backtest if catastrophic drawdown
    current_drawdown = (max(cap_history) - capital) / max(cap_history)
    if current_drawdown > 0.25:
        break

res_df = pd.DataFrame(results)
res_df.to_csv("wf_model_results_summary.csv", index=False)

log_df = pd.DataFrame(trade_log)
log_df.to_csv("wf_trade_log.csv", index=False)

# basic diagnostics printed (confusion matrices per window)
for r in results:
    print(r["window"], "acc:", r["accuracy"])
    print(r["confusion_matrix"])


def plot_equity_curve(capital_history):
    plt.figure(figsize=(12, 5))
    plt.plot(capital_history, label="Capital")
    plt.title("Capital Over Time")
    plt.xlabel("Walk-forward Window")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_equity_curve(cap_history)
