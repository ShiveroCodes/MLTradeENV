#!/usr/bin/env python3
"""
Multi-Timeframe LSTM Training Script for Trading (PyTorch Version)

This script:
1. Reads all files named BITGET_ETHUSDT.P_*.csv in the current directory.
   (For example: BITGET_ETHUSDT.P_1.csv, BITGET_ETHUSDT.P_5.csv, etc.)
2. Merges them by 'time' (UNIX seconds) using an outer join and forward-filling missing values.
3. Computes technical indicators (ATR, RSI, Bollinger Bands) for each timeframe.
4. Constructs training sequences:
   - Each sequence has length window_size.
   - The target is defined as the percentage return of the 1-minute close over a specified horizon.
     That is, if horizon = 5, then:
       target = (close_1 at time t+window_size+horizon - close_1 at time t+window_size) / (close_1 at time t+window_size)
5. Splits the data into training and testing sets.
6. Trains an LSTM model (using PyTorch) to predict this aggregated return.
7. Evaluates the model on the test set.
8. Saves the trained model as "multi_tf_model.pt".

This aggregated return target should lead to fewer signals and more sensible trade frequency.

Usage:
  python training_script.py --window 60 --horizon 5 --epochs 10 --batch_size 32
"""

import os
import glob
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse

# ----------------------------
# Data Loading and Merging Functions
# ----------------------------

def parse_timeframe_from_filename(filename):
    # Expect filename like BITGET_ETHUSDT.P_5.csv
    part = os.path.basename(filename).split('_')[-1]  # e.g., "5.csv"
    return int(part.replace('.csv',''))

def load_csv_with_timeframe(filename):
    tf_value = parse_timeframe_from_filename(filename)
    df = pd.read_csv(filename)
    df.columns = [col.lower() for col in df.columns]
    rename_dict = {
        'open': f'open_{tf_value}',
        'high': f'high_{tf_value}',
        'low': f'low_{tf_value}',
        'close': f'close_{tf_value}',
        'volume': f'volume_{tf_value}'
    }
    df.rename(columns=rename_dict, inplace=True)
    df['time'] = df['time'].astype(int)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, tf_value

def load_all_csvs():
    files = glob.glob("BITGET_ETHUSDT.P_*.csv")
    if not files:
        raise FileNotFoundError("No CSV files matching BITGET_ETHUSDT.P_*.csv found!")
    dfs, tf_values = [], []
    for f in files:
        df, tf_val = load_csv_with_timeframe(f)
        dfs.append(df)
        tf_values.append(tf_val)
    return dfs, tf_values

def merge_timeframes(dfs):
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='time', how='outer')
    merged_df.sort_values('time', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

# ----------------------------
# Indicator Computation
# ----------------------------

def compute_indicators(merged_df, tf_values):
    for tf in tf_values:
        high_col = f'high_{tf}'
        low_col  = f'low_{tf}'
        close_col = f'close_{tf}'
        merged_df[f'atr_{tf}'] = ta.volatility.average_true_range(
            high=merged_df[high_col],
            low=merged_df[low_col],
            close=merged_df[close_col],
            window=14, fillna=True)
        merged_df[f'rsi_{tf}'] = ta.momentum.rsi(
            close=merged_df[close_col],
            window=14, fillna=True)
        bb = ta.volatility.BollingerBands(
            close=merged_df[close_col],
            window=20, window_dev=2, fillna=True)
        merged_df[f'bb_upper_{tf}'] = bb.bollinger_hband()
        merged_df[f'bb_lower_{tf}'] = bb.bollinger_lband()
        merged_df[f'bb_mid_{tf}'] = bb.bollinger_mavg()
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

# ----------------------------
# Sequence Creation with Horizon
# ----------------------------

def create_sequences(merged_df, window_size=60, horizon=5):
    """
    Constructs training sequences.
    - Features: All columns except 'time'
    - Target: The percentage return over the horizon for the 1-minute close.
    """
    feature_cols = [col for col in merged_df.columns if col != 'time']
    if 'close_1' not in merged_df.columns:
        raise ValueError("Merged DataFrame must contain 'close_1'.")
    data = merged_df[feature_cols].values
    X, y = [], []
    for i in range(len(data) - window_size - horizon):
        X.append(data[i:i+window_size])
        close_start = data[i+window_size, feature_cols.index('close_1')]
        close_future = data[i+window_size+horizon, feature_cols.index('close_1')]
        ret = (close_future - close_start) / close_start
        y.append(ret)
    return np.array(X), np.array(y), feature_cols

# ----------------------------
# PyTorch LSTM Model Definition
# ----------------------------

class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(TradingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last time-step output
        out = self.fc(out)
        return out

# ----------------------------
# Main Training Loop
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Timeframe LSTM Training Script (PyTorch)")
    parser.add_argument("--window", type=int, default=60, help="Sliding window size (default: 60)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in candles (default: 5)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()
    
    # Load CSVs
    dfs, tf_values = load_all_csvs()
    print("Loaded CSVs for timeframes:", tf_values)
    merged_df = merge_timeframes(dfs)
    print("Merged data shape:", merged_df.shape)
    merged_df = compute_indicators(merged_df, tf_values)
    print("After computing indicators, shape:", merged_df.shape)
    
    # Create sequences with the target being the return over the horizon
    X, y, feature_cols = create_sequences(merged_df, window_size=args.window, horizon=args.horizon)
    print("X shape:", X.shape, "y shape:", y.shape)
    
    # Split train/test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_size = X_train.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingLSTM(input_size=input_size).to(device)
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Training Loss: {train_loss:.4f}")
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test MSE Loss: {test_loss:.6f}")
    
    torch.save(model.state_dict(), "multi_tf_model.pt")
    print("Model saved to 'multi_tf_model.pt'.")

if __name__ == "__main__":
    main()