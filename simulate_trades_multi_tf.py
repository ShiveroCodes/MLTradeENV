#!/usr/bin/env python3
"""
Simulation Script for Multi-Timeframe Model

This script:
  1. Loads all CSV files matching:
         BITGET_ETHUSDT.P_*.csv
     Each CSV should have columns: time,open,high,low,close,volume
     and will be renamed so that, for example, the 1-minute file’s columns become
         open_1, high_1, low_1, close_1, volume_1.
  2. Computes technical indicators for each timeframe:
         ATR, RSI, Bollinger Bands (upper, lower, mid)
     which are appended as columns (e.g. atr_1, rsi_1, bb_upper_1, bb_lower_1, bb_mid_1).
  3. Merges all CSVs on the 'time' column (forward-filling missing values) and converts 'time' to datetime.
  4. For each target simulation timeframe (e.g. 1, 2, 5 minutes), randomly selects two days,
     and runs a sliding-window simulation:
       - The model is fed a window (of length --window) of the full merged feature vector.
       - The trade decision uses the target timeframe’s prices (e.g. close_1, open_1 for 1-minute).
       - For each trade, entry is taken at the next candle’s open and exit at its close.
       - Commission (applied on entry and exit) is subtracted, then leverage is applied, and the result is scaled by margin.
  5. Prints daily metrics: number of trades, success rate, total profit (percentage and dollars), and profit factor.

Usage example:
  python simulate_trades_multi_tf.py --commission 0.12 --leverage 1 --margin 300 --window 60
"""

import os, glob, argparse, random
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import ta

# --- Helper Functions ---
def parse_tf_from_filename(filename):
    """Extract timeframe integer from filename, e.g. BITGET_ETHUSDT.P_5.csv -> 5"""
    base = os.path.basename(filename)
    parts = base.split('_')
    last = parts[-1]  # e.g. "5.csv"
    return int(last.split('.')[0])

def load_csv_with_tf(tf_value):
    """
    Load CSV for a given timeframe (e.g. 1, 2, 5, 15, ...) from file BITGET_ETHUSDT.P_{tf_value}.csv.
    Renames columns to include the timeframe suffix.
    Expects CSV columns: time, open, high, low, close, volume
    """
    filename = f"BITGET_ETHUSDT.P_{tf_value}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found!")
    df = pd.read_csv(filename)
    df.columns = df.columns.str.lower()
    rename_dict = {
        'open':   f'open_{tf_value}',
        'high':   f'high_{tf_value}',
        'low':    f'low_{tf_value}',
        'close':  f'close_{tf_value}',
        'volume': f'volume_{tf_value}'
    }
    df.rename(columns=rename_dict, inplace=True)
    df['time'] = df['time'].astype(int)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_all_csvs():
    """Load all CSVs matching BITGET_ETHUSDT.P_*.csv and return a list of DataFrames and a sorted list of timeframes."""
    files = glob.glob("BITGET_ETHUSDT.P_*.csv")
    if not files:
        raise FileNotFoundError("No files matching 'BITGET_ETHUSDT.P_*.csv' found!")
    dfs = []
    tf_values = []
    for f in files:
        tf_val = parse_tf_from_filename(f)
        df = load_csv_with_tf(tf_val)
        dfs.append(df)
        tf_values.append(tf_val)
    # Sort by timeframe ascending
    sorted_indices = np.argsort(tf_values)
    dfs = [dfs[i] for i in sorted_indices]
    tf_values = [tf_values[i] for i in sorted_indices]
    return dfs, tf_values

def merge_timeframes(dfs):
    """Merge a list of DataFrames on 'time' using outer join and forward-fill missing values. Convert time to datetime."""
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='time', how='outer')
    merged_df.sort_values('time', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df['time'] = pd.to_datetime(merged_df['time'], unit='s')
    return merged_df

def compute_indicators(merged_df, tf_values):
    """
    For each timeframe in tf_values, compute:
       - ATR_{tf}, RSI_{tf}, bb_upper_{tf}, bb_lower_{tf}, bb_mid_{tf}
    using the respective high, low, close columns.
    """
    for tf in tf_values:
        high_col = f'high_{tf}'
        low_col  = f'low_{tf}'
        close_col= f'close_{tf}'
        merged_df[f'atr_{tf}'] = ta.volatility.average_true_range(high=merged_df[high_col],
                                                                     low=merged_df[low_col],
                                                                     close=merged_df[close_col],
                                                                     window=14, fillna=True)
        merged_df[f'rsi_{tf}'] = ta.momentum.rsi(close=merged_df[close_col],
                                                  window=14, fillna=True)
        bb = ta.volatility.BollingerBands(close=merged_df[close_col],
                                          window=20, window_dev=2, fillna=True)
        merged_df[f'bb_upper_{tf}'] = bb.bollinger_hband()
        merged_df[f'bb_lower_{tf}'] = bb.bollinger_lband()
        merged_df[f'bb_mid_{tf}']   = bb.bollinger_mavg()
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

# ----------------------------
# Simulation Functions
# ----------------------------
def simulate_day(merged_df_day, model, window_size, commission, leverage, margin, sim_tf):
    """
    Simulate trades over one day using a sliding window.
    - Input window: all numeric features (all columns except 'time' and 'date')
    - Prediction: model.predict(window) produces a predicted price.
    - Trading decision uses target columns for the simulation timeframe (e.g. open_1, close_1).
    """
    trades = []
    target_open = f'open_{sim_tf}'
    target_close = f'close_{sim_tf}'

    if target_open not in merged_df_day.columns or target_close not in merged_df_day.columns:
        raise KeyError(f"Target columns {target_open} or {target_close} not found.")

    feature_cols = [col for col in merged_df_day.columns if col not in ['time', 'date']]
    # Sliding window simulation from index=window_size to len(day)-2 (so that i+1 exists)
    for i in range(window_size, len(merged_df_day) - 1):
        window = merged_df_day.iloc[i-window_size:i][feature_cols].values.astype(np.float32)
        window = np.expand_dims(window, axis=0)
        predicted_price = model.predict(window, verbose=0)[0, 0]
        current_price = merged_df_day.iloc[i][target_close]
        if predicted_price > current_price:
            entry_price = merged_df_day.iloc[i+1][target_open]
            exit_price  = merged_df_day.iloc[i+1][target_close]
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            entry_price = merged_df_day.iloc[i+1][target_open]
            exit_price  = merged_df_day.iloc[i+1][target_close]
            profit_pct = (entry_price - exit_price) / entry_price
        net_trade_pct = (profit_pct - 2 * commission) * leverage
        trades.append(net_trade_pct)
    if trades:
        trades = np.array(trades)
        success_rate = np.mean(trades > 0)
        total_profit_pct = np.sum(trades)
        wins = np.sum(trades[trades > 0])
        losses = np.sum(trades[trades < 0])
        profit_factor = (wins / abs(losses)) if losses < 0 else float('inf')
        total_profit_dollars = margin * total_profit_pct
    else:
        success_rate = 0
        total_profit_pct = 0
        profit_factor = 0
        total_profit_dollars = 0
    return {
        'trades_executed': len(trades),
        'success_rate': success_rate,
        'total_profit_pct': total_profit_pct,
        'profit_factor': profit_factor,
        'total_profit_dollars': total_profit_dollars
    }

def simulate_for_tf(merged_df, model, commission, leverage, margin, window_size, sim_tf):
    """
    For a given simulation timeframe (sim_tf), group the merged DataFrame by date,
    randomly select two days, and simulate trades.
    """
    target_col = f'close_{sim_tf}'
    if target_col not in merged_df.columns:
        print(f"Target column {target_col} not found. Skipping simulation for {sim_tf}-minute.")
        return None
    merged_df['date'] = merged_df['time'].dt.date
    available_dates = merged_df['date'].unique()
    if len(available_dates) < 2:
        print(f"Not enough days for simulation on {sim_tf}-minute chart!")
        return None
    selected_dates = random.sample(list(available_dates), 2)
    results = {}
    for d in selected_dates:
        df_day = merged_df[merged_df['date'] == d].reset_index(drop=True)
        if len(df_day) < window_size + 1:
            continue
        metrics = simulate_day(df_day, model, window_size, commission, leverage, margin, sim_tf)
        results[str(d)] = metrics
    return results

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulate trades using multi_tf_model.h5 on multi-timeframe data")
    parser.add_argument("--commission", type=float, required=True,
                        help="Commission percentage (e.g. 0.12 for 0.12%)")
    parser.add_argument("--leverage", type=float, required=True,
                        help="Leverage (e.g. 1 for 1x)")
    parser.add_argument("--margin", type=float, required=True,
                        help="Margin in dollars (e.g. 300)")
    parser.add_argument("--window", type=int, default=60,
                        help="Sliding window size (default: 60)")
    args = parser.parse_args()

    commission = args.commission / 100.0
    leverage = args.leverage
    margin = args.margin
    window_size = args.window

    # Load all CSV files and get list of DataFrames and their timeframes
    try:
        dfs, tf_values = load_all_csvs()
    except Exception as e:
        print(e)
        return

    # Merge all timeframes
    merged_df = merge_timeframes(dfs)
    print(f"Merged data: {len(merged_df)} rows, columns: {merged_df.columns.tolist()}")

    # Compute technical indicators for each timeframe that was loaded
    merged_df = compute_indicators(merged_df, tf_values)
    print(f"After computing indicators, merged data has {merged_df.shape[1]-1} feature columns (excluding 'time').")

    # Load the trained multi-timeframe model
    print("Loading model 'multi_tf_model.h5'...")
    model = load_model("multi_tf_model.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    print("Model loaded.")

    # For each target simulation timeframe (e.g., 1, 2, 5), simulate trades
    for sim_tf in [1, 2, 5]:
        print(f"\n--- Simulating for {sim_tf}-minute chart ---")
        sim_results = simulate_for_tf(merged_df.copy(), model, commission, leverage, margin, window_size, sim_tf)
        if sim_results is None:
            continue
        for day, metrics in sim_results.items():
            print(f"Date: {day}")
            print(f"  Trades executed:  {metrics['trades_executed']}")
            print(f"  Success rate:     {metrics['success_rate']:.2%}")
            print(f"  Total profit (%): {metrics['total_profit_pct']:.4f}")
            print(f"  Profit factor:    {metrics['profit_factor']:.4f}")
            print(f"  Total profit ($): {metrics['total_profit_dollars']:.2f}")
            print("-" * 40)

if __name__ == "__main__":
    main()
