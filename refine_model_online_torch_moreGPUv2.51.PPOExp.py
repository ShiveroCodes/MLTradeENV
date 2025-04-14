#!/usr/bin/env python3
"""
Unhinged Trading Framework v6.0
================================

This integrated monolith now supports:
  • Multi-asset handling (each episode uses one asset’s contiguous data)
  • Per-asset calibration for stop-loss and take-profit using a quick backtest
  • PineScript-inspired indicator computations (pivots, supertrend, chop, volatility)
  • A PPO-based RL agent with an LSTM-based policy network
  • Dynamic (adaptive) strategy selection: no fixed strategy parameters; the agent 
    adapts its behavior via risk_adjustment_factor and fail_safe_threshold.
  • A minimal Tkinter UI showing iteration, current asset, and profit
  • Debug mode and safety options so that if things go awry, logs are dumped
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import glob
import re
import json
import argparse
import random
import threading
import queue
import math
import datetime
import time
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# -----------------------------------------------------------------------------
# GLOBAL LOG QUEUE & CONSOLE BUFFER
# -----------------------------------------------------------------------------
LOG_QUEUE = queue.Queue()
CONSOLE_LOG_BUFFER = []

def log_message(msg):
    """
    Thread-safe logging function that prints to console and also queues the message.
    """
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    LOG_QUEUE.put(full_msg)
    CONSOLE_LOG_BUFFER.append(full_msg)
    print(full_msg)

def clear_logs():
    global CONSOLE_LOG_BUFFER
    CONSOLE_LOG_BUFFER = []

def save_logs_to_file(folder_path):
    """
    Save logs to a file.
    """
    os.makedirs(folder_path, exist_ok=True)
    logfile = os.path.join(folder_path, "console.log")
    with open(logfile, "w") as f:
        f.write("\n".join(CONSOLE_LOG_BUFFER))
    log_message(f"Logs saved to {logfile}")

def save_checkpoint(state, filename):
    """
    Save a PyTorch model checkpoint.
    """
    torch.save({"model_state": state}, filename)
    log_message(f"Checkpoint saved: {filename}")

def load_checkpoint(pattern="unhinged_model_v6_iter*.pt"):
    """
    Load the latest checkpoint matching the given pattern.
    """
    files = glob.glob(pattern)
    if not files:
        log_message("No checkpoint found. Starting fresh.")
        return None
    pattern_re = re.compile(r"unhinged_model_v6_iter(\d+)\.pt")
    iter_files = []
    for f in files:
        m = pattern_re.search(f)
        if m:
            iter_files.append((int(m.group(1)), f))
    if not iter_files:
        log_message("No valid checkpoint found. Starting fresh.")
        return None
    last_iter, file_path = max(iter_files, key=lambda x: x[0])
    log_message(f"Loaded checkpoint from {file_path} (iteration {last_iter})")
    checkpoint = torch.load(file_path)
    return checkpoint["model_state"]

# -----------------------------------------------------------------------------
# TKINTER UI (MINIMAL)
# -----------------------------------------------------------------------------
class UnhingedUI:
    """
    Minimal UI displaying iteration, current asset, and profit. Includes a Stop button.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Unhinged Trading v6.0 UI")
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.stop_button = tk.Button(self.top_frame, text="Stop", command=self.stop_request)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.info_label = tk.Label(self.top_frame, text="Iteration: N/A | Asset: N/A | Profit: N/A")
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.stop_requested = False
        self.root.after(100, self.poll_log_queue)

    def update_info(self, iteration, asset, profit):
        self.info_label.config(text=f"Iteration: {iteration} | Asset: {asset} | Profit: {profit:.2f}%")

    def stop_request(self):
        self.stop_requested = True
        self.stop_button.config(text="Stop Requested", state="disabled")

    def poll_log_queue(self):
        while not LOG_QUEUE.empty():
            msg = LOG_QUEUE.get_nowait()
            # Could display in a text widget if desired.
        self.root.after(100, self.poll_log_queue)

# -----------------------------------------------------------------------------
# DATA LOADER & INDICATORS
# -----------------------------------------------------------------------------
class DataLoader:
    """
    Loads CSV files for given assets and timeframes, merges them, and computes relative changes.
    """
    def __init__(self, assets, timeframes):
        self.assets = assets
        self.timeframes = timeframes

    def load_csv(self, asset, tf):
        filename = f"BITGET_{asset}_{tf}.csv"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"CSV file {filename} not found!")
        df = pd.read_csv(filename)
        df.columns = df.columns.str.lower()
        df["asset"] = asset
        renames = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
        df.rename(columns=renames, inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.sort_values('time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = self.compute_relative_changes(df)
        return df

    def compute_relative_changes(self, df):
        if len(df) > 0:
            first_price = df["close"].iloc[0]
            df["open_pct"]  = ((df["open"]  - first_price) / first_price) * 100
            df["high_pct"]  = ((df["high"]  - first_price) / first_price) * 100
            df["low_pct"]   = ((df["low"]   - first_price) / first_price) * 100
            df["close_pct"] = ((df["close"] - first_price) / first_price) * 100
        return df

    def load_all_csvs(self):
        dfs = []
        for asset in self.assets:
            for tf in self.timeframes:
                try:
                    df = self.load_csv(asset, tf)
                    dfs.append(df)
                    log_message(f"Loaded {len(df)} rows for {asset} TF {tf}")
                except Exception as e:
                    log_message(f"Error loading {asset} TF {tf}: {e}")
        if not dfs:
            raise ValueError("No CSV data loaded.")
        merged = pd.concat(dfs, ignore_index=True)
        merged.sort_values("time", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def compute_indicators(self, df):
        for tf in self.timeframes:
            window = 14
            df[f'atr_{tf}'] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=window, fillna=True)
            df[f'rsi_{tf}'] = ta.momentum.rsi(df["close"], window=window, fillna=True)
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2, fillna=True)
            df[f'bb_upper_{tf}'] = bb.bollinger_hband()
            df[f'bb_lower_{tf}'] = bb.bollinger_lband()
            df[f'bb_mid_{tf}']   = bb.bollinger_mavg()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        log_message(f"Indicators computed; final shape {df.shape}")
        return df

# -----------------------------------------------------------------------------
# PINESCRIPT-INSPIRED SIGNALS & FILTERS
# -----------------------------------------------------------------------------
def detect_pivots(df, left=5, right=5):
    pivots_high = []
    pivots_low = []
    for i in range(left, len(df) - right):
        window_high = df["high"].iloc[i-left:i+right+1]
        window_low  = df["low"].iloc[i-left:i+right+1]
        if df["high"].iloc[i] == window_high.max():
            pivots_high.append(i)
        if df["low"].iloc[i] == window_low.min():
            pivots_low.append(i)
    return pivots_high, pivots_low

def compute_supertrend(df, atr_period=10, multiplier=3.0, change_atr=True):
    if change_atr:
        st_atr = ta.atr(df["high"], df["low"], df["close"], window=atr_period, fillna=True)
    else:
        st_atr = df["high"].rolling(atr_period).mean() - df["low"].rolling(atr_period).mean()
        st_atr.fillna(method="bfill", inplace=True)
    hl2 = (df["high"] + df["low"])/2
    up   = hl2 - multiplier * st_atr
    down = hl2 + multiplier * st_atr
    st_line = [np.nan]*len(df)
    st_trend = [1]*len(df)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            st_line[i] = max(up.iloc[i], st_line[i-1] if not np.isnan(st_line[i-1]) else up.iloc[i])
            st_trend[i] = 1
        else:
            st_line[i] = min(down.iloc[i], st_line[i-1] if not np.isnan(st_line[i-1]) else down.iloc[i])
            st_trend[i] = -1
    return pd.Series(st_line), pd.Series(st_trend)

def chop_filter(df, chop_bars=10, chop_threshold=1.0):
    chop_high = df["high"].rolling(chop_bars).max()
    chop_low  = df["low"].rolling(chop_bars).min()
    chop_range = ((chop_high - chop_low)/chop_low)*100.0
    chop_ok = chop_range > chop_threshold
    return chop_ok.fillna(True)

def volatility_filter(df, threshold=1.0):
    atr_val = ta.atr(df["high"], df["low"], df["close"], window=14, fillna=True)
    bb_width = ta.stdev(df["close"], 20, fillna=True)/ta.sma(df["close"], 20, fillna=True)
    super_vola = (atr_val + bb_width)/2
    return (super_vola > threshold).fillna(True)

def compute_pine_filters(df, config):
    piv_h, piv_l = detect_pivots(df, left=5, right=5)
    pivot_col = np.zeros(len(df))
    pivot_col[piv_h] = 1
    pivot_col[piv_l] = -1
    df["pivot_signal"] = pivot_col
    if config.get("adaptive_strategy_selection", False) is False:
        use_supertrend = config.get("useSupertrend", False)
    else:
        use_supertrend = False  # In adaptive mode, we let the model decide
    if use_supertrend:
        st_line, st_trend = compute_supertrend(df,
            atr_period=config.get("st_ATRPeriod", 10),
            multiplier=config.get("st_Multiplier", 3.0),
            change_atr=config.get("st_changeATR", True))
        df["supertrend_line"] = st_line
        df["supertrend_trend"] = st_trend
    else:
        df["supertrend_line"] = np.nan
        df["supertrend_trend"] = np.nan
    use_chop = config.get("useChopFilter", True)
    if use_chop:
        chop_ok = chop_filter(df, chop_bars=config.get("chopBars", 10), chop_threshold=config.get("chopThreshold", 1.0))
        df["chop_ok"] = chop_ok.astype(int)
    else:
        df["chop_ok"] = 1
    use_vola = config.get("useVolaFilter", True)
    if use_vola:
        vola_ok = volatility_filter(df, threshold=config.get("volaThreshold", 1.0))
        df["vola_ok"] = vola_ok.astype(int)
    else:
        df["vola_ok"] = 1
    return df

# -----------------------------------------------------------------------------
# PER-ASSET CALIBRATION
# -----------------------------------------------------------------------------
class PerAssetCalibrator:
    """
    Calibrates stop-loss (SL) and take-profit (TP) for each asset separately
    via a quick backtest (using a simple EMA crossover signal).
    """
    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.assets = config.get("target_assets", [])
        self.best_params = {}
        for a in self.assets:
            self.best_params[a] = {
                "SL": config.get("fixedStopLossPercent", 1.0),
                "TP": config.get("fixedTakeProfitPercent", 2.0),
                "performance": -9999.0
            }

    def calibrate_all_assets(self, iterations=30):
        log_message("Starting per-asset calibration...")
        for asset in self.assets:
            df_asset = self.df[self.df["asset"] == asset].copy()
            if len(df_asset) < 300:
                log_message(f"Skipping {asset}: insufficient data.")
                continue
            for i in range(iterations):
                sl = random.uniform(0.5, 2.0)
                tp = random.uniform(1.0, 3.0)
                perf = self.backtest(df_asset, sl, tp)
                if perf > self.best_params[asset]["performance"]:
                    self.best_params[asset]["SL"] = sl
                    self.best_params[asset]["TP"] = tp
                    self.best_params[asset]["performance"] = perf
                log_message(f"[{asset}] Calib {i}: SL={sl:.2f}%, TP={tp:.2f}%, Perf={perf:.4f}")
            log_message(f"[{asset}] Best: SL={self.best_params[asset]['SL']:.2f}%, TP={self.best_params[asset]['TP']:.2f}%, Perf={self.best_params[asset]['performance']:.4f}")
        log_message("Per-asset calibration complete.")

    def backtest(self, df_asset, sl, tp):
        initial_capital = self.config.get("starting_capital", 300.0)
        capital = initial_capital
        position = 0
        entry_price = 0.0
        for i in range(200, len(df_asset)):
            fast_ema = df_asset["close"].iloc[i-20:i].ewm(span=20).mean().iloc[-1]
            slow_ema = df_asset["close"].iloc[i-200:i].ewm(span=200).mean().iloc[-1]
            price = df_asset["close"].iloc[i]
            if position == 0:
                if fast_ema > slow_ema:
                    position = 1
                    entry_price = price
                elif fast_ema < slow_ema:
                    position = -1
                    entry_price = price
            else:
                if position == 1:
                    if price >= entry_price*(1+tp/100):
                        capital *= (1+tp/100)
                        position = 0
                    elif price <= entry_price*(1-sl/100):
                        capital *= (1-sl/100)
                        position = 0
                elif position == -1:
                    if price <= entry_price*(1-tp/100):
                        capital *= (1+tp/100)
                        position = 0
                    elif price >= entry_price*(1+sl/100):
                        capital *= (1-sl/100)
                        position = 0
        performance = (capital - initial_capital)/initial_capital
        return performance

# -----------------------------------------------------------------------------
# TRADING POLICY (PPO)
# -----------------------------------------------------------------------------
class TradingPolicy(nn.Module):
    """
    LSTM-based policy network for PPO. Action space: 0=Hold, 1=Enter Long, 2=Enter Short, 3=Close.
    """
    def __init__(self, input_size, hidden_size=512, lstm_layers=2, action_space=4):
        super(TradingPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        logits = self.fc_action(final)
        return logits

class TrajectoryBuffer:
    """
    Buffer to store observations, actions, log_probs, and rewards.
    """
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def store(self, obs, action, log_prob, reward):
        self.obs.append(obs.detach().cpu())
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.rewards.append(reward)

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

def ppo_update(policy, trajectories, device, ppo_epochs=4, epsilon=0.2, gamma=0.99,
               c1=0.5, c2=0.01, lr=0.0002):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    all_obs, all_actions, all_old_log_probs, all_returns = [], [], [], []
    for traj in trajectories:
        R = 0.0
        returns = []
        for r in reversed(traj.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        all_obs.extend(traj.obs)
        all_actions.extend(traj.actions)
        all_old_log_probs.extend(traj.log_probs)
        all_returns.extend(returns_tensor.tolist())
        traj.clear()

    obs_batch = torch.stack(all_obs).to(device)
    actions_batch = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_log_probs_batch = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
    returns_batch = torch.tensor(all_returns, dtype=torch.float32, device=device)

    values_batch = returns_batch.clone()  # using returns as a proxy for value estimates
    advantages = returns_batch - values_batch
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    for _ in range(ppo_epochs):
        logits = policy(obs_batch)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_batch)

        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = c1 * (returns_batch - values_batch).pow(2).mean()
        entropy_bonus = c2 * dist.entropy().mean()

        loss = policy_loss + value_loss - entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    log_message("PPO update complete.")

# -----------------------------------------------------------------------------
# TRADING ENVIRONMENT (PER-ASSET)
# -----------------------------------------------------------------------------
class TradingEnv:
    """
    Environment that, upon reset, picks one asset and uses its contiguous data.
    Uses calibrated SL/TP and optionally applies adaptive adjustments.
    """
    def __init__(self, df, config, per_asset_params):
        self.df_all = df.copy()
        self.config = config
        self.per_asset_params = per_asset_params
        self.assets = config.get("target_assets", [])
        self.window_size = config.get("window_size", 30)
        self.episode_length = config.get("episode_length", 500)
        self.trade_cost = config.get("trade_cost", 0.0012)
        self.starting_capital = config.get("starting_capital", 300.0)
        self.extreme_move_threshold = config.get("extreme_move_threshold", 5000.0)
        self.reset()

    def reset(self):
        self.current_asset = random.choice(self.assets)
        df_asset = self.df_all[self.df_all["asset"] == self.current_asset].copy()
        df_asset.reset_index(drop=True, inplace=True)
        if len(df_asset) < self.episode_length + self.window_size + 2:
            self.current_asset = random.choice(self.assets)
            df_asset = self.df_all[self.df_all["asset"] == self.current_asset].copy()
            df_asset.reset_index(drop=True, inplace=True)
        self.df_current = df_asset
        self.sl = self.per_asset_params[self.current_asset]["SL"]
        self.tp = self.per_asset_params[self.current_asset]["TP"]

        self.start_index = random.randint(self.window_size, len(self.df_current) - self.episode_length - 1)
        self.index = self.start_index
        self.position = 0
        self.entry_price = 0.0
        self.last_price = self.df_current["close"].iloc[self.index-1]
        self.trade_count = 0
        self.accumulated_pct = 0.0
        self.accumulated_dollars = 0.0
        self.trade_events = []
        self.done = False

        log_message(f"[{self.current_asset}] Env reset. SL={self.sl:.2f}%, TP={self.tp:.2f}%")
        return self.get_observation()

    def get_observation(self):
        start = max(0, self.index - self.window_size)
        obs_df = self.df_current.iloc[start:self.index]
        exclude_cols = ["time", "asset"]
        obs_df = obs_df.drop(columns=[c for c in exclude_cols if c in obs_df.columns], errors="ignore")
        obs = obs_df.values.astype("float32")
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), 0.0, self.done, {}

        curr_price = self.df_current["close"].iloc[self.index]
        reward = 0.0
        trade_executed = False

        if self.position == 0:
            if action == 1:
                self._open_trade(curr_price, 1)
                trade_executed = True
            elif action == 2:
                self._open_trade(curr_price, -1)
                trade_executed = True
        else:
            if action == 3:
                reward += self._close_trade(curr_price)
                trade_executed = True
            else:
                delta = (curr_price - self.last_price) / self.last_price * 100.0 * self.position
                if abs(delta) > self.extreme_move_threshold:
                    delta = 0.0
                reward += delta / 100.0

                if self.position == 1:
                    if curr_price >= self.entry_price*(1+self.tp/100):
                        reward += self._close_trade(curr_price)
                        trade_executed = True
                    elif curr_price <= self.entry_price*(1-self.sl/100):
                        reward += self._close_trade(curr_price)
                        trade_executed = True
                elif self.position == -1:
                    if curr_price <= self.entry_price*(1-self.tp/100):
                        reward += self._close_trade(curr_price)
                        trade_executed = True
                    elif curr_price >= self.entry_price*(1+self.sl/100):
                        reward += self._close_trade(curr_price)
                        trade_executed = True

        # Adaptive strategy: adjust reward using risk_adjustment_factor,
        # and if reward magnitude is below fail_safe_threshold, ignore it.
        if self.config.get("adaptive_strategy_selection", False):
            factor = self.config.get("risk_adjustment_factor", 1.0)
            reward *= factor
            if abs(reward) < self.config.get("fail_safe_threshold", 0.1):
                reward = 0.0

        if trade_executed:
            reward += self.config.get("trade_open_reward", 0.005)
        self.last_price = curr_price
        self.index += 1

        if self.index >= self.start_index + self.episode_length or self.index >= len(self.df_current):
            self.done = True
            if self.position != 0:
                reward += self._close_trade(curr_price)

        return self.get_observation(), reward, self.done, {"trade_executed": trade_executed}

    def _open_trade(self, price, position):
        self.position = position
        self.entry_price = price
        self.trade_count += 1
        cost_pct = (self.trade_cost / price) * 100.0 if price != 0 else 0.0
        self.accumulated_pct -= cost_pct
        self.accumulated_dollars -= self.starting_capital * (cost_pct / 100.0)
        event = {
            "type": "open",
            "asset": self.current_asset,
            "price": price,
            "position": position,
            "timestamp": str(self.df_current["time"].iloc[self.index])
        }
        self.trade_events.append(event)
        log_message(f"[{self.current_asset}] Trade opened at {price:.4f}, pos={position}")

    def _close_trade(self, price):
        if self.position == 1:
            trade_pct = ((price - self.entry_price)/self.entry_price)*100.0
        elif self.position == -1:
            trade_pct = ((self.entry_price - price)/self.entry_price)*100.0
        else:
            trade_pct = 0.0

        if abs(trade_pct) > self.extreme_move_threshold:
            trade_pct = 0.0

        cost_pct = (self.trade_cost / price)*100.0 if price != 0 else 0.0
        net_pct = trade_pct - cost_pct
        profit_dollars = self.starting_capital*(net_pct/100.0)
        self.accumulated_pct += net_pct
        self.accumulated_dollars += profit_dollars
        event = {
            "type": "close",
            "asset": self.current_asset,
            "price": price,
            "net_pct": net_pct,
            "profit_dollars": profit_dollars,
            "timestamp": str(self.df_current["time"].iloc[self.index])
        }
        self.trade_events.append(event)
        log_message(f"[{self.current_asset}] Trade closed at {price:.4f}: Net% {net_pct:.2f}, PnL ${profit_dollars:.2f}")
        self.position = 0
        return net_pct / 100.0

# -----------------------------------------------------------------------------
# MAIN FRAMEWORK CLASS
# -----------------------------------------------------------------------------
class UnhingedTradingFramework:
    """
    Orchestrates data loading, indicator computation, per-asset calibration,
    environment creation, and PPO-based RL training. Incorporates adaptive strategy
    selection if enabled.
    """
    def __init__(self, config, root_ui=None):
        self.config = config
        self.root_ui = root_ui
        self.assets = config.get("target_assets", [])
        self.timeframes = config.get("target_timeframes", [])

        self.data_loader = DataLoader(self.assets, self.timeframes)
        df_raw = self.data_loader.load_all_csvs()
        df_raw = self.data_loader.compute_indicators(df_raw)

        df_final = compute_pine_filters(df_raw, config)
        self.df = df_final

        self.per_asset_calibrator = PerAssetCalibrator(self.df, self.config)
        self.per_asset_calibrator.calibrate_all_assets(iterations=config.get("calibration_iterations", 30))
        self.per_asset_params = {}
        for a in self.assets:
            self.per_asset_params[a] = {
                "SL": self.per_asset_calibrator.best_params[a]["SL"],
                "TP": self.per_asset_calibrator.best_params[a]["TP"]
            }
        log_message(f"Per-asset params: {self.per_asset_params}")

        self.env = TradingEnv(self.df, self.config, self.per_asset_params)

        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_policy()

        self.iter_profit_history = []
        self.stop_requested = False
        self.current_iteration = 0

    def init_policy(self):
        num_features = self.env.get_observation().shape[1]
        hidden_size = self.config.get("model_hidden_size", 512)
        lstm_layers = self.config.get("model_lstm_layers", 3)
        self.policy = TradingPolicy(input_size=num_features, hidden_size=hidden_size, lstm_layers=lstm_layers).to(self.device)
        state = load_checkpoint()
        if state:
            self.policy.load_state_dict(state)
            log_message("Loaded policy from checkpoint.")
        else:
            log_message("Initialized new policy from scratch.")

    def run_training_iteration(self, iteration):
        batch_episodes = self.config.get("batch_episodes", 10)
        ppo_epochs = self.config.get("ppo_epochs", 4)
        lr = self.config.get("learning_rate", 0.0002)
        trajectories = []

        for ep in range(batch_episodes):
            traj = TrajectoryBuffer()
            obs = self.env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                new_obs, reward, done, info = self.env.step(action.item())
                traj.store(obs_tensor.squeeze(0), action.item(), log_prob, reward)
                obs = new_obs
            trajectories.append(traj)

        # If adaptive_strategy_selection is enabled, adjust rewards via risk_adjustment_factor.
        if self.config.get("adaptive_strategy_selection", False):
            factor = self.config.get("risk_adjustment_factor", 1.0)
            for traj in trajectories:
                traj.rewards = [r * factor for r in traj.rewards]

        ppo_update(self.policy, trajectories, self.device, ppo_epochs=ppo_epochs, lr=lr)
        iter_profit = self.env.accumulated_pct
        self.iter_profit_history.append(iter_profit)
        if self.root_ui:
            self.root_ui.update_info(iteration, self.env.current_asset, iter_profit)
        log_message(f"[Iter {iteration}] Asset={self.env.current_asset}, Profit={iter_profit:.2f}%")

        if iteration % self.config.get("diskwrite_cycle", 10) == 0:
            ckpt_name = f"unhinged_model_v6_iter{iteration}.pt"
            save_checkpoint(self.policy.state_dict(), ckpt_name)

    def run_training_loop(self):
        max_iters = self.config.get("iterations", 1000)
        start_time = dt.now()
        for it in range(1, max_iters+1):
            if self.root_ui and self.root_ui.stop_requested:
                log_message("Stop requested from UI.")
                break
            elapsed = (dt.now() - start_time).total_seconds() / 60.0
            if self.config.get("max_time", 30) > 0 and elapsed >= self.config["max_time"]:
                log_message("Max time reached. Exiting loop.")
                break
            self.current_iteration = it
            self.run_training_iteration(it)
        log_message("Training loop completed.")

    def visualize_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.iter_profit_history, label="Iteration Profit %")
        plt.xlabel("Iteration")
        plt.ylabel("Profit %")
        plt.title("Profit History Over Training Iterations")
        plt.legend()
        plt.grid(True)
        plt.show()

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unhinged Trading Framework v6.0")
    parser.add_argument("config_file", nargs="?", default="config.json", help="Path to config JSON file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    defaults = {
        "iterations": 1000,
        "batch_episodes": 10,
        "max_time": 30,
        "target_timeframes": [2],
        "target_assets": ["ETHUSDT.P", "BTCUSDT.P", "ADAUSDT.P", "ETCUSDT.P", "SOLUSDT.P"],
        "window_size": 30,
        "episode_length": 500,
        "trade_cost": 0.0012,
        "starting_capital": 300.0,
        "fixedStopLossPercent": 1.0,
        "fixedTakeProfitPercent": 2.0,
        "calibration_iterations": 30,
        "model_hidden_size": 512,
        "model_lstm_layers": 3,
        "ppo_epochs": 4,
        "learning_rate": 0.0002,
        "diskwrite_cycle": 10,
        "extreme_move_threshold": 5000.0,
        "trade_open_reward": 0.005,
        "adaptive_strategy_selection": True,
        "risk_adjustment_factor": 0.75,
        "fail_safe_threshold": 0.1,
        "debug_mode": True,
        "useChopFilter": True,
        "chopBars": 10,
        "chopThreshold": 1.0,
        "useVolaFilter": True,
        "volaThreshold": 1.0,
        "useSupertrend": False
    }
    for k,v in defaults.items():
        config.setdefault(k, v)

    root = tk.Tk()
    ui = UnhingedUI(root)
    framework = UnhingedTradingFramework(config, root_ui=ui)

    def run_training():
        try:
            framework.run_training_loop()
            framework.visualize_results()
        except Exception as e:
            log_message(f"Exception in training loop: {e}")
            if config.get("debug_mode", False):
                folder = dt.now().strftime("logs_%Y%m%d_%H%M%S")
                save_logs_to_file(folder)
            raise e
        folder = dt.now().strftime("logs_%Y%m%d_%H%M%S")
        save_logs_to_file(folder)
        log_message("Exiting main loop.")
        root.quit()

    train_thread = threading.Thread(target=run_training, daemon=True)
    train_thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# END OF Unhinged Trading Framework v6.0
# -----------------------------------------------------------------------------
