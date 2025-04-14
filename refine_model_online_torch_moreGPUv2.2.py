#!/usr/bin/env python3
"""
RL Trading Agent 2.2j – Multi-Asset with Fixed Asset Per Worker and Relative Profit Calculation

Key changes (v2.2j):
  • Each worker’s environment picks a random asset once (and then fixes it) so that the worker’s ID and JSON match.
  • The environment uses only the relative percent change (minus open/close costs) to accumulate profit:
      final capital = starting_capital * (1 + accumulated_pct/100)
      profit_dollars = starting_capital * (accumulated_pct/100)
      profit_factor = 1 + accumulated_pct/100
  • Trade events now include "price_percent_diff_entry" and "PnL_absdiff_abs_abspct": [profit_dollars_for_trade, net_trade_pct].
  • Workers with 1 or fewer trades are skipped for global model updates.
  • Scatter plots are saved into the PNG/ subfolder.
"""

import os, glob, re, json, argparse, random, threading, queue, math, datetime
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

################################################################################
# GLOBAL LOG QUEUE & CONSOLE LOG BUFFER
################################################################################

log_queue = queue.Queue()
console_log_buffer = []

def log_message(msg):
    log_queue.put(("log", msg))
    console_log_buffer.append(f"{dt.now().strftime('%H:%M:%S')} - {msg}")

def ui_post_message(msg_type, data):
    log_queue.put((msg_type, data))

################################################################################
# ITERATION LOGGING
################################################################################

def save_iteration_logs(iteration, pnl, config):
    date_str = dt.now().strftime("%d_%m_%y")
    folder_path = os.path.join("logfiles", date_str, f"iter_{iteration}_{pnl:+.2f}%")
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(folder_path, "console.log"), "w") as f:
        f.write("\n".join(console_log_buffer))
    log_message(f"Logs and config saved in {folder_path}")
    console_log_buffer.clear()

################################################################################
# COMPUTE RELATIVE CHANGES (extra columns)
################################################################################

def compute_relative_changes(df):
    if len(df) > 0:
        first_price = df["close"].iloc[0]
        df["open_pct"]  = ((df["open"]  - first_price) / first_price) * 100
        df["high_pct"]  = ((df["high"]  - first_price) / first_price) * 100
        df["low_pct"]   = ((df["low"]   - first_price) / first_price) * 100
        df["close_pct"] = ((df["close"] - first_price) / first_price) * 100
    return df

################################################################################
# DATA LOADING
################################################################################

def load_csv(asset, tf_value):
    filename = f"BITGET_{asset}_{tf_value}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found!")
    df = pd.read_csv(filename)
    df.columns = df.columns.str.lower()
    renames = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    df.rename(columns=renames, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["asset"] = asset
    df = compute_relative_changes(df)
    return df

def load_all_csvs(config):
    assets = config.get("target_assets", ["ETHUSDT.P"])
    tfs = config.get("target_timeframes", [2])
    dfs = []
    for asset in assets:
        for tf in tfs:
            try:
                df = load_csv(asset, tf)
                dfs.append(df)
            except Exception as e:
                log_message(str(e))
    return dfs, tfs

def merge_timeframes(dfs, config):
    merged = pd.concat(dfs, ignore_index=True)
    merged.sort_values("time", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

################################################################################
# INDICATORS
################################################################################

def compute_indicators(merged, tfs):
    for tf in tfs:
        high, low, close = "high", "low", "close"
        merged[f'atr_{tf}'] = ta.volatility.average_true_range(merged[high], merged[low], merged[close], window=14, fillna=True)
        merged[f'rsi_{tf}'] = ta.momentum.rsi(merged[close], window=14, fillna=True)
        bb = ta.volatility.BollingerBands(merged[close], window=20, window_dev=2, fillna=True)
        merged[f'bb_upper_{tf}'] = bb.bollinger_hband()
        merged[f'bb_lower_{tf}'] = bb.bollinger_lband()
        merged[f'bb_mid_{tf}']   = bb.bollinger_mavg()
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

################################################################################
# TKINTER UI
################################################################################

class RefineUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Trading Refinement – Embrace the Chaos")
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        self.stop_button = tk.Button(self.top_frame, text="Stop after current iter", command=self.handle_stop)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.net_profit_canvas = tk.Canvas(self.top_frame, width=400, height=50, bg="white")
        self.net_profit_canvas.pack(side=tk.RIGHT, padx=10, pady=5)
        self.net_profit_history = []
        self.lookback = 10
        self.tree = ttk.Treeview(root, columns=("Trades", "Profit", "Factor", "Profit($)", "Style"), show="headings")
        self.tree.heading("Trades", text="Trades")
        self.tree.heading("Profit", text="Profit %")
        self.tree.heading("Factor", text="Factor")
        self.tree.heading("Profit($)", text="Profit($)")
        self.tree.heading("Style", text="Style")
        self.tree.column("Trades", width=60)
        self.tree.column("Profit", width=80)
        self.tree.column("Factor", width=80)
        self.tree.column("Profit($)", width=80)
        self.tree.column("Style", width=80)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.worker_rows = {}
        self.stop_callback = None
        self.root.after(100, self.poll_queue)

    def set_stop_callback(self, callback):
        self.stop_callback = callback

    def handle_stop(self):
        if self.stop_callback:
            self.stop_callback()
            self.stop_button.config(text="Stop Requested", state="disabled")

    def add_worker(self, worker_id):
        row = self.tree.insert("", tk.END, text=worker_id, values=("0", "0", "0", "0", "N/A"))
        self.worker_rows[worker_id] = row

    def update_sim_result(self, worker_id, result):
        # Only update UI if trades_executed > 1
        if result.get("trades_executed", 0) <= 1:
            log_message(f"Skipping UI update for worker {worker_id} due to insufficient trades.")
            return

        log_message(f"Updating UI for worker {worker_id}: {result}")
        if worker_id not in self.worker_rows:
            self.add_worker(worker_id)
        row = self.worker_rows[worker_id]
        self.tree.item(row, values=(
            result.get("trades_executed", 0),
            f"{result.get('total_profit_pct', 0.0):.4f}",
            f"{result.get('profit_factor', 1.0):.4f}",
            f"{result.get('total_profit_dollars', 0.0):.2f}",
            result.get("predicted_style", "N/A")
        ))

    def update_net_profit_bar(self, history, current):
        if not history:
            return
        min_val = min(history)
        max_val = max(history)
        X = max(abs(min_val), abs(max_val))
        if X == 0:
            X = 10
        self.net_profit_canvas.delete("all")
        width = int(self.net_profit_canvas['width'])
        height = int(self.net_profit_canvas['height'])
        self.net_profit_canvas.create_line(10, height//2, width-10, height//2, fill="black", width=2)
        self.net_profit_canvas.create_text(20, height-10, text=f"-{X:.0f}%", anchor="w", fill="red")
        self.net_profit_canvas.create_text(width-20, height-10, text=f"+{X:.0f}%", anchor="e", fill="green")
        pos = 10 + ((current + X) / (2 * X)) * (width - 20)
        color = "red" if current < 0 else "green" if current > 0 else "yellow"
        self.net_profit_canvas.create_line(pos, height//2 - 10, pos, height//2 + 10, fill=color, width=3)
        self.net_profit_canvas.create_text(pos, height//2 - 15, text=f"{current:+.2f}%", fill=color)

    def poll_queue(self):
        while not log_queue.empty():
            msg_type, data = log_queue.get_nowait()
            if msg_type == "log":
                print(data)
            elif msg_type == "sim_result":
                sim_res_copy = data["result"].copy() if data["result"] else {}
                sim_res_copy.pop("trade_events", None)
                self.update_sim_result(data["worker_id"], sim_res_copy)
            elif msg_type == "new_worker":
                self.add_worker(data)
            elif msg_type == "net_profit_update":
                self.net_profit_history = data.get("history", [])
                current = data.get("current", 0)
                self.lookback = data.get("lookback", self.lookback)
                self.update_net_profit_bar(self.net_profit_history, current)
        self.root.after(100, self.poll_queue)

################################################################################
# TRADING POLICY
################################################################################

class TradingPolicy(nn.Module):
    def __init__(self, input_size, hidden_size=512, lstm_layers=2):
        super(TradingPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        self.fc_style = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 2)
        )

    def forward(self, obs_seq):
        lstm_out, _ = self.lstm(obs_seq)
        final = lstm_out[:, -1, :]
        logits_action = self.fc_action(final)
        logits_style = self.fc_style(final)
        return logits_action, logits_style

################################################################################
# TRAJECTORY BUFFER
################################################################################

class TrajectoryBuffer:
    def __init__(self):
        self.obs = []
        self.log_probs = []
        self.rewards = []

    def store(self, obs_tensor, log_prob, reward):
        self.obs.append(obs_tensor)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def clear(self):
        self.obs = []
        self.log_probs = []
        self.rewards = []

################################################################################
# RL TRADING ENVIRONMENT (with fixed asset per worker and relative profit calculations)
################################################################################

class RLTradingEnv:
    def __init__(self, df_all, window_size=30, close_col='close', trade_cost=0.001,
                 stop_loss_pct=0.005, min_horizon=10, max_horizon=30,
                 penalty_coef=0.001, random_start=False, episode_length=500,
                 trade_open_reward=0.05, profit_trade_multi=1.0, starting_capital=300.0):
        self.df_all = df_all.copy()
        self.window_size = window_size
        self.close_col = close_col
        self.trade_cost = trade_cost
        self.stop_loss_pct = stop_loss_pct
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.penalty_coef = penalty_coef
        self.random_start = random_start
        self.episode_length = episode_length
        self.trade_open_reward = trade_open_reward
        self.profit_trade_multi = profit_trade_multi
        self.starting_capital = starting_capital

        # Accumulators for relative profit (in percent and dollars)
        self.accumulated_pct = 0.0
        self.accumulated_dollars = 0.0

        self.df_current = None
        self.fixed_asset = None  # once chosen, fix the asset for this worker
        self.reset()

    def reset(self):
        # If this is the first reset, pick a random asset and fix it.
        if self.fixed_asset is None:
            unique_assets = self.df_all["asset"].unique()
            chosen_asset = random.choice(unique_assets)
            self.fixed_asset = chosen_asset
        else:
            chosen_asset = self.fixed_asset

        df_asset = self.df_all[self.df_all["asset"] == chosen_asset].copy()
        df_asset.reset_index(drop=True, inplace=True)
        self.df_current = df_asset
        self.asset = chosen_asset.replace(".P", "")

        max_start = len(self.df_current) - self.episode_length - 1
        if max_start < self.window_size:
            self.start_index = self.window_size
        else:
            self.start_index = random.randint(self.window_size, max_start) if self.random_start else self.window_size
        self.index = self.start_index

        self.position = 0
        self.entry_price = 0.0
        self.last_price = self.df_current.loc[self.index - 1, self.close_col]
        self.trade_count = 0
        self.trade_open_index = None
        self.hold_times = []
        self.trade_events = []
        self.done = False
        self.dynamic_horizon = random.randint(self.min_horizon, self.max_horizon)
        self.rollover_ref = self.df_current.loc[self.start_index, self.close_col]

        # Reset accumulators for each new episode.
        self.accumulated_pct = 0.0
        self.accumulated_dollars = 0.0

        return self._get_observation()

    def step(self, action):
        curr_price = self.df_current.loc[self.index, self.close_col]
        reward = 0.0
        trade_executed = False
        forced_exit = False

        if self.position != 0:
            move_pct = ((curr_price - self.entry_price) / self.entry_price) if self.position == 1 else ((self.entry_price - curr_price) / self.entry_price)
            if move_pct <= -self.stop_loss_pct:
                forced_exit = True

        if forced_exit:
            reward += self._close_position(curr_price)
            trade_executed = True
        else:
            if self.position == 0:
                if action == 1:  # open long
                    self._open_trade(curr_price, 1)
                    trade_executed = True
                elif action == 2:  # open short
                    self._open_trade(curr_price, -1)
                    trade_executed = True
            else:
                if action == 3:  # exit
                    reward += self._close_position(curr_price)
                    trade_executed = True
                else:
                    reward += (curr_price - self.last_price)*self.position

        if trade_executed:
            reward += self.trade_open_reward
        self.last_price = curr_price
        self.index += 1

        if self.index >= self.start_index + self.episode_length or self.index >= len(self.df_current):
            self.done = True
            if self.position != 0:
                reward += self._close_position(curr_price)

        return self._get_observation(), reward, self.done, {"trade_executed": trade_executed}

    def _open_trade(self, price, position):
        self.position = position
        self.entry_price = price
        self.trade_open_index = self.index
        self.trade_count += 1

        # Subtract open cost (in relative percentage)
        cost_pct = (self.trade_cost / price)*100 if price != 0 else 0.0
        self.accumulated_pct -= cost_pct
        self.accumulated_dollars -= self.starting_capital * (cost_pct/100)

        event = {
            "type": "open",
            "timestamp": str(self.df_current.loc[self.index, "time"]),
            "price_abs": float(price),
            "price_percent_diff_rollover": ((price - self.rollover_ref)/self.rollover_ref)*100 if self.rollover_ref != 0 else 0.0,
            "price_percent_diff_entry": None,
            "PnL_absdiff_abs_abspct": None,
            "held_for_candles": None,
            "position": position,
            "asset": self.asset
        }
        self.trade_events.append(event)

    def _close_position(self, curr_price):
        if self.position == 1:
            trade_pct = ((curr_price - self.entry_price)/self.entry_price)*100
        elif self.position == -1:
            trade_pct = ((self.entry_price - curr_price)/self.entry_price)*100
        else:
            trade_pct = 0.0

        # Subtract close cost
        cost_pct = (self.trade_cost/curr_price)*100 if curr_price != 0 else 0.0
        net_trade_pct = trade_pct - cost_pct

        self.accumulated_pct += net_trade_pct
        profit_dollars = self.starting_capital * (net_trade_pct/100)
        self.accumulated_dollars += profit_dollars

        held = self.index - self.trade_open_index if self.trade_open_index is not None else None

        event = {
            "type": "close",
            "timestamp": str(self.df_current.loc[self.index, "time"]),
            "price_abs": float(curr_price),
            "price_percent_diff_rollover": ((curr_price - self.rollover_ref)/self.rollover_ref)*100 if self.rollover_ref != 0 else 0.0,
            "price_percent_diff_entry": float(net_trade_pct),
            "PnL_absdiff_abs_abspct": [float(profit_dollars), float(net_trade_pct)],
            "held_for_candles": held,
            "position": self.position,
            "asset": self.asset
        }
        self.trade_events.append(event)
        if self.trade_open_index is not None:
            self.hold_times.append(held)
        self.position = 0
        self.trade_open_index = None
        return 0  # For training, reward from trade is handled via accumulators

    def _get_observation(self):
        start = max(0, self.index - self.window_size)
        obs_df = self.df_current.iloc[start:self.index]
        obs_cols = [c for c in obs_df.columns if c not in ['time','date','asset','open_pct','high_pct','low_pct','close_pct']]
        return obs_df[obs_cols].values.astype("float32")

    def get_avg_hold_time(self):
        if self.hold_times:
            return float(sum(self.hold_times)/len(self.hold_times))
        return 0.0

    def get_final_profit(self):
        return None

################################################################################
# PLOT TRADING STYLES (PNG/ subfolder)
################################################################################

def plot_trading_styles(sim_results, iteration):
    png_folder = "PNG"
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    x_vals, y_vals, colors = [], [], []
    for sim in sim_results:
        if sim:
            x_vals.append(sim.get("avg_hold_time", 0))
            y_vals.append(sim.get("trades_executed", 0))
            style = sim.get("style", "scalper")
            colors.append("blue" if style=="scalper" else "orange")
    plt.figure(figsize=(6,4))
    plt.scatter(x_vals, y_vals, c=colors, alpha=0.7, edgecolors='k')
    plt.xlabel("Average Hold Time")
    plt.ylabel("Trade Count")
    plt.title(f"Trading Styles at Iteration {iteration}")
    plt.grid(True)
    plot_filename = os.path.join(png_folder, f"trading_styles_iter{iteration}.png")
    plt.savefig(plot_filename)
    plt.close()
    log_message(f"Scatter plot saved: {plot_filename}")

################################################################################
# VISUALIZER: Write trade_events.json to proper folder
################################################################################

def call_visualizer(sim_res, iteration, worker_id):
    date_str = dt.now().strftime("%d_%m_%y")
    folder_path = os.path.join("logfiles", date_str, f"iter_{iteration}_worker_{worker_id}")
    os.makedirs(folder_path, exist_ok=True)
    events_path = os.path.join(folder_path, "trade_events.json")
    with open(events_path, "w") as f:
        json.dump(sim_res.get("trade_events", []), f, indent=2)
    log_message(f"Trade events written for worker {worker_id} at iteration {iteration} to {events_path}")

################################################################################
# WORKER FUNCTION
################################################################################

def refine_worker_torch(worker_id, tf_val, config, merged_df, global_state, device="cuda"):
    try:
        ui_post_message("new_worker", worker_id)

        env = RLTradingEnv(
            merged_df,
            window_size=config.get("window_size", 30),
            close_col="close",
            trade_cost=config.get("trade_cost", 0.001),
            stop_loss_pct=config.get("stop_loss_pct", 0.005),
            min_horizon=config.get("min_horizon", 10),
            max_horizon=config.get("max_horizon", 30),
            penalty_coef=config.get("penalty_coef", 0.001),
            random_start=True,
            episode_length=config.get("episode_length", 500),
            trade_open_reward=config.get("trade_open_reward", 0.05),
            profit_trade_multi=config.get("profit_trade_multi", 1.0),
            starting_capital=config.get("starting_capital", 300.0)
        )

        # Use first reset to fix asset for this worker.
        sample_obs = env.reset()
        # Fix asset for all future resets
        fixed_asset = env.asset
        worker_id_with_asset = f"{worker_id}_{fixed_asset}"

        _, num_features = sample_obs.shape
        hidden_size = config.get("model_hidden_size", 512)
        lstm_layers = config.get("model_lstm_layers", 2)
        policy = TradingPolicy(input_size=num_features, hidden_size=hidden_size, lstm_layers=lstm_layers).to(device)

        try:
            policy.load_state_dict(global_state)
        except Exception as ex:
            log_message(f"[{worker_id_with_asset}] Error loading global state: {ex}")
            return {}, None

        optimizer = optim.Adam(policy.parameters(), lr=config.get("learning_rate", 0.001))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
        gamma = 0.99
        entropy_coef = config.get("entropy_coef", 0.05)
        epochs = config["epochs"]
        batch_episodes = config.get("batch_episodes", 10)
        num_batches = epochs // batch_episodes

        for b in range(num_batches):
            total_loss = 0.0
            for _ in range(batch_episodes):
                obs = env.reset()
                done = False
                trajectory = TrajectoryBuffer()
                while not done:
                    obs_seq = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _ = policy(obs_seq)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    new_obs, reward, done, _ = env.step(action.item())
                    trajectory.store(obs_seq.squeeze(0), log_prob, reward)
                    obs = new_obs
                returns = []
                G = 0.0
                for r in reversed(trajectory.rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32, device=device)
                if returns.std() > 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                log_probs_tensor = torch.stack(trajectory.log_probs).squeeze()
                entropies = []
                for obs_t in trajectory.obs:
                    logits_t, _ = policy(obs_t.unsqueeze(0))
                    p = torch.softmax(logits_t, dim=-1)
                    d = torch.distributions.Categorical(p)
                    entropies.append(d.entropy())
                entropy = torch.mean(torch.stack(entropies))
                loss = - (log_probs_tensor * returns).sum() - entropy_coef * entropy
                total_loss += loss
                trajectory.clear()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        # Final evaluation phase
        obs = env.reset()
        done = False
        style_predictions = []
        total_reward = 0.0
        while not done:
            obs_seq = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, style_logits = policy(obs_seq)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
                predicted_style = torch.argmax(style_logits, dim=-1).item()
                style_predictions.append(predicted_style)
            new_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = new_obs

        style_label = "scalper" if (sum(1 for p in style_predictions if p==0) > len(style_predictions)/2) else "swing"
        # Final sim result uses accumulated values:
        factor = 1.0 + (env.accumulated_pct/100)
        sim_res = {
            "trades_executed": int(env.trade_count),
            "avg_hold_time": float(env.get_avg_hold_time()),
            "total_profit_pct": float(env.accumulated_pct),
            "profit_factor": float(factor),
            "total_profit_dollars": float(env.starting_capital * (factor - 1)),
            "predicted_style": style_label,
            "trade_events": env.trade_events
        }

        ui_post_message("sim_result", {"worker_id": worker_id_with_asset, "result": sim_res})
        log_message(f"[{worker_id_with_asset}] Simulation complete. style={style_label}, Profit($)={env.starting_capital * (factor - 1):.2f}, Profit(%)={env.accumulated_pct:.2f}, factor={factor:.2f}")
        call_visualizer(sim_res, config.get("current_iteration", 0), worker_id_with_asset)
        return policy.state_dict(), sim_res

    except Exception as e:
        log_message(f"[{worker_id}] Exception in worker: {e}")
        return {}, None

################################################################################
# CHECKPOINT LOADER
################################################################################

def load_latest_checkpoint_torch(base_filename="multi_tf_model.pt", prefix="multi_tf_model_refined_iter", ext=".pt"):
    files = glob.glob(f"{prefix}*{ext}")
    if not files:
        log_message("No refined checkpoint found. Starting from base model (new initialization).")
        return None
    pattern = re.compile(f"{prefix}(\\d+){ext}")
    iters = []
    for f in files:
        m = pattern.search(f)
        if m:
            iters.append((int(m.group(1)), f))
    if not iters:
        log_message("No valid checkpoint found. Starting from base model.")
        return None
    best_iter, best_file = max(iters, key=lambda x: x[0])
    log_message(f"Found checkpoint {best_file} (iteration {best_iter}). Loading this model.")
    ckpt = torch.load(best_file)
    return ckpt["model_state"]

################################################################################
# MAIN REFINEMENT APP
################################################################################

class RefineApp:
    def __init__(self, ui, config, merged_df, device):
        self.ui = ui
        self.config = config
        self.merged_df = merged_df
        self.device = device
        self.global_policy = None
        self.global_state = None
        self.start_time = dt.now()
        self.iter_profit_history = []
        self.stop_requested = False

        self.initial_entropy = config.get("entropy_coef", 0.03)
        self.entropy_decrease_pct = config.get("entropy_decrease_pct", 1)

        self.init_global_policy()
        self.refine_thread = threading.Thread(target=self.refinement_loop, daemon=True)
        self.refine_thread.start()

    def request_stop(self):
        self.stop_requested = True

    def init_global_policy(self):
        num_features = len([c for c in self.merged_df.columns if c not in ['time','date','asset','open_pct','high_pct','low_pct','close_pct']])
        hidden_size = self.config.get("model_hidden_size", 512)
        lstm_layers = self.config.get("model_lstm_layers", 2)
        self.global_policy = TradingPolicy(input_size=num_features, hidden_size=hidden_size, lstm_layers=lstm_layers).to(self.device)
        refined_ckpt = load_latest_checkpoint_torch()
        if refined_ckpt is not None:
            try:
                self.global_policy.load_state_dict(refined_ckpt)
                log_message("Global model loaded from refined checkpoint.")
            except Exception as ex:
                log_message(f"Failed to load refined checkpoint: {ex}. Using base model if available.")
                refined_ckpt = None
        if refined_ckpt is None and os.path.exists("multi_tf_model.pt"):
            try:
                base_ckpt = torch.load("multi_tf_model.pt")
                self.global_policy.load_state_dict(base_ckpt)
                log_message("Loaded base model from multi_tf_model.pt")
            except Exception as ex:
                log_message(f"Failed to load base model: {ex}. Using new global model.")
        elif refined_ckpt is None:
            log_message("No refined checkpoint found; using new global model.")
        self.global_state = self.global_policy.state_dict()

    def refinement_loop(self):
        pattern = re.compile(r"multi_tf_model_refined_iter(\d+)\.pt")
        files = glob.glob("multi_tf_model_refined_iter*.pt")
        start_iter = 1
        if files:
            iter_numbers = [int(pattern.search(f).group(1)) for f in files if pattern.search(f)]
            if iter_numbers:
                last_iter = max(iter_numbers)
                start_iter = last_iter + 1
                log_message(f"Resuming training from iteration {start_iter}.")
        iterations = self.config["iterations"]
        worker_multiplier = self.config.get("workersperasset", 2)
        tfs = self.config.get("target_timeframes", [2])
        discard_iter = self.config.get("discard_losers_after_iter", 150)
        lookback = self.config.get("lookback_iterations", 10)

        for it in range(start_iter, iterations+1):
            self.config["current_iteration"] = it
            elapsed = (dt.now() - self.start_time).total_seconds()/60.0
            if self.config["max_time"]>0 and elapsed>=self.config["max_time"]:
                log_message("Max training time reached. Exiting loop.")
                break

            decay_factor = (it/iterations)**2
            current_entropy = self.initial_entropy*(1 - (self.entropy_decrease_pct/100)*decay_factor)
            self.config["entropy_coef"] = current_entropy

            log_message(f"\n--- Global Iteration {it}/{iterations} | Entropy Coef: {current_entropy:.5f} ---")

            worker_futures = []
            total_workers = len(tfs) * worker_multiplier

            with ThreadPoolExecutor(max_workers=total_workers) as executor:
                for tf_val in tfs:
                    for w in range(worker_multiplier):
                        worker_id = f"tf{tf_val}_w{w}"
                        worker_futures.append(
                            executor.submit(
                                refine_worker_torch,
                                worker_id,
                                tf_val,
                                self.config,
                                self.merged_df.copy(),
                                self.global_state,
                                self.device
                            )
                        )
                worker_states = []
                sim_results = []
                for future in as_completed(worker_futures):
                    try:
                        st_dict, sim_res = future.result()
                        if st_dict:
                            worker_states.append(st_dict)
                        sim_results.append(sim_res)
                        sim_res_print = sim_res.copy() if sim_res else {}
                        sim_res_print.pop("trade_events", None)
                        log_message(f"Worker simulation results: {sim_res_print}")
                    except Exception as ex:
                        log_message(f"Worker error: {ex}")

            # Skip workers with <= 1 trade.
            filtered_ws = []
            filtered_sims = []
            for st, sim in zip(worker_states, sim_results):
                if sim and sim.get("trades_executed", 0) > 1:
                    filtered_ws.append(st)
                    filtered_sims.append(sim)
            worker_states = filtered_ws
            sim_results = filtered_sims

            # set style based on avg_hold_time threshold
            for sim in sim_results:
                if sim:
                    threshold = self.config.get("scalper_threshold", 7)
                    sim["style"] = "scalper" if sim.get("avg_hold_time", 0) < threshold else "swing"

            plot_trading_styles(sim_results, it)

            all_profits = [sim.get("total_profit_pct", 0.0) for sim in sim_results if sim]
            if it > discard_iter:
                good = [(st, sim) for st, sim in zip(worker_states, sim_results) if sim and sim.get("total_profit_pct", 0.0) > 0]
                if not good:
                    log_message(f"No profitable workers at iteration {it}. Skipping update.")
                    continue
                worker_states, sims_filtered = zip(*good)
                profits = [sim.get("total_profit_pct", 0.0) for sim in sims_filtered]
                weights = [1.0/len(profits)]*len(profits)
            else:
                profits = all_profits
                P = sum(1 for p in profits if p>0)
                L = len(profits)-P
                N = len(profits)
                if N==0:
                    log_message("No workers contributed. Skipping update.")
                    continue
                if P==0:
                    weights = [1/N]*N
                else:
                    weights = []
                    for sim in sim_results:
                        if sim:
                            profit = sim.get("total_profit_pct", 0.0)
                            if profit > 0:
                                base_weight = (100 - (10*L)) / P
                            else:
                                base_weight = 10
                            style = sim.get("style", "scalper")
                            multiplier = 1.2 if style=="scalper" and profit>0 else 1.0
                            weights.append(base_weight*multiplier)
                    total_weight = sum(weights)
                    weights = [w/total_weight for w in weights]

            new_state = {}
            for key in self.global_state.keys():
                weighted_tensors = [w * st[key] for st, w in zip(worker_states, weights)]
                new_state[key] = sum(weighted_tensors)
            self.global_state = new_state
            self.global_policy.load_state_dict(self.global_state)
            log_message(f"Global policy updated for iteration {it}.")
            ckpt_name = f"multi_tf_model_refined_iter{it}.pt"
            torch.save({"model_state": self.global_state}, ckpt_name)
            log_message(f"Saved checkpoint: {ckpt_name}")

            if len(all_profits) > 0:
                iter_profit = sum(all_profits)/len(all_profits)
            else:
                iter_profit = 0.0
            self.iter_profit_history.append(iter_profit)
            ui_post_message("net_profit_update", {
                "history": self.iter_profit_history[-lookback:],
                "current": iter_profit,
                "lookback": lookback
            })

            scalper_avg = np.mean([sim["total_profit_pct"] for sim in sim_results if sim and sim.get("style")=="scalper"] or [0])
            swing_avg = np.mean([sim["total_profit_pct"] for sim in sim_results if sim and sim.get("style")=="swing"] or [0])
            log_message(f"Iteration {it} style summary: Scalper Avg Profit: {scalper_avg:.4f}, Swing Avg Profit: {swing_avg:.4f}")

            if it % 100 == 0:
                last_100_avg = np.mean(self.iter_profit_history[-100:]) if len(self.iter_profit_history)>=100 else iter_profit
                save_iteration_logs(it, last_100_avg*100, self.config)

            if self.stop_requested:
                log_message("Stop requested. Exiting refinement loop after current iteration.")
                break

        log_message("Refinement loop completed. The chaos subsides...")

################################################################################
# MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(description="RL Trading with Multi-Asset, Net Profit via Relative %s, and Logging")
    parser.add_argument("config_file", nargs="?", default="config.json", help="Path to JSON config file (default: config.json)")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    config.setdefault("starting_capital", 300.0)
    config.setdefault("entropy_decrease_pct", 1)
    config.setdefault("scalper_threshold", 7)
    config.setdefault("workersperasset", 2)

    print("Starting UI... Prepare for the maelstrom.")
    root = tk.Tk()
    ui = RefineUI(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    try:
        dfs, tfs = load_all_csvs(config)
    except Exception as e:
        log_message(str(e))
        input("Press Enter to exit...")
        return

    merged_df = merge_timeframes(dfs, config)
    log_message(f"Merged data: {merged_df.shape[0]} rows, columns: {merged_df.columns.tolist()}")

    merged_df = compute_indicators(merged_df, tfs)
    log_message(f"After computing indicators: {merged_df.shape}")
    merged_df['date'] = merged_df['time'].dt.date

    app = RefineApp(ui, config, merged_df, device)
    ui.set_stop_callback(app.request_stop)
    root.mainloop()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
