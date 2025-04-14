#!/usr/bin/env python3
"""
RL Trading Agent 2.1 – The Chaotic Convergence, Logged & Unhedged

Key metamorphoses (v2.1):
  • Entropy ebbs in a parabolic decay, governed by the config’s entropy_decrease_pct parameter.
  • The agent now has two heads: one for trading actions (with 4 outputs: hold, open long, open short, exit)
    and one for classifying its trading style.
  • A scatter plot (avg_hold_time vs. trade_count) is saved each iteration to reveal clusters.
  • Every 100 iterations, a logging folder is created:
       logfiles/<dd_mm_yy>/iter_<iter>_<PnL%>/ 
    inside which the current config (config.json) and a console log (console.log) are saved.
  • The trading environment now enforces one trade at a time. When no trade is active, the agent may open a long (action 1)
    or short (action 2). If a trade is active, the only valid non-forced action is to exit (action 3); all other actions mean “hold”.
  
Embrace the chaos – every moment is logged, every trade is solitary.
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
# GLOBAL LOG QUEUE & CONSOLE LOG BUFFER – Where the mad whispers are collected
################################################################################

log_queue = queue.Queue()
console_log_buffer = []  # collect all log messages to dump every 100 iterations

def log_message(msg):
    log_queue.put(("log", msg))
    console_log_buffer.append(f"{dt.now().strftime('%H:%M:%S')} - {msg}")

def ui_post_message(msg_type, data):
    log_queue.put((msg_type, data))

################################################################################
# LOGGING UTILITY: Save logs & config every 100 iterations
################################################################################

def save_iteration_logs(iteration, pnl, config):
    date_str = dt.now().strftime("%d_%m_%y")
    folder_path = os.path.join("logfiles", date_str, f"iter_{iteration}_{pnl:+.2f}%")
    os.makedirs(folder_path, exist_ok=True)
    # Save current config snapshot
    with open(os.path.join(folder_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    # Save the console log messages
    with open(os.path.join(folder_path, "console.log"), "w") as f:
        f.write("\n".join(console_log_buffer))
    log_message(f"Logs and config saved in {folder_path}")
    console_log_buffer.clear()  # clear buffer for next 100 iterations

################################################################################
# DATA LOADING AND PREPROCESSING FUNCTIONS
################################################################################

def parse_tf_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    last_part = parts[-1]  # e.g., "5.csv"
    tf_str = last_part.replace('.csv', '')
    return int(tf_str)

def load_csv_with_tf(tf_value):
    filename = f"BITGET_ETHUSDT.P_{tf_value}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found!")
    df = pd.read_csv(filename)
    df.columns = df.columns.str.lower()
    renames = {
        'open': f'open_{tf_value}',
        'high': f'high_{tf_value}',
        'low': f'low_{tf_value}',
        'close': f'close_{tf_value}',
        'volume': f'volume_{tf_value}'
    }
    df.rename(columns=renames, inplace=True)
    df['time'] = df['time'].astype(int)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_all_csvs():
    files = glob.glob("BITGET_ETHUSDT.P_*.csv")
    if not files:
        raise FileNotFoundError("No CSV files matching 'BITGET_ETHUSDT.P_*.csv' found!")
    dfs, tf_values = [], []
    for f in files:
        tf_val = parse_tf_from_filename(f)
        dfs.append(load_csv_with_tf(tf_val))
        tf_values.append(tf_val)
    return dfs, tf_values

def merge_timeframes(dfs):
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='time', how='outer')
    merged.sort_values('time', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.ffill(inplace=True)
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged['time'] = pd.to_datetime(merged['time'], unit='s')
    return merged

def compute_indicators(merged, tfs):
    for tf in tfs:
        high = f'high_{tf}'
        low = f'low_{tf}'
        close = f'close_{tf}'
        merged[f'atr_{tf}'] = ta.volatility.average_true_range(merged[high], merged[low], merged[close], window=14, fillna=True)
        merged[f'rsi_{tf}'] = ta.momentum.rsi(merged[close], window=14, fillna=True)
        bb = ta.volatility.BollingerBands(merged[close], window=20, window_dev=2, fillna=True)
        merged[f'bb_upper_{tf}'] = bb.bollinger_hband()
        merged[f'bb_lower_{tf}'] = bb.bollinger_lband()
        merged[f'bb_mid_{tf}']   = bb.bollinger_mavg()
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

def normalize_features(df):
    norm_cols = [c for c in df.columns if c not in ['time', 'date']]
    for col in norm_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std
    return df

################################################################################
# TKINTER UI CLASS WITH STOP BUTTON & NET PROFIT BAR
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
        if worker_id not in self.worker_rows:
            self.add_worker(worker_id)
        row = self.worker_rows[worker_id]
        trades = result["trades_executed"]
        profit_pct = f"{result['total_profit_pct']:.4f}"
        factor = f"{result['profit_factor']:.4f}"
        profit_dollars = f"{result['total_profit_dollars']:.2f}"
        style = result.get("predicted_style", "N/A")
        self.tree.item(row, values=(trades, profit_pct, factor, profit_dollars, style))

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
                self.update_sim_result(data["worker_id"], data["result"])
            elif msg_type == "new_worker":
                self.add_worker(data)
            elif msg_type == "net_profit_update":
                self.net_profit_history = data.get("history", [])
                current = data.get("current", 0)
                self.lookback = data.get("lookback", self.lookback)
                self.update_net_profit_bar(self.net_profit_history, current)
        self.root.after(100, self.poll_queue)

################################################################################
# MASSIVE LSTM-BASED POLICY NETWORK WITH STYLE CLASSIFICATION
################################################################################

class TradingPolicy(nn.Module):
    def __init__(self, input_size, hidden_size=1024, lstm_layers=4):
        super(TradingPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        # Action head now outputs 4 logits: 0=hold, 1=open long, 2=open short, 3=exit
        self.fc_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        # Head for style classification (0: scalper, 1: swing)
        self.fc_style = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, obs_seq):
        lstm_out, _ = self.lstm(obs_seq)
        final = lstm_out[:, -1, :]
        logits_action = self.fc_action(final)
        logits_style = self.fc_style(final)
        return logits_action, logits_style

################################################################################
# TRAJECTORY BUFFER – The vessel of chaotic experience
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
# MODULAR RL TRADING ENVIRONMENT – One trade at a time
#
# New Action Space:
#   0: Hold
#   1: Open Long (if no trade open)
#   2: Open Short (if no trade open)
#   3: Exit trade (if trade is open)
################################################################################

class RLTradingEnv:
    def __init__(self, df, window_size=30, close_col='close_1', trade_cost=0.001,
                 stop_loss_pct=0.005, min_horizon=10, max_horizon=30,
                 penalty_coef=0.001, random_start=False, episode_length=500,
                 trade_open_reward=0.05, profit_trade_multi=1.0):
        self.df = df.reset_index(drop=True)
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
        self.reset()

    def reset(self):
        if self.random_start:
            max_start = len(self.df) - self.window_size - self.episode_length - 1
            self.start_index = random.randint(self.window_size, max_start)
            self.index = self.start_index
        else:
            self.start_index = self.window_size
            self.index = self.start_index

        self.position = 0      # 0: no trade, 1: long, -1: short
        self.entry_price = 0.0
        self.last_price = self.df.loc[self.index - 1, self.close_col]
        self.trade_count = 0
        self.trade_open_index = None
        self.hold_times = []
        self.done = False
        self.dynamic_horizon = random.randint(self.min_horizon, self.max_horizon)
        return self._get_observation()

    def step(self, action):
        curr_price = self.df.loc[self.index, self.close_col]
        reward = 0.0
        trade_executed = False

        # Forced exit if stop-loss hit:
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
                if action == 1:
                    self._open_trade(curr_price, 1)
                    trade_executed = True
                elif action == 2:
                    self._open_trade(curr_price, -1)
                    trade_executed = True
                # else: action 0 means hold when no trade is open.
            else:
                # Trade is open; only valid non-forced action is to exit (action 3)
                if action == 3:
                    reward += self._close_position(curr_price)
                    trade_executed = True
                else:
                    reward += (curr_price - self.last_price) * self.position

        if trade_executed:
            reward += self.trade_open_reward
            reward -= self.trade_cost

        self.last_price = curr_price
        self.index += 1

        if self.index >= self.start_index + self.episode_length:
            self.done = True
            if self.position != 0:
                reward += self._close_position(curr_price)
        if self.index >= len(self.df):
            self.done = True
            if self.position != 0:
                reward += self._close_position(curr_price)

        return self._get_observation(), reward, self.done, {"trade_executed": trade_executed}

    def _open_trade(self, price, position):
        self.position = position
        self.entry_price = price
        self.trade_open_index = self.index
        self.trade_count += 1

    def _close_position(self, curr_price):
        reward = 0.0
        if self.position == 1:
            reward = (curr_price - self.entry_price)
            profit_pct = (curr_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            reward = (self.entry_price - curr_price)
            profit_pct = (self.entry_price - curr_price) / self.entry_price
        else:
            profit_pct = 0.0

        if profit_pct > 0:
            bonus = self.trade_open_reward + self.profit_trade_multi * profit_pct
            reward += bonus

        if self.trade_open_index is not None:
            self.hold_times.append(self.index - self.trade_open_index)
        self.position = 0
        self.trade_open_index = None
        return reward

    def _get_observation(self):
        start = max(0, self.index - self.window_size)
        obs_df = self.df.iloc[start:self.index]
        obs_cols = [c for c in obs_df.columns if c not in ['time', 'date']]
        return obs_df[obs_cols].values.astype("float32")

    def get_avg_hold_time(self):
        return sum(self.hold_times) / len(self.hold_times) if self.hold_times else 0.0

    def get_final_profit(self):
        return None

################################################################################
# SCATTER PLOT FOR TRADING STYLES – A glimpse into the fractal soul
################################################################################

def plot_trading_styles(sim_results, iteration):
    x_vals, y_vals, colors = [], [], []
    for sim in sim_results:
        if sim:
            x_vals.append(sim.get("avg_hold_time", 0))
            y_vals.append(sim.get("trades_executed", 0))
            style = sim.get("style", "scalper")
            colors.append("blue" if style == "scalper" else "orange")
    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, y_vals, c=colors, alpha=0.7, edgecolors='k')
    plt.xlabel("Average Hold Time")
    plt.ylabel("Trade Count")
    plt.title(f"Trading Styles at Iteration {iteration}")
    plt.grid(True)
    plot_filename = f"trading_styles_iter{iteration}.png"
    plt.savefig(plot_filename)
    plt.close()
    log_message(f"Scatter plot saved: {plot_filename}")

################################################################################
# WORKER FUNCTION WITH ENTROPY BONUS & STYLE OBSERVATION
################################################################################

def refine_worker_torch(worker_id, tf_val, config, merged_df, global_state, device="cuda"):
    try:
        ui_post_message("new_worker", worker_id)
        close_col = f"close_{tf_val}"
        if close_col not in merged_df.columns:
            log_message(f"[{worker_id}] No close column for timeframe {tf_val}. Skipping.")
            return {}, None

        window_size = config.get("window_size", 30)
        trade_cost = config.get("trade_cost", 0.001)
        env = RLTradingEnv(
            merged_df,
            window_size=window_size,
            close_col=close_col,
            trade_cost=trade_cost,
            stop_loss_pct=config.get("stop_loss_pct", 0.005),
            min_horizon=config.get("min_horizon", 10),
            max_horizon=config.get("max_horizon", 30),
            penalty_coef=config.get("penalty_coef", 0.001),
            random_start=True,
            episode_length=config.get("episode_length", 500),
            trade_open_reward=config.get("trade_open_reward", 0.05),
            profit_trade_multi=config.get("profit_trade_multi", 1.0)
        )
        sample_obs = env._get_observation()
        _, num_features = sample_obs.shape

        hidden_size = config.get("model_hidden_size", 512)
        lstm_layers = config.get("model_lstm_layers", 2)
        policy = TradingPolicy(input_size=num_features, hidden_size=hidden_size, lstm_layers=lstm_layers).to(device)
        try:
            policy.load_state_dict(global_state)
        except Exception as ex:
            log_message(f"[{worker_id}] Error loading global state: {ex}")
            return {}, None

        optimizer = optim.Adam(policy.parameters(), lr=config.get("learning_rate", 1e-3))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
        gamma = 0.99
        entropy_coef = config.get("entropy_coef", 0.05)
        epochs = config["epochs"]
        batch_episodes = config.get("batch_episodes", 10)
        num_batches = epochs // batch_episodes

        for b in range(num_batches):
            total_loss = 0.0
            for _ in range(batch_episodes):
                trajectory = TrajectoryBuffer()
                obs = env.reset()
                done = False
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

        # Evaluation phase with style observation
        obs = env.reset()
        done = False
        total_reward = 0.0
        style_predictions = []
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

        style_label = "scalper" if (sum(1 for p in style_predictions if p == 0) > len(style_predictions)/2) else "swing"
        sim_res = {
            "trades_executed": env.trade_count,
            "avg_hold_time": env.get_avg_hold_time(),
            "total_profit_pct": total_reward,
            "profit_factor": 1.0 + total_reward if total_reward > 0 else 1.0,
            "total_profit_dollars": total_reward * config.get("margin", 300) / 100.0,
            "predicted_style": style_label
        }
        ui_post_message("sim_result", {"worker_id": worker_id, "result": sim_res})
        log_message(f"[{worker_id}] Simulation complete. Style predicted: {style_label}, Profit: {total_reward:.4f}")
        
        return policy.state_dict(), sim_res

    except Exception as e:
        log_message(f"[{worker_id}] Exception in worker: {e}")
        return {}, None

################################################################################
# LATEST CHECKPOINT LOADER FOR PYTORCH
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
# MAIN PARALLEL REFINEMENT APP – The vortex of self-observation
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
        window_size = self.config.get("window_size", 30)
        num_features = len([c for c in self.merged_df.columns if c not in ['time', 'date']])
        hidden_size = self.config.get("model_hidden_size", 512)
        lstm_layers = self.config.get("model_lstm_layers", 2)
        
        self.global_policy = TradingPolicy(input_size=num_features, hidden_size=hidden_size, lstm_layers=lstm_layers).to(self.device)
        
        refined_ckpt = load_latest_checkpoint_torch(
            base_filename="multi_tf_model.pt",
            prefix="multi_tf_model_refined_iter",
            ext=".pt"
        )
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
        worker_multiplier = self.config.get("worker_multiplier", 10)
        tfs = self.config.get("target_timeframes", [2])
        discard_iter = self.config.get("discard_losers_after_iter", 150)
        lookback = self.config.get("lookback_iterations", 10)

        for it in range(start_iter, iterations + 1):
            elapsed = (dt.now() - self.start_time).total_seconds() / 60.0
            if self.config["max_time"] > 0 and elapsed >= self.config["max_time"]:
                log_message("Max training time reached. Exiting loop.")
                break

            decay_factor = (it / iterations) ** 2
            current_entropy = self.initial_entropy * (1 - (self.entropy_decrease_pct/100) * decay_factor)
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
                        log_message(f"Worker simulation results: {sim_res}")
                    except Exception as ex:
                        log_message(f"Worker error: {ex}")

            for sim in sim_results:
                if sim:
                    threshold = self.config.get("scalper_threshold", 15)
                    sim["style"] = "scalper" if sim.get("avg_hold_time", 0) < threshold else "swing"

            plot_trading_styles(sim_results, it)

            all_profits = [sim.get("total_profit_pct", 0.0) for sim in sim_results if sim]
            if it > discard_iter:
                filtered = [(st, sim) for st, sim in zip(worker_states, sim_results) if sim and sim.get("total_profit_pct", 0.0) > 0]
                if not filtered:
                    log_message(f"No profitable workers at iteration {it}. Skipping update.")
                    continue
                worker_states, sims_filtered = zip(*filtered)
                profits = [sim.get("total_profit_pct", 0.0) for sim in sims_filtered]
                scalper_profits = [sim["total_profit_pct"] for sim in sims_filtered if sim.get("style") == "scalper"]
                swing_profits = [sim["total_profit_pct"] for sim in sims_filtered if sim.get("style") == "swing"]
                avg_scalper = np.mean(scalper_profits) if scalper_profits else 0
                avg_swing = np.mean(swing_profits) if swing_profits else 0
                if avg_scalper > avg_swing:
                    style_multipliers = {"scalper": 1.2, "swing": 0.8}
                elif avg_swing > avg_scalper:
                    style_multipliers = {"scalper": 0.8, "swing": 1.2}
                else:
                    style_multipliers = {"scalper": 1.0, "swing": 1.0}
                weights = [1.0 * style_multipliers.get(sim.get("style", "scalper"), 1.0) for sim in sims_filtered]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
            else:
                profits = all_profits
                P = sum(1 for p in profits if p > 0)
                L = len(profits) - P
                N = len(profits)
                if P == 0:
                    weights = [1 / N] * N
                else:
                    weights = []
                    for sim in sim_results:
                        if sim:
                            profit = sim.get("total_profit_pct", 0.0)
                            if profit > 0:
                                base_weight = (100 - (10 * L)) / P
                            else:
                                base_weight = 10
                            style = sim.get("style", "scalper")
                            multiplier = 1.2 if style == "scalper" and profit > 0 else 1.0
                            weights.append(base_weight * multiplier)
                    total_weight = sum(weights)
                    weights = [w / total_weight for w in weights]

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

            valid_profits = [p for p in all_profits if p is not None]
            iter_profit = sum(valid_profits) / len(valid_profits) if valid_profits else 0.0
            self.iter_profit_history.append(iter_profit)
            ui_post_message("net_profit_update", {"history": self.iter_profit_history[-lookback:], "current": iter_profit, "lookback": lookback})
            scalper_avg = np.mean([sim["total_profit_pct"] for sim in sim_results if sim and sim.get("style") == "scalper"] or [0])
            swing_avg = np.mean([sim["total_profit_pct"] for sim in sim_results if sim and sim.get("style") == "swing"] or [0])
            log_message(f"Iteration {it} style summary: Scalper Avg Profit: {scalper_avg:.4f}, Swing Avg Profit: {swing_avg:.4f}")

            # Save logs every 100 iterations
            if it % 100 == 0:
                last_100_avg = np.mean(self.iter_profit_history[-100:]) if len(self.iter_profit_history) >= 100 else iter_profit
                save_iteration_logs(it, last_100_avg * 100, self.config)

            if self.stop_requested:
                log_message("Stop requested. Exiting refinement loop after current iteration.")
                break

        log_message("Refinement loop completed. The chaos subsides...")

################################################################################
# MAIN FUNCTION – Enter the vortex
################################################################################

def main():
    parser = argparse.ArgumentParser(description="RL Trading with Dynamic Rewards, Short Episodes, Checkpoint Resumption, Self-Observing Style Guidance, and Iteration Logging")
    parser.add_argument("config_file", nargs="?", default="config.json", help="Path to JSON config file (default: config.json)")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
    config.setdefault("entropy_decrease_pct", 1)
    config.setdefault("scalper_threshold", 15)
    print("Starting UI... Prepare for the maelstrom.")
    root = tk.Tk()
    ui = RefineUI(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    try:
        dfs, tfs = load_all_csvs()
    except Exception as e:
        log_message(str(e))
        input("Press Enter to exit...")
        return
    merged_df = merge_timeframes(dfs)
    log_message(f"Merged data: {merged_df.shape[0]} rows, columns: {merged_df.columns.tolist()}")
    merged_df = compute_indicators(merged_df, tfs)
    merged_df = normalize_features(merged_df)
    log_message(f"After computing and normalizing indicators: {merged_df.shape}")
    merged_df['date'] = merged_df['time'].dt.date
    app = RefineApp(ui, config, merged_df, device)
    ui.set_stop_callback(app.request_stop)
    root.mainloop()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
