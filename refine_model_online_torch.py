#!/usr/bin/env python3
"""
Parallel Online Policy Gradient Refinement & Simulation for Trading (PyTorch)
using Real Candle Data in a Candle-by-Candle Environment, with a Tkinter UI.

High-Level Flow:
1. Reads configuration from a JSON file (default "config.json").
2. Loads CSV files (BITGET_ETHUSDT.P_*.csv), merges them by 'time', and computes indicators.
3. For each global iteration:
   - Spawns multiple workers (target timeframe × worker_multiplier).
   - Each worker runs a simplified policy gradient loop on a candle-by-candle environment:
       * The environment steps through the DataFrame row by row,
         tracking position (+1=long, -1=short, 0=flat) and computing reward from price changes.
   - The main thread averages worker parameters to update the global model.
   - A checkpoint is saved after each iteration.
4. A Tkinter UI in the main thread shows:
   - A text log,
   - A table of simulation results,
   - Progress bars for each worker's training progress.
5. The final model can be used or extended for live usage.
   
Disclaimer: This is an illustrative example. Real trading RL is more complex.
"""

import os, glob, re, json, argparse, random, threading, queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

################################################################################
# GLOBAL LOG QUEUE AND UI UPDATE FUNCTIONS
################################################################################

log_queue = queue.Queue()

def log_message(msg):
    """Posts a text message to the log queue."""
    log_queue.put(("log", msg))

def ui_post_message(msg_type, data):
    """
    Posts a structured message to the UI.
    msg_type can be "progress", "sim_result", "new_worker", or "log".
    """
    log_queue.put((msg_type, data))

################################################################################
# HELPER FUNCTIONS FOR DATA LOADING AND MERGING
################################################################################

def parse_tf_from_filename(filename):
    """
    Extracts the timeframe from a filename like "BITGET_ETHUSDT.P_5.csv".
    Returns an integer (e.g. 5).
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    last_part = parts[-1]  # e.g. "5.csv"
    tf_str = last_part.replace('.csv', '')
    return int(tf_str)

def load_csv_with_tf(tf_value):
    """
    Loads a CSV file for the given timeframe and renames columns to be unique.
    For example, "open" becomes "open_<tf_value>".
    """
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
    """
    Finds all files matching "BITGET_ETHUSDT.P_*.csv", loads them,
    and returns a tuple (list_of_dataframes, list_of_timeframes).
    """
    files = glob.glob("BITGET_ETHUSDT.P_*.csv")
    if not files:
        raise FileNotFoundError("No CSV files matching 'BITGET_ETHUSDT.P_*.csv' found!")
    dfs = []
    tf_values = []
    for f in files:
        tf_val = parse_tf_from_filename(f)
        df = load_csv_with_tf(tf_val)
        dfs.append(df)
        tf_values.append(tf_val)
    return dfs, tf_values

def merge_timeframes(dfs):
    """
    Merges a list of DataFrames on the 'time' column using an outer join,
    forward-fills missing values, drops remaining NaNs, and converts 'time' to datetime.
    """
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
    """
    Computes technical indicators (ATR, RSI, Bollinger Bands) for each timeframe.
    """
    for tf in tfs:
        high = f'high_{tf}'
        low = f'low_{tf}'
        close = f'close_{tf}'
        merged[f'atr_{tf}'] = ta.volatility.average_true_range(merged[high], merged[low], merged[close], window=14, fillna=True)
        merged[f'rsi_{tf}'] = ta.momentum.rsi(merged[close], window=14, fillna=True)
        bb = ta.volatility.BollingerBands(merged[close], window=20, window_dev=2, fillna=True)
        merged[f'bb_upper_{tf}'] = bb.bollinger_hband()
        merged[f'bb_lower_{tf}'] = bb.bollinger_lband()
        merged[f'bb_mid_{tf}'] = bb.bollinger_mavg()
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

################################################################################
# TKINTER UI CLASS (Advanced)
################################################################################

class RefineUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Online Trading RL Refinement")
        self.log_text = ScrolledText(root, state='normal', width=80, height=15)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self.table_frame = tk.Frame(root)
        self.table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(self.table_frame, columns=("Trades", "Profit", "Factor", "Profit($)"), show="headings")
        self.tree.heading("Trades", text="Trades")
        self.tree.heading("Profit", text="Profit %")
        self.tree.heading("Factor", text="Factor")
        self.tree.heading("Profit($)", text="Profit($)")
        self.tree.column("Trades", width=60)
        self.tree.column("Profit", width=80)
        self.tree.column("Factor", width=80)
        self.tree.column("Profit($)", width=80)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.worker_rows = {}
        self.prog_frame = tk.Frame(self.table_frame)
        self.prog_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.prog_bars = {}
        self.root.after(100, self.poll_queue)

    def add_worker(self, worker_id):
        row = self.tree.insert("", tk.END, text=worker_id, values=("0", "0", "0", "0"))
        self.worker_rows[worker_id] = row
        label = tk.Label(self.prog_frame, text=worker_id)
        label.pack()
        pb = ttk.Progressbar(self.prog_frame, orient="horizontal", length=200, mode="determinate")
        pb["value"] = 0
        pb["maximum"] = 100
        pb.pack()
        self.prog_bars[worker_id] = pb

    def update_progress(self, worker_id, progress):
        if worker_id in self.prog_bars:
            self.prog_bars[worker_id]["value"] = progress

    def update_sim_result(self, worker_id, result):
        if worker_id not in self.worker_rows:
            self.add_worker(worker_id)
        row = self.worker_rows[worker_id]
        trades = result["trades_executed"]
        profit_pct = f"{result['total_profit_pct']:.4f}"
        factor = f"{result['profit_factor']:.4f}"
        profit_dollars = f"{result['total_profit_dollars']:.2f}"
        self.tree.item(row, values=(trades, profit_pct, factor, profit_dollars))

    def poll_queue(self):
        while not log_queue.empty():
            msg_type, data = log_queue.get_nowait()
            if msg_type == "log":
                self.log_text.insert(tk.END, data + "\n")
                self.log_text.see(tk.END)
            elif msg_type == "progress":
                self.update_progress(data["worker_id"], data["progress"])
            elif msg_type == "sim_result":
                self.update_sim_result(data["worker_id"], data["result"])
            elif msg_type == "new_worker":
                self.add_worker(data)
        self.root.after(100, self.poll_queue)

################################################################################
# RL TRADING ENVIRONMENT
################################################################################

class RLTradingEnv:
    """
    A candle-by-candle environment that steps through a DataFrame.
    Observations: a window of past rows (default 30).
    Actions: 0=Hold, 1=Buy, 2=Sell.
    Position: +1=long, -1=short, 0=flat.
    Reward: computed from PnL changes when opening/closing or holding a position.
    """
    def __init__(self, df, window_size=30, close_col='close_1'):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.close_col = close_col
        self.position = 0
        self.entry_price = 0.0
        self.index = window_size
        self.done = False

    def reset(self):
        self.position = 0
        self.entry_price = 0.0
        self.index = self.window_size
        self.done = False
        return self._get_observation()

    def step(self, action):
        reward = 0.0
        curr_price = self.df.loc[self.index, self.close_col]
        if action == 1:  # Buy
            if self.position == -1:
                reward = (self.entry_price - curr_price)
            self.position = 1
            self.entry_price = curr_price
        elif action == 2:  # Sell
            if self.position == 1:
                reward = (curr_price - self.entry_price)
            self.position = -1
            self.entry_price = curr_price
        elif action == 0:  # Hold
            if self.position == 1:
                reward = (curr_price - self.entry_price)*0.01
            elif self.position == -1:
                reward = (self.entry_price - curr_price)*0.01
        self.index += 1
        if self.index >= len(self.df):
            self.done = True
        obs = self._get_observation()
        return obs, reward, self.done, {}

    def _get_observation(self):
        start = self.index - self.window_size
        end = self.index
        if start < 0:
            start = 0
        obs_df = self.df.iloc[start:end]
        obs_cols = [c for c in obs_df.columns if c not in ['time','date']]
        return obs_df[obs_cols].values.astype(np.float32)

################################################################################
# POLICY NETWORKS
################################################################################

class TradingPolicy(nn.Module):
    """A minimal policy network that outputs logits for actions: 0=Hold, 1=Buy, 2=Sell."""
    def __init__(self, input_size, hidden_size=64):
        super(TradingPolicy, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 3)
    def forward(self, obs):
        x = torch.relu(self.l1(obs))
        logits = self.l2(x)
        return logits

################################################################################
# WORKER FUNCTION: POLICY GRADIENT ON RLTradingEnv
################################################################################

def refine_worker_torch(worker_id, tf_val, config, merged_df, global_state, device="cuda"):
    ui_post_message("new_worker", worker_id)
    close_col = f'close_{tf_val}'
    if close_col not in merged_df.columns:
        log_message(f"[{worker_id}] No close column for timeframe {tf_val}. Skipping.")
        return {}, None
    window_size = config.get("window_size", 30)
    env = RLTradingEnv(merged_df, window_size=window_size, close_col=close_col)
    num_features = len([c for c in merged_df.columns if c not in ['time','date']])
    input_size = num_features * window_size
    policy = TradingPolicy(input_size=input_size).to(device)
    try:
        policy.load_state_dict(global_state)
    except Exception as ex:
        log_message(f"[{worker_id}] Error loading global state: {ex}")
        return {}, None
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    gamma = 0.99
    epochs = config["epochs"]
    for ep in range(epochs):
        prog = int((ep/epochs)*100)
        ui_post_message("progress", {"worker_id": worker_id, "progress": prog})
        obs = env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
            logits = policy(obs_flat)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            new_obs, r, done, _ = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            rewards.append(r)
            obs = new_obs
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if len(returns) > 1:
            returns = (returns - returns.mean())/(returns.std()+1e-8)
        loss = sum([-lp * R for lp, R in zip(log_probs, returns)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    obs = env.reset()
    done = False
    total_reward = 0.0
    trades = 0
    while not done:
        obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
        logits = policy(obs_flat)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()
        new_obs, r, done, _ = env.step(action)
        if action != 0:
            trades += 1
        total_reward += r
        obs = new_obs
    sim_res = {
        "trades_executed": trades,
        "success_rate": 1.0 if total_reward > 0 else 0.0,
        "total_profit_pct": total_reward,
        "profit_factor": 1.0+total_reward if total_reward > 0 else 1.0,
        "total_profit_dollars": total_reward * config.get("margin", 300) / 100.0
    }
    ui_post_message("sim_result", {"worker_id": worker_id, "result": sim_res})
    return policy.state_dict(), sim_res

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
# MAIN PARALLEL REFINEMENT APP
################################################################################

class RefineApp:
    def __init__(self, ui, config, merged_df, device):
        self.ui = ui
        self.config = config
        self.merged_df = merged_df
        self.device = device
        self.global_policy = None
        self.global_state = None
        self.start_time = datetime.now()
        self.init_global_policy()
        self.refine_thread = threading.Thread(target=self.refinement_loop, daemon=True)
        self.refine_thread.start()

    def init_global_policy(self):
        window_size = self.config.get("window_size", 30)
        num_features = len([c for c in self.merged_df.columns if c not in ['time','date']])
        input_size = num_features * window_size
        self.global_policy = TradingPolicy(input_size=input_size).to(self.device)
        base_file = "multi_tf_model.pt"
        if os.path.exists(base_file):
            try:
                base_ckpt = torch.load(base_file)
                self.global_policy.load_state_dict(base_ckpt)
                log_message("Loaded base model from multi_tf_model.pt")
            except Exception as ex:
                log_message(f"Failed to load base model: {ex}. Using new global model.")
        ckp = load_latest_checkpoint_torch(base_filename="multi_tf_model.pt", prefix="multi_tf_model_refined_iter", ext=".pt")
        if ckp is not None:
            try:
                self.global_policy.load_state_dict(ckp)
                log_message("Global model loaded from refined checkpoint.")
            except Exception as ex:
                log_message(f"Failed to load refined checkpoint: {ex}. Using new global model.")
        else:
            log_message("No refined checkpoint found; using new global model.")
        self.global_state = self.global_policy.state_dict()

    def refinement_loop(self):
        iterations = self.config["iterations"]
        worker_multiplier = self.config.get("worker_multiplier", 1)
        tfs = self.config.get("target_timeframes", [1])
        for it in range(1, iterations+1):
            elapsed = (datetime.now() - self.start_time).total_seconds()/60.0
            if self.config["max_time"] > 0 and elapsed >= self.config["max_time"]:
                log_message("Max training time reached. Exiting loop.")
                break
            log_message(f"\n--- Global Iteration {it}/{iterations} ---")
            worker_futures = []
            total_workers = len(tfs) * worker_multiplier
            with ThreadPoolExecutor(max_workers=total_workers) as executor:
                for tf_val in tfs:
                    for w in range(worker_multiplier):
                        worker_id = f"tf{tf_val}_w{w}"
                        worker_futures.append(
                            executor.submit(refine_worker_torch, worker_id, tf_val, self.config, self.merged_df.copy(), self.global_state, self.device)
                        )
                worker_states = []
                for future in as_completed(worker_futures):
                    try:
                        st_dict, sim_res = future.result()
                        if st_dict:
                            worker_states.append(st_dict)
                        log_message(f"Worker simulation results: {sim_res}")
                    except Exception as ex:
                        log_message(f"Worker error: {ex}")
            if not worker_states:
                log_message("No worker updated the model. Skipping averaging.")
                continue
            new_state = {}
            for key in self.global_state.keys():
                vals = [st[key] for st in worker_states]
                new_state[key] = torch.mean(torch.stack(vals), dim=0)
            self.global_state = new_state
            self.global_policy.load_state_dict(self.global_state)
            log_message(f"Global policy updated for iteration {it}.")
            ckpt_name = f"multi_tf_model_refined_iter{it}.pt"
            torch.save({"model_state": self.global_state}, ckpt_name)
            log_message(f"Saved checkpoint: {ckpt_name}")
        log_message("Refinement loop completed.")
        # Optionally, do not auto-quit UI
        # self.ui.root.after(0, self.ui.root.quit)

################################################################################
# MAIN FUNCTION
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Parallel RL Refinement with advanced UI (PyTorch) using real candle environment.")
    parser.add_argument("config_file", nargs="?", default="config.json", help="Path to JSON config file (default: config.json)")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    print("Starting UI...")

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
    log_message(f"After computing indicators: {merged_df.shape}")

    # Ensure 'date' column is present for simulation splitting
    merged_df['date'] = merged_df['time'].dt.date

    app = RefineApp(ui, config, merged_df, device)
    root.mainloop()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
