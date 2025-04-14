#!/usr/bin/env python3
"""
RL Trading Agent with Dynamic Rewards, Shortened Episodes, Hard Maximum Hold,
Checkpoint Resumption, Realized Profit Bonus, and Weighted Worker Updates

Key improvements (v0.5):
  - Splits the dataset into shorter episodes (episode_length) so trades can only
    accumulate rewards over a limited number of candles.
  - Enforces a hard maximum hold (episode_length) to force trade closure.
  - Enforces a fixed stop loss (SL) while letting the agent learn its own take profit (TP).
  - Rewards realized profit on trade closure with a fixed bonus plus a multiplier per percent gain.
  - Global model updates are now weighted: profitable workers’ updates receive higher weight.
  - Checkpoint logic resumes training from the last saved iteration.
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
# GLOBAL LOG QUEUE AND UI UPDATE
################################################################################

log_queue = queue.Queue()

def log_message(msg):
    log_queue.put(("log", msg))

def ui_post_message(msg_type, data):
    log_queue.put((msg_type, data))

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
# TKINTER UI CLASS
################################################################################

class RefineUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Trading Refinement")
        self.log_text = ScrolledText(root, state='normal', width=80, height=15)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self.table_frame = tk.Frame(root)
        self.table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            self.table_frame,
            columns=("Trades", "Profit", "Factor", "Profit($)"),
            show="headings"
        )
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
# MODULAR RL TRADING ENVIRONMENT WITH SHORT EPISODES, HARD MAX HOLD & PROFIT BONUS
################################################################################

class RLTradingEnv:
    """
    Candle-by-candle trading environment that:
      - Uses a dynamic maximum holding horizon (randomly chosen between min_horizon and max_horizon)
      - Applies a soft penalty for holding beyond that horizon.
      - Enforces a fixed stop loss (SL); no forced take profit (TP) so the agent learns TP.
      - Ends each episode after a fixed episode_length.
      - Rewards realized profit on trade closure via a fixed bonus and a multiplier per percent gain.
      - Handles both long and short positions (no pyramiding).
    """
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

        self.position = 0
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

        if self.position != 0:
            reward = (curr_price - self.last_price) * self.position

        forced_exit = False
        if self.position != 0:
            move_pct = (curr_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - curr_price) / self.entry_price
            if move_pct <= -self.stop_loss_pct:
                forced_exit = True

        if forced_exit:
            reward += self._close_position(curr_price)
            trade_executed = True
        else:
            reward += self._execute_trade(curr_price, action=action)

        if self.position != 0 and self.trade_open_index is not None:
            hold_duration = self.index - self.trade_open_index
            reward -= self._calculate_hold_penalty(hold_duration)

        if trade_executed:
            reward += self.trade_open_reward
            reward -= self.trade_cost

        self.last_price = curr_price
        self.index += 1

        if self.index >= self.start_index + self.episode_length:
            self.done = True
            if self.position != 0 and self.trade_open_index is not None:
                reward += self._close_position(curr_price)
        if self.index >= len(self.df):
            self.done = True
            if self.position != 0 and self.trade_open_index is not None:
                reward += self._close_position(curr_price)

        return self._get_observation(), reward, self.done, {"trade_executed": trade_executed}

    def _calculate_hold_penalty(self, hold_duration):
        if hold_duration > self.dynamic_horizon:
            return (hold_duration - self.dynamic_horizon) * self.penalty_coef
        return 0.0

    def _execute_trade(self, curr_price, action=None):
        reward = 0.0
        if action == 1:  # Buy
            if self.position != 1:
                if self.position == -1 and self.trade_open_index is not None:
                    reward += self._close_position(curr_price)
                self._open_trade(curr_price, 1)
        elif action == 2:  # Sell
            if self.position != -1:
                if self.position == 1 and self.trade_open_index is not None:
                    reward += self._close_position(curr_price)
                self._open_trade(curr_price, -1)
        return reward

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
# MASSIVE LSTM-BASED POLICY NETWORK
################################################################################

class TradingPolicy(nn.Module):
    def __init__(self, input_size, hidden_size=1024, lstm_layers=4):
        super(TradingPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, obs_seq):
        lstm_out, _ = self.lstm(obs_seq)
        final = lstm_out[:, -1, :]
        logits = self.fc(final)
        return logits

################################################################################
# WORKER FUNCTION WITH ENTROPY BONUS & DYNAMIC REWARDS
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
                    logits = policy(obs_seq)
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
                    logits_t = policy(obs_t.unsqueeze(0))
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
            progress = int(((b+1) / num_batches) * 100)
            ui_post_message("progress", {"worker_id": worker_id, "progress": progress})

        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_seq = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_seq)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
            new_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = new_obs

        sim_res = {
            "trades_executed": env.trade_count,
            "avg_hold_time": env.get_avg_hold_time(),
            "total_profit_pct": total_reward,
            "profit_factor": 1.0 + total_reward if total_reward > 0 else 1.0,
            "total_profit_dollars": total_reward * config.get("margin", 300) / 100.0
        }
        ui_post_message("sim_result", {"worker_id": worker_id, "result": sim_res})
        
        # --- Weighted Global Model Update is handled in the refinement loop ---
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
        num_features = len([c for c in self.merged_df.columns if c not in ['time', 'date']])
        input_size = num_features
        hidden_size = self.config.get("model_hidden_size", 512)
        lstm_layers = self.config.get("model_lstm_layers", 2)
        
        self.global_policy = TradingPolicy(input_size=input_size, hidden_size=hidden_size, lstm_layers=lstm_layers).to(self.device)
        
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
        worker_multiplier = self.config.get("worker_multiplier", 1)
        tfs = self.config.get("target_timeframes", [1])

        for it in range(start_iter, iterations + 1):
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60.0
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

            if not worker_states:
                log_message("No worker updated the model. Skipping averaging.")
                continue

            # --- Weighted Averaging Based on Profitability ---
            profits = [sim.get("total_profit_pct", 0.0) for sim in sim_results]
            P = sum(1 for p in profits if p > 0)
            L = len(profits) - P
            N = len(profits)
            if P == 0:
                weights = [1 / N] * N
            else:
                weights = []
                for profit in profits:
                    if profit > 0:
                        weight = (100 - (10 * L)) / P
                    else:
                        weight = 10
                    weights.append(weight)
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

        log_message("Refinement loop completed.")

################################################################################
# MAIN FUNCTION
################################################################################

def main():
    parser = argparse.ArgumentParser(description="RL Trading with Dynamic Rewards, Short Episodes, Hard Max Hold, Checkpoint Resumption & Weighted Updates")
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
    merged_df = normalize_features(merged_df)
    log_message(f"After computing and normalizing indicators: {merged_df.shape}")
    merged_df['date'] = merged_df['time'].dt.date
    app = RefineApp(ui, config, merged_df, device)
    root.mainloop()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
