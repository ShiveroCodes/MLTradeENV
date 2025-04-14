# Unhinged Trading Framework v6.0

A reinforcement learning-based trading simulator built around PPO (Proximal Policy Optimization), with a per-asset calibration mechanism, PineScript-inspired technical indicators, and a basic Tkinter UI.

This project was developed as an experimental platform for testing dynamic trading strategies in multi-asset environments using deep learning, signal-based filters, and adaptive reward shaping.

## 🧠 Features

- **Per-asset SL/TP Calibration**: Each asset is auto-calibrated using a basic EMA crossover backtest.
- **PineScript-style Indicator Logic**: Includes supertrend, RSI, Bollinger Bands, pivot detection, and more.
- **Adaptive Strategy Selection**: Agent adapts its behavior based on custom risk and reward parameters.
- **PPO with LSTM**: A multi-layer LSTM policy network learns sequential price dependencies.
- **Tkinter UI**: Displays current iteration, active asset, and profit percentage in a tiny GUI.
- **Multi-timeframe, Multi-asset Ready**: Works with multiple crypto pairs and timeframe-specific CSVs.
- **Safety Logging**: Automatically logs console output and saves checkpoints every N iterations.

## ⚙️ Dependencies

- Python 3.8+
- `torch`, `numpy`, `pandas`, `matplotlib`, `ta`, `tkinter`

Install dependencies:
```bash
pip install torch numpy pandas matplotlib ta