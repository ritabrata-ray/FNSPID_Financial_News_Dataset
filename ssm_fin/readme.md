# Portfolio Optimization and Trading Strategy

## Overview

This project contains:

- **SSQP-based portfolio optimization** (`optimize_portfolio`) — for maximizing Sharpe ratio.
- **ONS-based baseline optimizer** — for online portfolio updates.
- **Accumulated wealth calculator** — to track backtest performance.
- **S4 State Space Model** — for time-series modeling.

All training and trading are orchestrated through the `train_and_run_trade.py` script.

## File Structure

- `tst/model_based_portfolio.py`  
  Contains:
  - `optimize_portfolio` (SSQP method)
  - ONS baseline optimizer
  - Accumulated wealth computation

- `tst/ssm.py`  
  Contains:
  - S4 State Space Model (SSM) for sequential data tasks

- `train_and_run_trade.py`  
  Main file:
  - Trains models
  - Runs trading strategies (with and without sentiment signals)
  - Backtests over specified periods

## How to Run

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

To run, use
```
python train_and_run_trade.py