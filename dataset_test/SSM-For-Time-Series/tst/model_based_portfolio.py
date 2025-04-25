import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if portfolio_volatility < 1e-8:
        return 1e6  # penalize near-zero volatility portfolios
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(expected_returns, cov_matrix):
    N = len(expected_returns)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(N))
    initial_weights = np.array([1/N] * N)

    result = sco.minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success or result.x is None:
        return initial_weights  # fallback to uniform weights
    return result.x

def backtest_portfolio(N, historical, predicted, window=60):
    D = historical.shape[1]
    daily_returns = []
    weights_list = []

    for t in range(window, D):
        hist_window = historical[:, t - window:t].T  # shape (window, N)
        pred_today = predicted[:, t]                 # shape (N,)
        cov_matrix = np.cov(hist_window, rowvar=False) * 252

        try:
            weights = optimize_portfolio(pred_today, cov_matrix)
        except Exception:
            weights = np.ones(N) / N  # fallback

        realized_return = np.dot(weights, historical[:, t])
        daily_returns.append(realized_return)
        weights_list.append(weights)

    returns = np.array(daily_returns)
    weights_matrix = np.array(weights_list)  # shape (days, N)

    # Metrics
    avg_daily_return = np.mean(returns)
    std_daily_return = np.std(returns)
    sharpe_ratio = avg_daily_return / (std_daily_return + 1e-8) * np.sqrt(252)
    cumulative_return = np.prod(1 + returns) - 1
    volatility = std_daily_return * np.sqrt(252)

    # Plot cumulative return
    cum_returns = np.cumprod(1 + returns)
    plt.figure(figsize=(10, 5))
    plt.plot(cum_returns, label='Cumulative Return')
    plt.title("Portfolio Cumulative Return")
    plt.xlabel("Days")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot portfolio weights
    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.plot(weights_matrix[:, i], label=f'Stock {i+1}')
    plt.title("Portfolio Weights Over Time")
    plt.xlabel("Days")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'cumulative_return': cumulative_return,
        'annualized_volatility': volatility,
        'annualized_return': avg_daily_return * 252,
        'sharpe_ratio': sharpe_ratio,
        'weights': weights_matrix,
        'daily_returns': returns
    }
def portfolio_final_wealth(historical_returns, true_returns, predicted_returns, risk_free_rate=0.0):
    historical_returns = historical_returns.T
    true_returns = true_returns.T
    predicted_returns = predicted_returns.T
    num_days = true_returns.shape[0]
    num_stocks = true_returns.shape[1]

    W_t = np.zeros(num_days + 1)
    W_t[0] = 1.0
    daily_returns = []

    for day in range(num_days):
        hist_window = historical_returns[:day+1]  # shape (day+1, num_stocks)
        cov_matrix = (
            np.cov(hist_window, rowvar=False) * (day + 1)
            if day > 0 else np.eye(num_stocks) * 1e-4
        )

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_stocks))
        initial_weights = np.array([1 / num_stocks] * num_stocks)

        try:
            result = sco.minimize(
                negative_sharpe_ratio,
                initial_weights,
                args=(predicted_returns[day], cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            optimal_weights = result.x if result.success and result.x is not None else initial_weights
        except Exception:
            optimal_weights = initial_weights

        realized_return = np.sum(optimal_weights * true_returns[day])
        W_t[day + 1] = W_t[day] * (1 + realized_return)
        daily_returns.append(realized_return)

    # Convert returns to np.array for metric computation
    daily_returns = np.array(daily_returns)
    cum_wealth = W_t[1:]

    # Plot cumulative wealth
    plt.figure(figsize=(10, 5))
    plt.plot(cum_wealth, label='Cumulative Wealth')
    plt.title("Final Wealth Trajectory")
    plt.xlabel("Days")
    plt.ylabel("Wealth")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Compute metrics
    avg_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    sharpe_ratio = avg_daily_return / (std_daily_return + 1e-8) * np.sqrt(252)
    cumulative_return = cum_wealth[-1] - 1
    annualized_return = avg_daily_return * 252
    annualized_volatility = std_daily_return * np.sqrt(252)

    return {
        'final_wealth': cum_wealth[-1],
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'daily_returns': daily_returns,
        'wealth_trajectory': cum_wealth
    }
