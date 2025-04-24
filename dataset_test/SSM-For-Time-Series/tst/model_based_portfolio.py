import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt

def optimize_portfolio(expected_returns, cov_matrix):
    N = len(expected_returns)
    
    def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
        port_return = np.sum(weights * expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_vol

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
    
    return result['x']

def backtest_portfolio(N, historical, predicted, window=60):
    D = historical.shape[1]
    daily_returns = []
    weights_list = []

    for t in range(window, D - 1):
        hist_window = historical[:, t-window:t].T  # shape (window, N)
        pred_today = predicted[:, t]               # shape (N,)
        cov_matrix = np.cov(hist_window, rowvar=False) * 252
        
        weights = optimize_portfolio(pred_today, cov_matrix)
        realized_return = np.dot(weights, historical[:, t])
        
        daily_returns.append(realized_return)
        weights_list.append(weights)

    returns = np.array(daily_returns)
    weights_matrix = np.array(weights_list)  # shape (days, N)

    # Compute metrics
    avg_daily_return = np.mean(returns)
    std_daily_return = np.std(returns)
    sharpe_ratio = avg_daily_return / std_daily_return * np.sqrt(252)
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

    # Plot portfolio weights over time
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
    num_days = true_returns.shape[0]
    num_stocks = true_returns.shape[1]
    cov_matrix = historical_returns.cov() * num_days  # Annualized covariance
    W_t = np.zeros(num_days+1)
    W_t[0] = 1.0
    for day in range(num_days):
    # Define constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(num_stocks))  # No short-selling

    # Initial guess (equal weights)
        initial_weights = np.array([1/num_stocks] * num_stocks)

    # Optimize portfolio
        result = sco.minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(predicted_returns[day], cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result['x']
    W_t[day+1] = W_t[day] * (np.sum(optimal_weights * true_returns[day]))
    return W_t[num_days] #final wealth output

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility
