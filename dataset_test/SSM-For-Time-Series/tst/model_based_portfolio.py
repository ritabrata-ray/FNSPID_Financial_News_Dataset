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
        bounds = tuple((0, .5) for _ in range(num_stocks))
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



def online_newton_step_portfolio(returns, risk_free_rate=0.0, learning_rate=0.1, delta=1.0):
    """
    Optimize a portfolio using the Online Newton Step method to maximize Sharpe ratio.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Historical returns of assets, shape (n_samples, n_assets)
    risk_free_rate : float
        Risk-free rate (annualized)
    learning_rate : float
        Step size for the Newton update
    delta : float
        Regularization parameter
        
    Returns:
    --------
    weights_history : numpy.ndarray
        History of portfolio weights over time, shape (n_samples, n_assets)
    final_weights : numpy.ndarray
        Final optimal portfolio weights
    """
    returns = returns.T
    n_periods, n_assets = returns.shape
    
    # Initialize inverse Hessian approximation
    H = np.eye(n_assets) * delta
    
    # Initialize weights (uniform allocation)
    weights = np.ones(n_assets) / n_assets
    weights_history = np.zeros((n_periods, n_assets))
    
    # Mean returns and covariance for Sharpe ratio calculation
    mean_returns = np.zeros(n_assets)
    second_moment = np.zeros((n_assets, n_assets))
    
    # Run ONS for each time period
    for t in range(n_periods):
        # Store current weights
        weights_history[t] = weights
        
        # Portfolio return at time t
        port_return = np.dot(weights, returns[t])
        
        # Update running statistics
        if t > 0:
            mean_returns = (t * mean_returns + returns[t]) / (t + 1)
            outer_product = np.outer(returns[t], returns[t])
            second_moment = (t * second_moment + outer_product) / (t + 1)
            cov_matrix = second_moment - np.outer(mean_returns, mean_returns)
            
            # Gradient of negative Sharpe ratio
            portfolio_mean = np.dot(weights, mean_returns)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_var)
            
            # Calculate gradient (this is a simplified approximation)
            grad_mean = mean_returns / portfolio_std
            grad_std = -0.5 * (portfolio_mean - risk_free_rate) * np.dot(cov_matrix + cov_matrix.T, weights) / (portfolio_var * portfolio_std)
            gradient = -(grad_mean + grad_std)  # Negative because we maximize
            
            # Update Hessian approximation
            H_gradient = np.dot(H, gradient)
            H = H - np.outer(H_gradient, H_gradient) / (delta + np.dot(gradient, H_gradient))
            
            # Update weights using Newton step
            weights = weights - learning_rate * np.dot(H, gradient)
            
            # Project weights to simplex (ensure sum to 1 and non-negative)
            weights = project_simplex(weights)
    
    # Calculate final portfolio metrics
    final_mean_return = np.sum(mean_returns * weights)
    final_portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (final_mean_return - risk_free_rate) / final_portfolio_volatility
    
    print("Online Newton Step Results:")
    print(f"Final Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Final Expected Return: {final_mean_return:.4f}")
    print(f"Final Expected Volatility: {final_portfolio_volatility:.4f}")
    
    return weights_history, weights

def project_simplex(v):
    """
    Project a vector onto the probability simplex (sum to 1, all non-negative).
    """
    n = len(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    
    # Get sorted vector
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    
    # Find the index of the threshold
    rho = np.nonzero(u * np.arange(1, n+1) > cssv - 1)[0][-1]
    
    # Compute the threshold
    theta = (cssv[rho] - 1) / (rho + 1)
    
    # Project onto simplex
    w = np.maximum(v - theta, 0)
    
    return w

# Example usage
if __name__ == "__main__":
    # Example data - simulated returns for 5 assets
    np.random.seed(42)
    returns = np.random.normal(loc=[0.001, 0.002, 0.001, 0.0015, 0.0025], 
                              scale=[0.02, 0.025, 0.015, 0.018, 0.022],
                              size=(200, 5))  # 200 periods of data
    
    risk_free = 0.0  # risk-free rate
    
    weights_history, final_weights = online_newton_step_portfolio(returns, risk_free, 
                                                                 learning_rate=0.1, delta=1.0)
    
    # Display optimal portfolio allocation
    assets = [f"Asset {i+1}" for i in range(returns.shape[1])]
    for asset, weight in zip(assets, final_weights):
        print(f"{asset}: {weight:.4f}")
        
    # Plot weight evolution over time
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for i, asset in enumerate(assets):
            plt.plot(weights_history[:, i], label=asset)
        plt.title('Portfolio Weights Evolution (Online Newton Step)')
        plt.xlabel('Time Period')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")