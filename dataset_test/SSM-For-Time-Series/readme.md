model_based_portfolio.py in tst directory has the SSQP based portfolio optimization method called "optimize_portfolio" for maximizing Sharpe ratio, the accumulated wealth calculator, as well as the ONS based baseline portfolio optimization function defined within it.

In the same tst directory, ssm.py has the S4 state space model.

run_8layer.py runs trains and evaluates the S4 model with both sentiments and without sentiments.

debug_8layer.py was originally written for debugging run_8layer.py, but now it can be used to generate the pearson correlation plots, and to compute the ONS final wealth accummulation number.

