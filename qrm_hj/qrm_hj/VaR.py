import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm
from . import MC

def return_calculate(prices: pd.DataFrame, method="ARITHMETIC", date_col="Date") -> pd.DataFrame:
    columns = prices.columns.values.tolist()
    num_columns = len(columns)
    columns.remove(date_col)
    if num_columns == len(columns):
        raise ValueError(f"date_col: {date_col} not in DataFrame: {columns}")
    num_columns -= 1
    prices_arr = np.array(prices.drop(columns=[date_col]))
    rows, cols = prices_arr.shape
    adjusted_prices = np.empty((rows-1, cols))
    for i in range(rows-1):
        for j in range(cols):
            adjusted_prices[i,j] = prices_arr[i+1,j] / prices_arr[i,j]
    if method.upper() == "ARITHMETIC":
        adjusted_prices -= 1.0
    elif method.upper() == "LOG":
        adjusted_prices = np.log(adjusted_prices)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates_subset = prices[date_col][1:]
    output_df = pd.DataFrame({date_col: dates_subset})
    for i in range(num_columns):
        output_df[columns[i]] = adjusted_prices[:,i]
    return output_df

#Calculate VaR using a normal distribution
def var_normal(returns, significance_level=0.05, num_samples=10000):
    # Calculate the mean and standard deviation of the returns
    mean_return = returns.mean()
    std_return = returns.std()
    # Generate random samples from a normal distribution based on the historical mean and std deviation
    simulated_returns = np.random.normal(mean_return, std_return, num_samples)
    # Sort the simulated returns in ascending order
    simulated_returns.sort()
    # Compute the Value-at-Risk (VaR) at the given significance level
    VaR = -simulated_returns[int(significance_level * len(simulated_returns))]
    return VaR, simulated_returns

#Calculate VaR using a normal distribution with an Exponentially Weighted variance
def exp_w_variance(returns, w_lambda = 0.94):
    weight = np.zeros(len(returns))
    for i in range(len(returns)):
        weight[len(returns)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)    
    ret_means = returns - returns.mean()
    expo_w_var = ret_means.T @ np.diag(weight) @ ret_means
    return expo_w_var 

def norml_ew_var(returns, alpha = 0.05, N = 10000):
        mean = returns.mean()
        std = np.sqrt(exp_w_variance(returns))
        Rt = np.random.normal(mean, std, N)
        Rt.sort()
        var = Rt[int(alpha * len(Rt))] * (-1)
        #print(-np.percentile(returns, alpha*100))
        return var, Rt

def var_t(returns, significance_level=0.05, num_samples=10000):
    # Fit the returns to a t-distribution using MLE
    df, loc, scale = t.fit(returns, method="MLE")
    # Generate random samples from the fitted t-distribution
    simulated_returns = t.rvs(df, loc, scale, size=num_samples)
    # Sort the simulated returns in ascending order
    simulated_returns.sort()
    # Compute the Value-at-Risk (VaR) at the given significance level
    VaR = -simulated_returns[int(significance_level * len(simulated_returns))]
    return VaR, simulated_returns

#Calculate VaR using a fitted AR(1) model
def ar1_var(returns, alpha = 0.05, N = 10000):
    result = ARIMA(returns, order=(1, 0, 0)).fit()
    t_a = result.params[0]
    resid_std = np.std(result.resid)
    Rt = np.empty(N)
    Rt = t_a * returns[len(returns)] + np.random.normal(loc=0, scale=resid_std, size=N)
    Rt.sort()
    var = Rt[int(alpha * len(Rt))] * (-1)
    return var, Rt

#Calculate VaR using a historic simulation
def his_var(returns, alpha = 0.05):
    Rt = returns.values
    Rt.sort()
    var = Rt[int(alpha * len(Rt))] * (-1)
    return var, Rt

#Deal with portfolio
def process_portfolio_data(portfolio, prices, p_type):
    if p_type == "total":
        co_assets = portfolio.drop('Portfolio', axis = 1)
        co_assets = co_assets.groupby(["Stock"], as_index=False)["Holding"].sum()
    else:
        co_assets = portfolio[portfolio['Portfolio'] == p_type]
    dailyprices = pd.concat([prices["Date"], prices[co_assets["Stock"]]], axis=1)
    holdings = co_assets['Holding']
    portfolio_price = np.dot(prices[co_assets["Stock"]].tail(1), co_assets['Holding'])
    return portfolio_price, dailyprices, holdings

def expo_weighted_cov(ret_data,w_lambda):
    weight = np.zeros(len(ret_data))
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)
    ret_means = ret_data - ret_data.mean()
    expo_w_cov = ret_means.T.values @ np.diag(weight) @ ret_means.values
    return expo_w_cov

#Calculate VaR using MC simulation
def cal_MC_var(portfolio, prices, p_type, alpha=0.05, w_lambda=0.94, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    returns_0 = returns - returns.mean()
    np.random.seed(123)
    cov_mtx = expo_weighted_cov(returns_0, w_lambda)
    sim_ret = MC.pca_sim(cov_mtx, N, percent_explain = 1)
    sim_ret = np.add(sim_ret, returns.mean().values)
    dailyprices = dailyprices.drop('Date', axis=1)
    sim_change = np.dot(sim_ret * dailyprices.tail(1).values.reshape(dailyprices.shape[1]),holding)
    var = np.percentile(sim_change, alpha*100) * (-1)
    return var, sim_change

#Calculate VaR using Delta Normal
def cal_delta_var(portfolio, prices, p_type, alpha=0.05, w_lambda=0.94, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)

    returns = return_calculate(dailyprices).drop('Date',axis=1)
    dailyprices = dailyprices.drop('Date', axis=1)
    dR_dr = (dailyprices.tail(1).T.values * holding.values.reshape(-1, 1)) / portfolio_price
    cov_mtx = expo_weighted_cov(returns, w_lambda)
    R_std = np.sqrt(np.transpose(dR_dr) @ cov_mtx @ dR_dr)
    var = (-1) * portfolio_price * norm.ppf(alpha) * R_std
    return var[0][0]

#Calculate VaR using historic simulation
def cal_his_var(portfolio, prices, p_type, alpha=0.05, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    np.random.seed(0)
    sim_ret = returns.sample(N, replace=True)
    dailyprices = dailyprices.drop('Date', axis=1)
    sim_change = np.dot(sim_ret * dailyprices.tail(1).values.reshape(dailyprices.shape[1]),holding)
    var = np.percentile(sim_change, alpha*100) * (-1)
    return var, sim_change

def calculate_es(var, sim_data):
    return -np.mean(sim_data[sim_data <= -var])
