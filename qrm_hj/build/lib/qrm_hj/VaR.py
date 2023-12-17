import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm
from . import MC
import math
import numpy as np
import sys
import pandas as pd
from numpy.linalg import eigh
import itertools
from scipy.stats import norm, t, spearmanr
import statsmodels.api as sm
from scipy.optimize import fsolve, minimize
import inspect
from sklearn.linear_model import LinearRegression
import statsmodels.graphics.tsaplots as stmplot
import matplotlib.pyplot as plt
import seaborn as sns

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

# calculate covariance matrix for data with missing values
def cov_missing(x, skipMiss=True, fun=np.cov):
    n, m = x.shape
    nMiss = np.sum([np.isnan(x[:, i]) for i in range(m)], axis=1)
    
    if np.sum(nMiss) == 0:
        return fun(x)
    
    idxMissing = [set(np.where(np.isnan(x[:, i]))[0]) for i in range(m)]
    
    if skipMiss:
        rows = set(range(n))
        for c in range(m):
            for rm in idxMissing[c]:
                rows.discard(rm)
        rows = sorted(rows)

        return fun(x[rows, :].T)
    
    else:
        out = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                rows = set(range(n))
                for c in (i, j):
                    for rm in idxMissing[c]:
                        rows.discard(rm)
                rows = sorted(rows)

                temp_out = fun(x[rows][:,[i,j]].T)
                out[i,j] = temp_out[0, 1]
        return out


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

## Option Pricing
# calculate implied volatility for GBSM
def implied_vol_gbsm(underlying, strike, ttm, rf, b, price, type="call"):
    f = lambda ivol: gbsm(underlying, strike, ttm, rf, b, ivol, type="call") - price
    result = fsolve(f,0.5)
    return result

# calculate implied volatility for American options with dividends
def implied_vol_americandiv(underlying, strike, ttm, rf, divAmts, divTimes, N, price, type="call"):
    f = lambda ivol: bt_american_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call") - price
    result = fsolve(f,0.5)
    return result
    
# Black Scholes Model for European option
def gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    d2 = d1 - ivol*np.sqrt(ttm)

    if type == "call":
        return underlying * np.exp((b-rf)*ttm) * norm.cdf(d1) - strike*np.exp(-rf*ttm)*norm.cdf(d2)
    elif type == "put":
        return strike*np.exp(-rf*ttm)*norm.cdf(-d2) - underlying*np.exp((b-rf)*ttm)*norm.cdf(-d1)
    else:
        print("Invalid type of option")
        
# binomial trees used to price American option with no dividends
def bt_american(underlying, strike, ttm, rf, b, ivol, N, otype="call"):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    if otype == "call":
        z = 1
    elif otype == "put":
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)
    optionValues = [0.0] * nNodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u**i * d**(j-i)
            optionValues[idx] = max(0, z * (price - strike))
            
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1, j+1)] + pd*optionValues[idxFunc(i, j+1)]))
    
    return optionValues[0]

# binomial trees used to price American option with dividends
def bt_american_div(underlying, strike, ttm, rf, divAmts, divTimes, ivol, N, type="call"):
    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_american(underlying, strike, ttm, rf, rf, ivol, N, otype=type)
    
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    if type == "call":
        z = 1
    elif type == "put":
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nDiv = len(divTimes)
    nNodes = nNodeFunc(divTimes[0])

    optionValues = np.zeros(len(range(nNodes)))

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u ** i * d ** (j - i)

            if j < divTimes[0]:
                # times before the dividend working backward induction
                optionValues[idx] = max(0, z * (price - strike))
                optionValues[idx] = max(optionValues[idx], df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd * optionValues[idxFunc(i, j + 1)]))
            else:
                # time of the dividend
                valNoExercise = bt_american_div(price - divAmts[0], strike, ttm - divTimes[0] * dt, rf, divAmts[1:], [t - divTimes[0] for t in divTimes[1:]], ivol, N - divTimes[0], type=type)
                valExercise = max(0, z * (price - strike))
                optionValues[idx] = max(valNoExercise, valExercise)

    return optionValues[0]

## Option Greeks
# calculate delta of options with closed-form formulas
def delta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))

    if type == "call":
        return np.exp((b - rf) * ttm) * norm.cdf(d1)
    elif type == "put":
        return np.exp((b - rf) * ttm) * (norm.cdf(d1) - 1)
    else:
        print("Invalid type of option")
        
# calculate Gamma of options with closed-form formulas
def gamma_gbsm(underlying, strike, ttm, rf, b, ivol):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    result = norm.pdf(d1) * np.exp((b - rf) * ttm) / (underlying * ivol * np.sqrt(ttm))
    return result

# calculate Vega of options with closed-form formulas
def vega_gbsm(underlying, strike, ttm, rf, b, ivol):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    result = underlying * norm.pdf(d1) * np.exp((b - rf) * ttm) * np.sqrt(ttm)
    return result

# calculate Theta of options with closed-form formulas
def theta_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    d2 = d1 - ivol*np.sqrt(ttm)

    if type == "call":
        result_call = - underlying * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm)) - (b - rf) * underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) - rf * strike * np.exp(-rf * ttm) * norm.cdf(d2)
        return result_call
    elif type == "put":
        result_put = - underlying * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm)) + (b - rf) * underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1) + rf * strike * np.exp(-rf * ttm) * norm.cdf(-d2)
        return result_put
    else:
        print("Invalid type of option")

# calculate Rho of options with closed-form formulas
def rho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    d2 = d1 - ivol*np.sqrt(ttm)

    if type == "call":
        return ttm * strike * np.exp(-rf * ttm) * norm.cdf(d2)
    elif type == "put":
        return -ttm * strike * np.exp(-rf * ttm) * norm.cdf(-d2)
    else:
        print("Invalid type of option")

# calculate Carry Rho of options with closed-form formulas
def crho_gbsm(underlying, strike, ttm, rf, b, ivol, type="call"):
    d1 = (np.log(underlying/strike) + (b+ivol**2/2)*ttm)/(ivol*np.sqrt(ttm))
    
    if type == "call":
        return ttm * underlying * np.exp((b - rf) * ttm) * norm.cdf(d1)
    elif type == "put":
        return -ttm * underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1)
    else:
        print("Invalid type of option")

