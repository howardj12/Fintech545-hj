import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm
from . import MC
def return_calculate(prices: pd.DataFrame, method="ARITHMETIC", dateColumn="Date") -> pd.DataFrame:
    vars = prices.columns.values.tolist() #list of the column names
    nVars = len(vars)
    vars.remove(dateColumn) #remove the column of "date"
    if nVars == len(vars):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = nVars - 1
    p = np.array(prices.drop(columns=[dateColumn]))
    n = p.shape[0] #the number of rows
    m = p.shape[1] #the number of column
    p2 = np.empty((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]
    if method.upper() == "ARITHMETIC":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates = prices[dateColumn][1:]
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:,i]
    return out

#Calculate VaR using a normal distribution
def norml_var(returns, alpha = 0.05, N = 10000):
        mean = returns.mean()
        std = returns.std()
        Rt = np.random.normal(mean, std, N)
        Rt.sort()
        var = Rt[int(alpha * len(Rt))] * (-1)
        #print(-np.percentile(returns, alpha*100))
        return var, Rt

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

#Calculate VaR using a MLE fitted T distribution
def MLE_T_var(returns, alpha = 0.05, N = 10000):
    result = t.fit(returns, method="MLE")
    df = result[0]
    loc = result[1]
    scale = result[2]

    Rt = t(df, loc, scale).rvs(N)
    Rt.sort()
    var = Rt[int(alpha * len(Rt))] * (-1)
    return var, Rt

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

    np.random.seed(0)
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
