import numpy as np
import scipy.stats
from scipy.optimize import minimize
import pandas as pd

def drawdown(returns_series: pd.Series) -> pd.DataFrame:
    """
    Compute drawdown
    """
    wealth_index = 1000 * (1 + returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'PreviousPeak': previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns() -> pd.DataFrame:
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    returns = pd.read_csv(
        './data/Portfolios_Formed_on_ME_monthly_EW.csv',
        header=0, index_col=0, parse_dates=True, na_values=-99.99
    )
    returns = returns[['Lo 10', 'Hi 10']]
    returns.columns = ['SmallCap', 'LargeCap']
    returns.index = pd.to_datetime(returns.index, format='%Y%m')
    returns = returns / 100
    return returns

def get_hfi_returns() -> pd.DataFrame:
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv(
        './data/edhec-hedgefundindices.csv',
        header=0, index_col=0, parse_dates=True
    )
    hfi.index = hfi.index.to_period('M')
    hfi = hfi / 100
    return hfi
    
def semi_deviation(df_returns: pd.DataFrame) -> pd.Series:
    """
    Returns the semideviation aka negative semi deviation of df_returns
    """
    is_negative = df_returns < 0
    return df_returns[is_negative].std(ddof=0)
    
def VaR_historic(df_returns, level=5):
    """
    Returns the historic Value at Risk at a speciifed level
    i.e returns the number such that "level" percent of returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(df_returns, pd.DataFrame):
        return df_returns.apply(lambda x: -np.percentile(x, level), axis=0)
    elif isinstance(df_returns, pd.Series):
        return -np.percentile(df_returns, level)
    else:
        raise TypeError("Excepted df_returns to be a pd.DataFrame or a pd.Series")
        
def VaR_gaussian(df_returns, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR
    If modified returns the semi-parametric Cornish-Fisher VaR
    """
    z = scipy.stats.norm.ppf(level / 100)
    if modified:
        s = df_returns.apply(scipy.stats.skew) \
            if isinstance(df_returns, pd.DataFrame) \
            else scipy.stats.skew(df_returns)
        k = df_returns.apply(scipy.stats.kurtosis) \
            if isinstance(df_returns, pd.DataFrame) \
            else scipy.stats.kurtosis(df_returns)
        z += (z**2 - 1) * s / 6 \
            + (z**3 - 3 * z) * k / 24 \
            - (2 * z**3 - 5 * z) * (s**2) / 36
    return -(df_returns.mean() + z * df_returns.std(ddof=0))

def CVaR_historic(df_returns, level=5):
    """
    Computes the Conditional VaR
    """
    is_beyond = df_returns < -VaR_historic(df_returns, level=level)
    return -df_returns[is_beyond].mean()

def get_ind_returns():
    """
    Load and format Ken French 30 Industry Portofolio Value Weighted Monthly Returns 
    """
    ind = pd.read_csv(
        './data/ind30_m_vw_rets.csv',
        header=0, index_col=0, parse_dates=True,
        date_parser=lambda x: pd.to_datetime(x, format='%Y%m').to_period('M')
    ) / 100
    ind.columns = ind.columns.str.strip()
    return ind

def annualized_return(r, pediods_of_year):
    """Annualizes set of returns"""
    return (1 + r).prod()**(pediods_of_year / r.shape[0]) - 1

def annualized_volatility(r, pediods_of_year):
    """Annualizes volatility of a set of returns"""
    return r.std() * np.sqrt(pediods_of_year)

def sharpe_ratio(r, Rf, pediods_of_year):
    """Annualized sharpe ratio"""
    Rf_per_period = (1 + Rf)**(1 / pediods_of_year) - 1
    excess_ret = r - Rf_per_period
    ann_ret = annualized_return(excess_ret, pediods_of_year)
    ann_vol = annualized_volatility(r, pediods_of_year)
    return ann_ret / ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portolio_volatility(weights, cov_mat):
    """
    Weights -> Vol
    """
    return np.sqrt(weights.T @ cov_mat @ weights)

def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plot 1 asset efficient frontier
    """
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontier")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatitlities": vols
    })
    ef.plot.line(x="Volatitlities", y="Returns", style=style)
    
def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0., 1.),) * n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portolio_volatility, init_guess,
                       args=(cov,), method="SLSQP",
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    return results.x

def optimal_weights(n_points, er, cov):
    """
    List of weightsto run the optimizer on to minimize the vol
    """
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights

def msr(riskfree_rate, er, cov):
    """
    riskfree_rate + ER + COV -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0., 1.),) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portolio_volatility(weights, cov)
        return -(r - riskfree_rate) / vol
    
    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(riskfree_rate, er, cov,), method="SLSQP",
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    return results.x

def gmv(cov):
    """
    Returns the weights of the Global Minimun Vol portfolio,
    given covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def plot_ef(n_points, er, cov, riskfree_rate=0., show_cml=False, show_ew=False, show_gmv=False, style=".-"):
    """
    Plot multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatitlities": vols
    })
    
    ax = ef.plot.line(x="Volatitlities", y="Returns", style=style)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portolio_volatility(w_ew, cov)
        
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=10)
        
    if show_gmv: # Global Minimun Variance portfolio
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portolio_volatility(w_gmv, cov)
        
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)
    
    if show_cml:
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portolio_volatility(w_msr, cov)

        # Add the Capital Market Line - CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]

        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=10)
    
    return ax