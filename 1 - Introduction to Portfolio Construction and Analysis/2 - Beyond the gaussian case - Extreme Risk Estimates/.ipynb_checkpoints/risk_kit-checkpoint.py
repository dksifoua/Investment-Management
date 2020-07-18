import numpy as np
import scipy.stats
import pandas as pd

def drawdown(returns_series: pd.Series) -> pd.DataFrame:
    """
    Compute drawdown
    """
    wealth_index = 1000 * (1 + returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peak) / previous_peak
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