# Function of the heatwave index
from __future__ import annotations
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from typing import Union


def hw_pctl(data : xr.Dataset, years : list[str], pctl: Union[int, float] = 90, var_name : str = 't2m_dailyMax') -> xr.Dataset:
    
    """
    This function calculates the heatwave percentile threshold for a given dataset, years and pctl.

    Parameters
    ----------
    data : xr.Dataset.
        Temperature Dataset where the percentile threshold should be performed (usually t2m temperature).
    years : str.
        Period to perform the percentile threshold, given the start and end year (usually [1981, 2010]).
    pctl : int or float.
        Value (integer or float) wich defines the percentile to perform (default 90).
    var_name : str
        In case of using a different dataset, with another name

    Returns
    -------
    pctl_th_ds : xr.Dataset.
        A xr.Dataset with the percentile threshold (called pctl_th as variable) for the given period.

    """
    # Data must be with the variable already selected
    # Selecting the period of interest (15 days are added at the beginning and end to avoid problems with the rolling window)
    data_period = data.sel(time=slice(datetime.datetime.strptime(str(years[0]), "%Y") - datetime.timedelta(days=15), datetime.datetime.strptime(str(years[1]), "%Y") + datetime.timedelta(days=15)))
    # Calculating the rolling mean (31 days). This is the window mean
    data_period = data_period.rolling(time=31, center=True).mean().dropna(dim='time')
    
    # Calculating the percentile for each day of the year
    pctl_th = data_period.groupby('time.dayofyear').reduce(np.percentile, dim='time', q=pctl)
    # Creating a date range considering the days. A leap year is considered to calculate the date range. 
    # time = pd.date_range(start='1980-01-01', end='1980-12-31', freq='D').strftime('%m-%d')
    # Adding the percentile to the Dataset
    pctl_th_ds = pctl_th.to_dataset()
    # Changing the name of the variable
    pctl_th_ds = pctl_th_ds.rename({var_name: 'pctl_th'})
    # Return the file
    return pctl_th_ds


def isHW_in_ds(data : xr.Dataset, pctl_th : xr.Dataset) -> xr.Dataset:
    """
    This function add a new variable to the dataset with the boolean values of the heatwave index.

    Parameters
    ----------
    data : xr.Dataset.
        Temperature Dataset where the identification of HW is performed (usually t2m temperature).
    pctl_th : xr.Dataset.
        Dataset with the corresponding threshold, based on percentile, that identifies the HW.

    Returns
    -------
    data : xr.Dataset.
        A xr.Dataset with the boolean value (called isHW) thath indicate if HW does or does not
        exist.
    """
    # Checking k if the data is a xarray dataset or a xarray dataarray
    if not isinstance(data, xr.core.dataset.Dataset):
        data = data.to_dataset()
    else:
        pass
    
    # Identifying the heatwave 
    data['isHW'] = xr.where(data['t2m_dailyMax'].groupby('time.dayofyear') > pctl_th.pctl_th, 1, 0)
    
    return data