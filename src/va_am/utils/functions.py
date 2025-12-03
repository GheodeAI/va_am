# Function of the heatwave index
from __future__ import annotations
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from typing import Union


def hw_pctl(data : xr.Dataset, years : list[str], pctl: Union[int, float] = 90, window_size : int = 15) -> xr.Dataset:
    
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
    window_size: int
        Size of the rolling window.

    Returns
    -------
    pctl_th_ds : xr.Dataset.
        A xr.Dataset with the percentile threshold (called pctl_th as variable) for the given period.

    """
    # Remove February 29th
    data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))
    
    # Selecting the period with buffer
    start_date = datetime.datetime(int(years[0]), 1, 1) - datetime.timedelta(days=window_size)
    end_date = datetime.datetime(int(years[1]), 12, 31) + datetime.timedelta(days=window_size)
    temp_data = data.sel(time=slice(start_date, end_date))
    
    # Create day-of-year array
    doys = temp_data.time.dt.dayofyear.values
    years_arr = temp_data.time.dt.year.values
    
    # Initialize array for thresholds
    pctl_values = np.full(365, np.nan)
    
    # Vectorized calculation for each day of year
    for doy in range(1, 366):
        # Find all indices within ± window_size days of this DOY
        # We need to handle year boundaries properly
        mask = np.zeros_like(doys, dtype=bool)
        
        # For each year in reference period
        for year in range(int(years[0]), int(years[1]) + 1):
            # Calculate the window for this year
            year_mask = (years_arr == year)
            
            # Calculate day-of-year range (handle wrap-around)
            doy_min = doy - window_size
            doy_max = doy + window_size
            
            if doy_min < 1:
                # Window spans previous year
                mask |= (year_mask & ((doys >= doy_min + 365) | (doys <= doy_max)))
            elif doy_max > 365:
                # Window spans next year
                mask |= (year_mask & ((doys >= doy_min) | (doys <= doy_max - 365)))
            else:
                # Normal case
                mask |= (year_mask & (doys >= doy_min) & (doys <= doy_max))
        
        # Extract values and calculate percentile
        window_values = temp_data.values[mask]
        window_values = window_values[~np.isnan(window_values)]
        
        if len(window_values) > 0:
            pctl_values[doy-1] = np.percentile(window_values, pctl)
    
    # Create DataArray
    pctl_th = xr.DataArray(
        pctl_values,
        dims=['dayofyear'],
        coords={'dayofyear': range(1, 366)}
    )
    
    return pctl_th.to_dataset(name='pctl_th')


def isHW_in_ds(data : xr.Dataset, pctl_th : xr.Dataset, var_name : str = 't2m_dailyMax') -> xr.Dataset:
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
    data['isHW'] = xr.where(data[var_name].groupby('time.dayofyear') > pctl_th.pctl_th, 1, 0)
    
    return data