from __future__ import annotations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12
})
import xarray as xr
import tensorflow as tf
import keras
from keras import layers
from keras import regularizers
from .utils import AutoEncoders, functions
import datetime
import json
import argparse
from itertools import groupby
import warnings
from typing import Union
import requests
import traceback
import sympy
from scipy.spatial import minkowski_distance
from pathlib import Path
import glob
import seaborn as sns
sns.set(font_scale=1.3, rc={'text.usetex' : True})
sns.set_style(style='white')

def square_dims(size:Union[int, list[int], np.ndarray[int]], ratio_w_h:Union[int,float]=1):
    """
    square_dims
    
    Function that return the needed dimensions for the plots of the encoded, given the latent dimension and the ratio between width and height.            
      
    Parameters
    ----------
    size: int of list[int]
        The latent dimension size.
    ratio_w_h: int or float
        Desired ration between width and height.
      
    Returns
    -------
    : tuple
        (width, height) dimensions for grid.
    
    Raises
    ------
    ValueError
        If input size is not positive.
        If ratio_w_h is <= 0.
    
    Notes
    -----
    - Uses sympy.divisors for efficient factor calculation
    - Always returns smaller dimension first in tuple
    - For non-integer ratios, finds closest divisor match
    """
    divs = np.array(sympy.divisors(size))
    dist_to_root = np.abs(divs-np.sqrt(size)*ratio_w_h)
    i = np.argmin(dist_to_root)
    x_size = int(divs[i])
    y_size = size//x_size
    return (x_size, y_size) if x_size < y_size else (y_size, x_size)

def standardize_dims(data: Union[xr.DataArray, xr.Dataset]):
    """
    standardize_dims
    
    Standardize dimension names to 'latitude' and 'longitude' across common variants.
    
    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data with spatial dimensions to standardize.
    
    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Data with standardized dimension names.
    
    Raises
    ------
    ValueError
        If input is not an xarray object.
    
    Notes
    -----
    - Handles common dimension name variants:
      - Latitude: 'lat', 'latitude'
      - Longitude: 'lon', 'long', 'longitude'
    - Case-insensitive matching
    - Preserves all other dimensions unchanged
    - Returns original object if no dimension renaming needed
    
    Examples
    --------
    >>> ds = xr.Dataset(coords={'LAT': [0, 1], 'LON': [0, 1]})
    >>> standardized = standardize_dims(ds)
    >>> list(standardized.dims)
    ['latitude', 'longitude']
    """
    dim_map = {}
    for dim in data.dims:
        dim_lower = dim.lower()
        if dim_lower in {'lat', 'latitude'}:
            dim_map[dim] = 'latitude'
        elif dim_lower in {'lon', 'long', 'longitude'}:
            dim_map[dim] = 'longitude'
    
    if dim_map:
        data = data.rename(dim_map)
    return data

def runAE(input_dim: Union[int, list[int]], latent_dim: int, arch: int, use_VAE: bool, with_cpu: bool, n_epochs: int, data_pred: Union[np.ndarray, list, xr.DataArray], file_save: str, verbose: bool, compile_params: dict = {}, fit_params : dict() = {}):
    """
    runAE
    
    Function that performs the AE traing.
    
    Parameters
    ----------
    input_dim: int or list of int
        Contains the shape of the input data to the keras.model.     
    latent_dim: int
        Represent the shape of the latent (code) space.                            
    arch: int
        Value that determine which model architecture sould be used to build the model.  
    use_VAE : bool
        Value that determines if the model should be a Variational Autoencoder or not.
    with_cpu: bool
        Value that determines if the cpu should be used instead of (default) gpu.
    n_epochs: int 
        The number of epochs for the keras.model.                     
    data_pred: np.ndarray
        Driver/predictor data (usually) to train the model.        
    file_save: str
        Where to save the .h5 model.
    verbose: bool
        Value that determines if the execution information should be displayed.  
    compile_params: dict
        Dictionary that contains all the parameters (avaible depending on tensorflow/keras version) to use for the .compile() function. 
    fit_params: dict
        Dictionary that contains all the parametes (avaible depending on tensorflow/keras version) to use for the .fit() function, except for epochs and verbose.
    
    Returns
    ----------                                            
    AE.encoder: keras.model.
        Keras object that correspond to the fitted encoder model.
    
    Raises
    ------
    ValueError
        If input/output dimensions mismatch.
    RuntimeError
        If GPU device unavailable when requested.
    
    Notes
    -----
    - Saves encoder/decoder as separate files
    - Generates training loss plots in ./figures/
    - Automatically creates directories if missing
    """
    verbose = 1 if verbose else 0
    if with_cpu:
        with tf.device("/cpu:0"):
            AE = AutoEncoders.AE_conv(input_dim=input_dim,latent_dim=latent_dim,arch=arch,in_channels=np.shape(data_pred)[-1],out_channels=np.shape(data_pred)[-1],VAE=use_VAE)

            AE.compile(**compile_params)

            history = AE.fit(data_pred, data_pred, epochs=n_epochs, min_delta=1e-6, patience=50, verbose=verbose, **fit_params)
    else:
        AE = AutoEncoders.AE_conv(input_dim=input_dim,latent_dim=latent_dim,arch=arch,in_channels=np.shape(data_pred)[-1],out_channels=np.shape(data_pred)[-1],VAE=use_VAE)

        AE.compile(**compile_params)

        history = AE.fit(data_pred, data_pred, epochs=n_epochs, min_delta=1e-6, patience=50, verbose=verbose, **fit_params)

    Path("./figures").mkdir(parents=True, exist_ok=True)
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'./figures/history-{file_save[9:-3]}.png')
    plt.savefig(f'./figures/history-{file_save[9:-3]}.pdf')
    plt.close()

    Path(file_save.split("/")[-2]).mkdir(parents=True, exist_ok=True)
    AE.encoder.save(file_save)
    AE.decoder.save(file_save[:-3]+"_dec_"+file_save[-3:])
    return AE.encoder

def get_AE_stats(with_cpu: bool, use_VAE: bool, AE_pre = None, AE_ind = None, pre_indust_pred: Union[list, np.ndarray] = None, indust_pred: Union[list, np.ndarray] = None, data_of_interest_pred: Union[list, np.ndarray] = None, period : str = 'both') -> Union[np.ndarray, list]:
    """
    get_AE_stats                                       
    
    Function used to obtain statistical information about the encoded data by the Autoencoder. It codifies train data based on the period and specific details of the architecture.
    
    Parameters
    ----------                                             
    use_VAE, with_cpu: bool
        Booleans values that determines if the model should be VAE, if the cpu should be used instead gpu or if the, respectively.    
    AE_pre, AE_ind: keras.model
        Encoders keras.model for pre and post industrial period.                         
    pre_indust_pred, indust_pred: list or np.ndarray
        Driver/predictor data of pre and post industrial period.                     
    data_of_interes_pred:list or np.ndarray
        Driver/predictor data of interest.    
    period: str
        Value that handle wich part of the data is used.                                   
    
    Returns
    ----------                                            
    : list or ndarray.
        A ndarray containind the data.                     
    
    Raises
    ------
    ValueError
        If invalid period specified.
        If missing required models/data for period.
    
    Notes
    -----
    - Computes absolute differences in latent space
    - Concatenates results for multi-period analyses
    - Flattens outputs to 1D array
    """
    if with_cpu:
        with tf.device("/cpu:0"):
            if use_VAE:
                if period == 'both':
                    pre_indust_pred_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                    indust_pred_encoded = np.abs(AE_pre.predict(indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                    pre_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
                    pre_indust_pred_encoded = np.abs(AE_ind.predict(pre_indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                    indust_pred_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                    ind_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
                elif period == 'pre':
                    pre_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                else:
                    ind_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
            else:
                if period == 'both':
                    pre_indust_pred_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                    indust_pred_encoded = np.abs(AE_pre.predict(indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                    pre_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
                    pre_indust_pred_encoded = np.abs(AE_ind.predict(pre_indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                    indust_pred_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                    ind_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
                elif period == 'pre':
                    pre_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                else:
                    ind_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
    else:        
        if use_VAE:
            if period == 'both':
                pre_indust_pred_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                indust_pred_encoded = np.abs(AE_pre.predict(indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                pre_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
                pre_indust_pred_encoded = np.abs(AE_ind.predict(pre_indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                indust_pred_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                ind_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
            elif period == 'pre':
                pre_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
            else:
                ind_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
        else:
            if period == 'both':
                pre_indust_pred_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                indust_pred_encoded = np.abs(AE_pre.predict(indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
                pre_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
                pre_indust_pred_encoded = np.abs(AE_ind.predict(pre_indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                indust_pred_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
                ind_encoded = np.concatenate((pre_indust_pred_encoded, indust_pred_encoded), axis=0)
            elif period == 'pre':
                pre_encoded = np.abs(AE_pre.predict(pre_indust_pred)-AE_pre.predict(data_of_interest_pred)).flatten()
            else:
                ind_encoded = np.abs(AE_ind.predict(indust_pred)-AE_ind.predict(data_of_interest_pred)).flatten()
    if period == 'both':
        res = np.concatenate((pre_encoded, ind_encoded), axis=0)
    elif period == 'pre':
        res = pre_encoded
    else:
        res = ind_encoded
    return res

def am(file_params_name: str, ident: bool, teleg: bool, save_recons: bool, teleg_file: str = '.secret.txt'):
    """
    am
    
    Main function that orchestrates the Analog Method (AM) workflow. It handles configuration loading, preprocessing, 
    analog search execution, and post-processing. Supports Telegram notifications and result saving.
    
    Parameters
    ----------
    file_params_name : str
        Path to the JSON configuration file containing analysis parameters.
    ident : bool
        Flag to enable heatwave period identification before analysis.
    teleg : bool
        Flag to enable Telegram notifications for warnings and errors.
    save_recons : bool
        Flag to save reconstructed data as NetCDF files in the './data/' directory.
    teleg_file : str, optional
        Path to Telegram credentials file (default is '.secret.txt').
    
    Raises
    ------
    OSError
        If the configuration file is not found.
    ValueError
        If invalid parameters are detected in the configuration.
    
    Notes
    -----
    - Reads parameters from JSON configuration file
    - Handles preprocessing of climate data
    - Performs analog search using specified method
    - Manages Telegram notifications if enabled
    - Saves reconstructions if enabled
    - Executes post-processing for result analysis
    
    The function coordinates the entire AM workflow including optional heatwave identification,
    data preprocessing, model training, analog search, and result post-processing.
    
    Returns
    ----------
    """
    # Teleg
    token = None
    chat_id = None
    user_name = None
    if teleg:
        with open(teleg_file) as f:
            token = f.readline().strip()
            chat_id = f.readline().strip()
            user_name = f.readline().strip()
        f.close()
    # Read params
    try:
        file_params = open(file_params_name)
    except:
        message = OSError(f'File {file_params_name} not found. To identify the Heat wave period a configuration parameters file is needed.')
        if teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
            print(requests.get(url).json())
        raise message
    else:
        params = json.load(file_params)
        file_params.close()
    
    # Perform preprocessing
    params, img_size, data_pred, data_target, time_pre_indust_pred, time_indust_pred, data_of_interest_pred, data_of_interest_target, x_train_pre_pred, x_train_ind_pred, x_test_pre_pred, x_test_ind_pred, pre_indust_pred, pre_indust_target, indust_pred, indust_target = perform_preprocess(params) 
    
    # Call analogSearch
    ## Obtain stats
    ## Stats analog
    if params["enhanced_distance"]:
        if params["period"] == 'both':
            stat_data = np.concatenate(((x_train_ind_pred-data_of_interest_pred).flatten(),(x_train_pre_pred-data_of_interest_pred).flatten()),axis=0)
        elif params["period"] == 'pre':
            stat_data = (x_train_pre_pred-data_of_interest_pred).flatten()
        else:
            stat_data = (x_train_ind_pred-data_of_interest_pred).flatten()
        stat_mean = np.abs(stat_data).mean()
        stat_std = np.abs(stat_data).std()
        stat_max = np.abs(stat_data).max()
        stat_min = np.abs(stat_data).min()
        print(f'Mean analog: {stat_mean}')
        print(f'Std analog: {stat_std}')
        print(f'Max anlog: {stat_max}')
        print(f'Min analog: {stat_min}')
        print(f'len analog: {len(stat_data)}')
        print(f'len th analog: {len(stat_data[np.abs(stat_data) < (stat_mean-0.3*stat_std)])}')

    ## Stats AE
    if params["enhanced_distance"]:
        if params["period"] == 'both':
            encoded = get_AE_stats(params["with_cpu"], params["use_VAE"], AE_pre, AE_ind, x_train_pre_pred, x_train_ind_pred, data_of_interest_pred)
        elif params["period"] == 'pre':
            encoded = get_AE_stats(with_cpu=params["with_cpu"], use_VAE=params["use_VAE"], AE_pre=AE_pre, pre_indust_pred=x_train_pre_pred, data_of_interest_pred=data_of_interest_pred, period=params["period"])
        else:
            encoded = get_AE_stats(with_cpu=params["with_cpu"], use_VAE=params["use_VAE"], AE_ind=AE_ind, indust_pred=x_train_ind_pred, data_of_interest_pred=data_of_interest_pred, period=params["period"])
        print(f'Mean: {encoded.mean()}')
        print(f'Std: {encoded.std()}')
        print(f'Max AE: {encoded.max()}')
        print(f'Min AE: {encoded.min()}')
        print(f'len AE: {len(encoded)}')
        print(f'len th AE: {len(encoded[encoded < (encoded.mean() - 0.3*encoded.std())])}')

    ## This is the threshold of difference between driver/predictor maps and target
    ## to be acepted as low difference
    ## Only used for the local proximity of enhanced distance
    threshold = 0
    threshold_AE = 0
    if params["enhanced_distance"]:
        threshold = stat_mean-0.3*stat_std
        threshold_AE = encoded.mean()-0.3*encoded.std()
    
    Path("./comparison-csv").mkdir(parents=True, exist_ok=True)

    if params["period"] in ['both', 'pre']:
        file_time_name = f'./comparison-csv/analogues-pre-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.npy'.replace(" ","").replace("'", "").replace(",","")
        analog_pre = analogSearch(params["p"], params["k"], pre_indust_pred, data_of_interest_pred, time_pre_indust_pred, pre_indust_target, params["enhanced_distance"], threshold=threshold, img_size=img_size, iter=params["iter"], replace_choice=params["replace_choice"], target_var_name=params["target_var_name"], file_time_name=file_time_name)

    # Analog Post
    if params["period"] in ['both', 'post']:
        file_time_name = f'./comparison-csv/analogues-post-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.npy'.replace(" ","").replace("'", "").replace(",","")
        analog_ind = analogSearch(params["p"], params["k"], indust_pred, data_of_interest_pred, time_indust_pred, indust_target, params["enhanced_distance"], threshold=threshold, img_size=img_size, iter=params["iter"], replace_choice=params["replace_choice"], target_var_name=params["target_var_name"], file_time_name=file_time_name)
    
    post_process(file_params_name, is_atribution = period == 'both')
    message = f'Post process of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
    if teleg:
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
        requests.get(url).json()
    warnings.warn(message)
    return

def analogSearch(p:int, k: int, data_pred: Union[list, np.ndarray], data_of_interest_pred: Union[list, np.ndarray], time_pred: xr.DataArray, data_target: xr.Dataset, enhanced_distance:bool, threshold: Union[int, float], img_size: Union[list, np.ndarray], iter: int, threshold_offset_counter: int = 20, replace_choice: bool = True, target_var_name : str = 'air', file_time_name: str = 'analogues.npy') -> tuple:
    """
    analogSearch                                       
    
    Funtion that performs the Analog Search Method for a given diver/predictor and target variable.                                  
    
    Parameters
    ----------
    p: int
        The p-order of Minskowski distance to perform.
    k: int
        Number of near neighbours to search.            
    data_pred: list or ndarray
        Driver/predictor data where to search.
    data_of_interes_pred: list or ndarray
        Driver/predictor data to be searched.
    time_pred: DataArray
        Time DataArray corresponding to the driver/predictor data where is searching.
    data_target: Dataset
        Target Dataset Dataset used to check the target value.
    enhanced_distance: bool
        Flag that decides if local proximity has to be performed or no.
    threshold: int or float
        Threshold used in analogSearch to compute local proximity.
    img_size: list or ndarray
        List that determine the size of the driver/predictor and target images.
    iter: int
        How many random neighbours to select.
    threshold_offset_counter: int
        Number used to perform the local proximity. Default 20.
    replace_choice: bool
        Flag that indicates if iter selected can be replaced.
    target_var_name: str
        The name of the Target Dataset variable in case of working with different Dataset.
    file_time_name: str
        The name of the file where to save the found analogues.
    
    Returns
    ----------
    : tuple
        A tuple containing selected driver/predictor and target.
    
    Raises
    ------
    ValueError
        If k exceeds available analogs.
        If p <= 0.
    
    Notes
    -----
    - Uses scipy.spatial.minkowski_distance
    - Saves selected analog times to numpy file
    - Handles both encoded and raw predictor data
    """
    is_not_encoded = (len(np.shape(data_pred)) == 4)

    d_i_s = np.shape([data_of_interest_pred])
    d_of_i = np.reshape([data_of_interest_pred], (d_i_s[0], int(np.size(data_of_interest_pred)/d_i_s[0])))
    d_p_s = np.shape(data_pred)
    d_p = np.reshape(data_pred, (d_p_s[0],int(np.size(data_pred)/d_p_s[0])))
    dist = minkowski_distance(d_of_i, d_p, p=p)

    if enhanced_distance:
        data_diff = np.abs(data_pred-data_of_interest_pred)
        d = np.zeros_like(data_diff)
        d[data_diff <= threshold] = -1
        ax_to_sum = np.arange(len(np.shape(data_diff)))[1:]
        dist += np.sum(d, axis=tuple(ax_to_sum)) + threshold_offset_counter

    minindex = dist.argsort()
    time_prediction = time_pred[minindex[:k]]
    prediction = data_target.sel(time=time_prediction[:k])[target_var_name].data
    predf = data_pred[minindex[:k]]

    idx = np.random.choice(np.arange(predf.shape[0]), size=iter, replace=replace_choice)
    
    selected_target = prediction[idx,:,:]
    #print(f'\nTime prediction: \n {np.shape(time_prediction)} \n {time_prediction}')
    selected_time = time_prediction[idx]
    np.save(file_time_name, selected_time)
    if is_not_encoded:
        selected_psr = predf[idx,:,:,:]
    else:
        selected_psr = predf[idx,:]

    return selected_psr, selected_target, selected_time

def calculate_interest_region(interest_region: Union[list, np.ndarray], dims_list: int, resolution: Union[int, float, str] = 2, is_teleg: bool = False, secret_file:str = './secret.txt') -> list:
    """
    calculate_interest_region
    
    Method which transform latitude/longitude degrees to index. It is used to increase the speed of the methods by using numpy arrays insted of Dataset or DataArray.
    
    Parameters
    ----------
    interest_region: list or ndarray
        List which contains the latitude and longitude degrees to be converted as index. 
    dims_list: list of int
        List that contain, in this order, the minimum latitude, maximum latitude, minumin longitude, maximum longitude. When resolution is 'auto', dims_list should be a tuple with (latitude, longitude).
    resolution: int, float or str
        Degrees resolution employed. Default value is 2º. If resolution is 'auto' it will infer automatically the resolution (useful when resolution is not constant along the dimensions)
    is_teleg: bool
        Flag that indicate if the warnings have to be sent to Telegram or not.
    secret_file: str
        Auxiliar variable only needed if is_teleg True to read token and chat_id values.
    
    Returns
    ----------
    new_interest_region: list
        A list that contains the equivalent index values, as [lat_start_idx, lat_end_idx, lon_start_idx, lon_end_idx]
    
    Raises
    ------
    ValueError
        If resolution='auto' without coordinate data.
        If interest_region outside domain bounds.
    
    Notes
    -----
    - Handles both 0-360 and -180-180 longitude systems
    - Automatically adjusts out-of-bounds regions
    - Uses np.isclose for coordinate matching with 'auto' resolution
    """
    token = None
    chat_id = None
    if is_teleg:
        with open(secret_file) as f:
            token = f.readline().strip()
            chat_id = f.readline().strip()
            user_name = f.readline().strip()
        f.close()

    
    if resolution=='auto':
        new_interest_region = np.zeros(4)
        lat, lon = dims_list
        new_interest_region[0] = np.where(np.isclose(interest_region[0], lat))[0][0]
        new_interest_region[1] = np.where(np.isclose(interest_region[1], lat))[0][0]
        new_interest_region[2] = np.where(np.isclose(interest_region[2], lon))[0][0]
        new_interest_region[3] = np.where(np.isclose(interest_region[3], lon))[0][0] + 1
        new_interest_region = new_interest_region.astype(int).tolist()
    else:
        new_interest_region = []
        for idx, elem in enumerate(interest_region):
            if idx%2==0:
                if elem < dims_list[idx]:
                    message = 'Your interest region is out of lat or lon minimum limits. I correct it, but be aware'
                    if is_teleg:
                        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                        requests.get(url).json()
                    warnings.warn(message, stacklevel=2)
                new_elem = int(max((elem - dims_list[idx]) // resolution, 0))
                new_interest_region.append(new_elem)
            else:
                if elem > dims_list[idx]:
                    message = 'Your interest region is out of lat or lon maximum limits. I correct it, but be aware'
                    if is_teleg:
                        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                        requests.get(url).json()
                    warnings.warn(message, stacklevel=2)
                new_elem = int(min((elem - dims_list[idx-1]) // resolution , (dims_list[idx] - dims_list[idx-1]) // resolution) + 1)
                new_interest_region.append(new_elem)
    return new_interest_region

def save_reconstruction(params: dict, reconstructions_Pre_Analog: list, reconstructions_Post_Analog: list, reconstructions_Pre_AE: list, reconstructions_Post_AE: list):
    """
    save_reconstruction
    
    Method that save the target reconstruction based on the runs maded. It do not return anything, only save the Xarray Datasets on the corresponding file on data folder. Each file have the format [name]-[period]-[method]-[time].nc.

    Parameters
    ----------
    params: dict
        A dictionary which contains all the needed parameters and configuration. Mainly loaded from the configuration file, with some auxiliar parameters added by other functions.
    reconstruction_Pre_Analog: list
        A list with the multiple reconstructed pre-industrial data by the Analog Method, for each day (or week).
    reconstruction_Post_Analog: list
        A list with the multiple reconstructed post-industrial data by the Analog Method, for each day (or week).
    reconstruction_Pre_AE: list
        A list with the multiple reconstructed pre-industrial data by the AutoEncoder, for each day (or week).
    reconstruction_Post_AE: list
        A list with the multiple reconstructed post-industrial data by the AutoEncoder, for each day (or week).

    Returns
    ----------
    
    Raises
    ------
    IOError
        If NetCDF writing fails.
    
    Notes
    -----
    - Saves files to ./data/ directory
    - Uses xarray for NetCDF export
    - Filenames include execution timestamp
    - Averages multiple reconstructions
    """
    current = datetime.datetime.now()
    int_reg = params["interest_region"]
    resolution = params["resolution"]
    Path("./data").mkdir(parents=True, exist_ok=True)
    if params["save_recons"]:
        if params["period"] in ["both", "pre"]:
            Path("./data").mkdir(parents=True, exist_ok=True)
            reconstruction_Pre_Analog = np.mean(reconstructions_Pre_Analog, axis=0)
            print('Size Recons Pre Analog: ', np.size(reconstruction_Pre_Analog))
            xr_Pre_Analog = xr.Dataset(data_vars=dict(y=(["reconstruction", "latitude", "longitude"], reconstruction_Pre_Analog)),
                                    coords=dict(reconstruction = np.arange(params["iter"]),
                                                latitude = np.arange(int_reg[0], int_reg[1]+resolution, resolution),
                                                longitude = np.arange(int_reg[2], int_reg[3]+resolution, resolution)
                                                ))
            xr_Pre_Analog.to_netcdf(f'./data/reconstruction-{params["season"]}{params["name"]}x{params["iter"]}-Pre-AM-{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.nc'.replace(" ","").replace("'", "").replace(",",""))
        if params["period"] in ["both", "post"]:
            Path("./data").mkdir(parents=True, exist_ok=True)
            reconstruction_Post_AE = np.mean(reconstructions_Post_AE, axis=0)
            print('Size Recons Post AE: ', np.size(reconstruction_Post_AE))
            xr_Post_AE = xr.Dataset(data_vars=dict(y=(["reconstruction", "latitude", "longitude"], reconstruction_Post_AE)),
                                    coords=dict(reconstruction = np.arange(params["iter"]),
                                                latitude = np.arange(int_reg[0], int_reg[1]+resolution, resolution),
                                                longitude = np.arange(int_reg[2], int_reg[3]+resolution, resolution)
                                            ))
            xr_Post_AE.to_netcdf(f'./data/reconstruction-{params["season"]}{params["name"]}x{params["iter"]}-Post-AE-AM-{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.nc'.replace(" ","").replace("'", "").replace(",",""))
            
            reconstruction_Post_Analog = np.mean(reconstructions_Post_Analog, axis=0)
            xr_Post_Analog = xr.Dataset(data_vars=dict(y=(["reconstruction", "latitude", "longitude"], reconstruction_Post_Analog)),
                                        coords=dict(reconstruction = np.arange(params["iter"]),
                                                    latitude = np.arange(int_reg[0], int_reg[1]+resolution, resolution),
                                                    longitude = np.arange(int_reg[2], int_reg[3]+resolution, resolution)
                                                   ))
            xr_Post_Analog.to_netcdf(f'./data/reconstruction-{params["season"]}{params["name"]}x{params["iter"]}-Post-AM-{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.nc'.replace(" ","").replace("'", "").replace(",",""))
            


def perform_preprocess(params: dict) -> tuple:
    """
    perform_preprocess
    
    Method that perform the preprocessing stage
    
    Parameters
    ----------
    params: dict
        A dictionary with needed parameters and configuration. Mainly loaded from the configuration file, with some auxiliar parameters added by other functions.
    
    Returns
    ----------
    : tuple
        A tuple of all needed data.
    
    Raises
    ------
    FileNotFoundError
        If input datasets missing.
    ValueError
        If invalid time ranges.
    
    Notes
    -----
    - Handles multiple time resolutions (daily/weekly/monthly)
    - Normalizes data per variable
    - Converts xarray DataArrays to numpy arrays
    - Manages train/test splits
    """
    # Set teleg
    is_teleg = False
    token = None
    chat_id = None
    if "teleg" in params:
        if params["teleg"]:
            is_teleg = True
            with open(params["secret_file"]) as f:
                token = f.readline().strip()
                chat_id = f.readline().strip()
                user_name = f.readline().strip()
            f.close()

    # Check period
    if params["period"] not in ['both', 'pre', 'post']:
        params["period"] = 'both'
        message = f'I do not understand the period {params["period"]}. I correct it to both, but be aware. Please check \n \t\t python va_am.py -h \n to see correct posible values for -p   --period.'
        if is_teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
            requests.get(url).json()
        warnings.warn(message, stacklevel=2)

    # Default values
    if not "target_dataset" in params:
        params["target_dataset"] = 'data/air.sig995.day.nc'
    if not "pred_dataset" in params:
        params["pred_dataset"] = 'data/prmsl.day.mean.nc'
    if not "interest_dataset" in params:
        params["interest_dataset"] = params["target_dataset"]
    interest_isnot_target = params["interest_dataset"] != params["target_dataset"]

    # Load data
    target=xr.load_dataset(params["target_dataset"], engine='netcdf4')   #Target Dataset
    pred=xr.load_dataset(params["pred_dataset"], engine='netcdf4')    #Predictors Dataset
    if "time_bnds" in list(pred.data_vars):
        pred = pred.drop("time_bnds")
    if interest_isnot_target:
        interest = xr.load_dataset(params["interest_dataset"], engine='netcdf4') # Target Dataset for Interest Event 
        pred_interest = xr.load_dataset(params["pred_interest_dataset"], engine='netcdf4') # Predictors Dataset for Interest Event
        if "time_bnds" in list(pred_interest.data_vars):
            pred_interest = pred_interest.drop("time_bnds")
    if params["verbose"]:
        print('Data loaded')

    # Check dims names
    if "latitude" not in target.dims or "longitude" not in target.dims:
        target = standardize_dims(target)
    if "latitude" not in pred.dims or "longitude" not in pred.dims:
        pred = standardize_dims(pred)
    if interest_isnot_target:
        if "latitude" not in interest.dims or "longitude" not in interest.dims:
            interest = standardize_dims(interest)
        if "latitude" not in pred_interest.dims or "longitude" not in pred_interest.dims:
            pred_interest = standardize_dims(pred_interest)
    if params["verbose"]:
        print('Checked dims names')

    # Attribute names
    if not "target_var_name" in params:
        params["target_var_name"] = list(target.data_vars)[0]
    if not "pred_var_name" in params:
        params["pred_var_name"] = list(pred.data_vars)
    if not "interest_var_name" in params:
        params["interest_var_name"] = params["target_var_name"]
    if not "pred_interest_var_name" in params:
        if interest_isnot_target: 
            params["pred_interest_var_name"] = list(pred_interest.data_vars)
        else:
            params["pred_interest_var_name"] = params["pred_var_name"]
    

    if params["target_var_name"] == "t2m_dailyMax":
        params["pre_init"] = str(target.time.data[0].astype('datetime64[D]'))
    
    # Preprocess
    ## Change longitude coordinates from 0 - 359 to -180 - 179
    if target.longitude.min() >= 0:
        target = target.assign_coords(longitude=(((target.longitude + 180) % 360) - 180))
    target = target.sortby(target.longitude)
    target = target.sortby(target.latitude)
    if interest_isnot_target:
        if interest.longitude.min() >= 0:
            interest = interest.assign_coords(longitude=(((interest.longitude + 180) % 360) - 180))
        interest = interest.sortby(interest.longitude)
        interest = interest.sortby(interest.latitude)
        if pred_interest.longitude.min() >= 0:
            pred_interest = pred_interest.assign_coords(longitude=(((pred_interest.longitude + 180) % 360) - 180))
        pred_interest = pred_interest.sortby(pred_interest.longitude)
        pred_interest = pred_interest.sortby(pred_interest.latitude)
    if pred.longitude.min() >= 0:
        pred = pred.assign_coords(longitude=(((pred.longitude + 180) % 360) - 180))
    pred = pred.sortby(pred.longitude)
    pred = pred.sortby(pred.latitude)
    if params["verbose"]:
        print('Coord changed')

    ## Season selection
    if params["season"] == "winter":
        season_months = [12, 1, 2]
    elif params["season"] == "spring":
        season_months = [3, 4, 5]
    elif params["season"] == "summer":
        season_months = [6, 7, 8]
    elif params["season"] == "autumn":
        season_months = [9, 10, 11]
    elif params["season"] == "autumn-winter":
        season_months = [10, 11, 12, 1, 2, 3]
    elif params["season"] == "spring-summer":
        season_months = [4, 5, 6, 7, 8, 9]
    else:
        season_months = list(range(1, 13))

    ## Load Target Dataset
    data_target = target.sel(latitude=slice(params["latitude_min"],params["latitude_max"]),longitude=slice(params["longitude_min"],params["longitude_max"]))
    ### Pre-Industrial data
    pre_indust_target = None
    if params["period"] in ['both', 'pre']:
        pre_indust_target = data_target.sel(time=slice(params["pre_init"],params["pre_end"]))
    ### Industrial (or Post-Industrial) data
    if params["period"] in ['both', 'post']:
        indust_target = data_target.sel(time=slice(params["post_init"],params["post_end"]))
    else:
        indust_target = data_target.copy()

    ## Load Interest if not Target
    if interest_isnot_target:
        interest = interest.sel(latitude=slice(params["latitude_min"],params["latitude_max"]),longitude=slice(params["longitude_min"],params["longitude_max"]))
        pred_interest = pred_interest.sel(latitude=slice(params["latitude_min"],params["latitude_max"]),longitude=slice(params["longitude_min"],params["longitude_max"]))

    ## Load Driver/predictor
    data_pred = pred.sel(latitude=slice(params["latitude_min"],params["latitude_max"]),longitude=slice(params["longitude_min"],params["longitude_max"]))
    ### Pre-Industrial data
    pre_indust_pred = None
    if params["period"] in ['both', 'pre']:
        pre_indust_pred = data_pred.sel(time=slice(params["pre_init"],params["pre_end"]))
    ### Industrial (or Post-Industrial) data
    if params["period"] in ['both', 'post']:
        indust_pred = data_pred.sel(time=slice(params["post_init"],params["post_end"]))
    else:
        indust_pred = data_pred.copy()

    ## Calculate image size
    img_size = data_target.dims.get('longitude')*data_target.dims.get('latitude')
    
    if params["verbose"]:
        print('Data splited by epoch')

    ## Mean over week
    if params["per_what"] == "per_week":
        if params["period"] in ['both', 'pre']:
            pre_indust_target = pre_indust_target.resample(time="7D").mean()
            pre_indust_pred = pre_indust_pred.resample(time="7D").mean()
        
        indust_target = indust_target.resample(time="7D").mean()
        indust_pred = indust_pred.resample(time="7D").mean()
        if interest_isnot_target:
            interest = interest.resample(time="7D").mean()
            pred_interest = pred_interest.resample(time="7D").mean()

    ## Mean over month
    if params["per_what"] == "per_month":
        if params["period"] in ['both', 'pre']:
            pre_indust_target = pre_indust_target.resample(time="MS", closed="left", label="left").mean()
            pre_indust_pred = pre_indust_pred.resample(time="MS", closed="left", label="left").mean()
        
        indust_target = indust_target.resample(time="MS", closed="left", label="left").mean()
        indust_pred = indust_pred.resample(time="MS", closed="left", label="left").mean()
        if interest_isnot_target:
            interest = interest.resample(time="MS", closed="left", label="left").mean()
            pred_interest = pred_interest.resample(time="MS", closed="left", label="left").mean()

    elif params["per_what"] != "per_day":
        message = ValueError(f'Per what? What is {params["pre_what"]} supposed to be? For now I only understand per_week and per_day')
        if is_teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
            requests.get(url).json()
        raise message
    
    if interest_isnot_target:
        data_of_interest_target = interest.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))
        data_of_interest_pred = pred_interest.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))
    else:
        data_of_interest_target = indust_target.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))
        data_of_interest_pred = indust_pred.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))
    
    post_init_limit = params["post_init"]
    if type(post_init_limit) == str:
        post_init_limit = datetime.datetime.strptime(params["post_init"], "%Y-%m-%d")
    post_end_limit = params["post_end"]
    if type(post_end_limit) == str:
        post_end_limit = datetime.datetime.strptime(params["post_end"], "%Y-%m-%d")
    pre_init_limit = params["pre_init"]
    if type(pre_init_limit) == str:
        pre_init_limit = datetime.datetime.strptime(params["pre_init"], "%Y-%m-%d")
    pre_end_limit = params["pre_end"]
    if type(pre_end_limit) == str:
        pre_end_limit = datetime.datetime.strptime(params["pre_end"], "%Y-%m-%d")
    if params["remove_year"]:
        if params["period"] in ["all", "post"]:
            if params["data_of_interest_init"] > post_init_limit and params["data_of_interest_init"] < post_end_limit:
                indust_target = indust_target.drop_sel(time=(indust_target.sel(time=slice(str(params["data_of_interest_init"].year),str(params["data_of_interest_end"].year)))).get_index('time'))
                indust_pred = indust_pred.drop_sel(time=(indust_pred.sel(time=slice(str(params["data_of_interest_init"].year),str(params["data_of_interest_end"].year)))).get_index('time'))
        if params["period"] in ["all", "pre"]:
            if params["data_of_interest_init"] > pre_init_limit and params["data_of_interest_init"] < pre_end_limit:
                pre_indust_target = pre_indust_target.drop_sel(time=(pre_indust_target.sel(time=slice(str(params["data_of_interest_init"].year),str(params["data_of_interest_end"].year)))).get_index('time'))
                pre_indust_pred = pre_indust_pred.drop_sel(time=(pre_indust_pred.sel(time=slice(str(params["data_of_interest_init"].year),str(params["data_of_interest_end"].year)))).get_index('time'))        
    else:
        if params["period"] in ["all", "post"]:
            if params["data_of_interest_init"] > post_init_limit and params["data_of_interest_init"] < post_end_limit:
                indust_target = indust_target.drop_sel(time=(indust_target.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))).get_index('time'))
                indust_pred = indust_pred.drop_sel(time=(indust_pred.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))).get_index('time'))
        if params["period"] in ["all", "pre"]:
            if params["data_of_interest_init"] > pre_init_limit and params["data_of_interest_init"] < pre_end_limit:
                pre_indust_target = pre_indust_target.drop_sel(time=(pre_indust_target.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))).get_index('time'))
                pre_indust_pred = pre_indust_pred.drop_sel(time=(pre_indust_pred.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"]))).get_index('time'))
    del post_init_limit, post_end_limit, pre_init_limit, pre_end_limit
    

    if params["verbose"]:
        print('Mean over week')

    ## Season split
    if params["period"] in ['both', 'pre']:
        pre_indust_target = pre_indust_target.sel(time=pre_indust_target.time.dt.month.isin(season_months))
        pre_indust_pred = pre_indust_pred.sel(time=pre_indust_pred.time.dt.month.isin(season_months))

    indust_target = indust_target.sel(time=indust_target.time.dt.month.isin(season_months))
    indust_pred = indust_pred.sel(time=indust_pred.time.dt.month.isin(season_months))

    if params["verbose"]:
        print('Season split')

    # Train/Test split & normalization
    ## Train/test split for driver/predictor
    x_train_pre_pred = None
    x_test_pre_pred = None
    if params["period"] in ['both', 'pre']:
        x_train_pre_pred = pre_indust_pred.isel(time=slice(0,int(pre_indust_pred.dims.get('time')*0.75)))
        x_test_pre_pred = pre_indust_pred.isel(time=slice(int(pre_indust_pred.dims.get('time')*0.75), int(pre_indust_pred.dims.get('time'))))

    x_train_ind_pred = indust_pred.isel(time=slice(0,int(indust_pred.dims.get('time')*0.75)))
    x_test_ind_pred = indust_pred.isel(time=slice(int(indust_pred.dims.get('time')*0.75), indust_pred.dims.get('time')))

    ## Get labels for time
    time_pre_indust_pred = None
    if params["period"] in ['both', 'pre']:
        time_pre_indust_pred = pre_indust_pred.time
    time_indust_pred = indust_pred.time

    if params["verbose"]:
        print('Train/Test split')
    
    ## Normalization
    if params["period"] in ['both', 'pre']:
        x_train_pre_norm_pred = x_train_pre_pred.copy()
        x_test_pre_norm_pred = x_test_pre_pred.copy()
        pre_indust_norm_pred = pre_indust_pred.copy()
    x_train_ind_norm_pred = x_train_ind_pred.copy()
    x_test_ind_norm_pred = x_test_ind_pred.copy()
    data_of_interest_norm_pred = data_of_interest_pred.copy()
    indust_norm_pred = indust_pred.copy()
    for pred_var, pred_interest_var in zip(params["pred_var_name"], params["pred_interest_var_name"]):
        if params["period"] == 'both':
            min_scale_pred = np.min(np.array([x_train_ind_pred[pred_var].min(), x_train_pre_pred[pred_var].min()]))
            norm_scale_pred = np.max(np.array([x_train_ind_pred[pred_var].max(), x_train_pre_pred[pred_var].max()])) - min_scale_pred
        elif params["period"] == 'post':
            min_scale_pred = np.min(np.array([x_train_ind_pred[pred_var].min()]))
            norm_scale_pred = np.max(np.array([x_train_ind_pred[pred_var].max()])) - min_scale_pred
        else:
            min_scale_pred = np.min(np.array([x_train_pre_pred[pred_var].min()]))
            norm_scale_pred = np.max(np.array([x_train_pre_pred[pred_var].max()])) - min_scale_pred

        if params["period"] in ['both', 'pre']:
            x_train_pre_norm_pred[pred_var].data = (x_train_pre_pred[pred_var].data - min_scale_pred) / norm_scale_pred
            x_test_pre_norm_pred[pred_var].data = (x_test_pre_pred[pred_var].data - min_scale_pred) / norm_scale_pred
            pre_indust_norm_pred[pred_var].data = (pre_indust_pred[pred_var].data - min_scale_pred) / norm_scale_pred    
        
        x_train_ind_norm_pred[pred_var].data = (x_train_ind_pred[pred_var].data - min_scale_pred) / norm_scale_pred
        x_test_ind_norm_pred[pred_var].data = (x_test_ind_pred[pred_var].data - min_scale_pred) / norm_scale_pred
        data_of_interest_norm_pred[pred_interest_var].data = (data_of_interest_pred[pred_interest_var].data - min_scale_pred) / norm_scale_pred
        indust_norm_pred[pred_var].data = (indust_pred[pred_var].data - min_scale_pred) / norm_scale_pred

    if params["verbose"]:
        print('Normalization')

    ## Reshape
    if params["period"] in ['both', 'pre']:
        x_train_pre_pred = x_train_pre_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()
        x_test_pre_pred = x_test_pre_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()
        pre_indust_pred = pre_indust_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()
    x_train_ind_pred = x_train_ind_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()
    x_test_ind_pred = x_test_ind_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()
    data_of_interest_pred = data_of_interest_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()
    indust_pred = indust_norm_pred.to_array().transpose("time", "latitude", "longitude", "variable").to_numpy()

    if params["arch"]==8:
        if params["period"] in ['both', 'pre']:
            x_train_pre_pred = np.mean(x_train_pre_pred, axis=3)
            x_test_pre_pred = np.mean(x_test_pre_pred, axis=3)
            pre_indust_pred = np.mean(pre_indust_pred, axis=3)
        x_train_ind_pred = np.mean(x_train_ind_pred, axis=3)
        x_test_ind_pred = np.mean(x_test_ind_pred, axis=3)
        data_of_interest_pred = np.mean(data_of_interest_pred, axis=3)
        indust_pred = np.mean(indust_pred, axis=3)
    d_return = {
        "params": params,
        "img_size": img_size,
        "data_pred": data_pred,
        "data_target": data_target,
        "time_pre_indust_pred": time_pre_indust_pred,
        "time_indust_pred": time_indust_pred,
        "data_of_interest_pred": data_of_interest_pred,
        "data_of_interest_target": data_of_interest_target,
        "x_train_pre_pred": x_train_pre_pred,
        "x_train_ind_pred": x_train_ind_pred,
        "x_test_pre_pred": x_test_pre_pred,
        "x_test_ind_pred": x_test_ind_pred,
        "pre_indust_pred": pre_indust_pred,
        "pre_indust_target": pre_indust_target,
        "indust_pred": indust_pred,
        "indust_target": indust_target
    }

    if not "out_preprocess" in params.keys():
        params["out_preprocess"] = "all"

    if params["out_preprocess"] != "all":
        return tuple(map(d_return.get, params["out_preprocess"]))
    else:
        return params, img_size, data_pred, data_target, time_pre_indust_pred, time_indust_pred, data_of_interest_pred, data_of_interest_target, x_train_pre_pred, x_train_ind_pred, x_test_pre_pred, x_test_ind_pred, pre_indust_pred, pre_indust_target, indust_pred, indust_target
    
    

def runComparison(params: dict)-> tuple:
    """
    runComparison                                      
    
    Method that perform the preprocessing, use of the others previous methods, and comparison between analogSearch and AE + analogSearch.                
    
    Parameters
    ----------
    params: dict
        A dictionary with needed parameters and configuration. Mainly loaded from the configuration file, with some auxiliar parameters added by other functions.
    
    Returns
    ----------
    : tuple
        A tuple of 4 elemets, each containing the corresponding reconstructions list data.
    
    Raises
    ------
    RuntimeError
        If model loading fails.
    
    Notes
    -----
    - Generates comparison CSV files
    - Produces KDE plots of results
    - Handles both CPU/GPU execution
    - Supports multiple latent dimensions
    """
    # Set teleg
    is_teleg = False
    token = None
    chat_id = None
    if "teleg" in params:
        if params["teleg"]:
            is_teleg = True
            with open(params["secret_file"]) as f:
                token = f.readline().strip()
                chat_id = f.readline().strip()
                user_name = f.readline().strip()
            f.close()
    params, img_size, data_pred, data_target, time_pre_indust_pred, time_indust_pred, data_of_interest_pred, data_of_interest_target, x_train_pre_pred, x_train_ind_pred, x_test_pre_pred, x_test_ind_pred, pre_indust_pred, pre_indust_target, indust_pred, indust_target = perform_preprocess(params)

    if params["verbose"]:
        print('Reshape')
        print(np.shape(data_of_interest_pred))
        print(np.shape(data_of_interest_target[params["interest_var_name"]].data))
        if params["period"] in ['both', 'pre']:
            print(np.shape(pre_indust_pred))
            print(np.shape(pre_indust_target[params["target_var_name"]].data))
        print(np.shape(indust_pred))
        print(np.shape(indust_target[params["target_var_name"]].data))

    # AutoEncoder
    if not "compile_params" in params.keys():
        params["compile_params"] = {}
    if not "fit_params" in params.keys():
        params["fit_params"] = {}
    if params["load_AE"]:
        if params["period"] in ['both', 'pre']:
            AE_pre = keras.models.load_model(params["file_AE_pre"], custom_objects={'keras': keras,'AutoEncoders': AutoEncoders})
        if params["period"] in ['both', 'post']:
            AE_ind = keras.models.load_model(params["file_AE_post"], custom_objects={'keras': keras,'AutoEncoders': AutoEncoders})
        if params["verbose"]:
            print('AE loaded')
    elif params["load_AE_pre"]:
        if params["verbose"]:
            print('Start fitting post')
        if "kl_factor" in params.keys():
            input_dim = [data_pred.dims.get('latitude'),data_pred.dims.get('longitude'),params["kl_factor"]]
        elif "cvae_params" in params.keys():
            input_dim = [data_pred.dims.get('latitude'),data_pred.dims.get('longitude')] + params["cvae_params"]
        else:
            input_dim = [data_pred.dims.get('latitude'),data_pred.dims.get('longitude')]
        if params["period"] in ['both', 'pre']:
            AE_pre = keras.models.load_model(params["file_AE_pre"], custom_objects={'keras': keras,'AutoEncoders': AutoEncoders})
        if params["period"] in ['both', 'post']:
            AE_ind = runAE(input_dim, params["latent_dim"], params["arch"], params["use_VAE"], params["with_cpu"], params["n_epochs"], x_train_ind_pred, params["file_AE_post"], params["verbose"], params["compile_params"], params["fit_params"])
        if params["verbose"]:
            print('Fitting finished for post & AE loaded for pre')
    else:
        if params["verbose"]:
            print('Start fitting')
        if "kl_factor" in params.keys():
            input_dim = [data_pred.dims.get('latitude'),data_pred.dims.get('longitude'),params["kl_factor"]]
        elif "cvae_params" in params.keys():
            input_dim = [data_pred.dims.get('latitude'),data_pred.dims.get('longitude')] + params["cvae_params"]
        else:
            input_dim = [data_pred.dims.get('latitude'),data_pred.dims.get('longitude')]
        if params["period"] in ['both', 'pre']:
            AE_pre = runAE(input_dim, params["latent_dim"], params["arch"], params["use_VAE"], params["with_cpu"], params["n_epochs"], x_train_pre_pred, params["file_AE_pre"], params["verbose"], params["compile_params"], params["fit_params"])
        if params["period"] in ['both', 'post']:
            AE_ind = runAE(input_dim, params["latent_dim"], params["arch"], params["use_VAE"], params["with_cpu"], params["n_epochs"], x_train_ind_pred, params["file_AE_post"], params["verbose"], params["compile_params"], params["fit_params"])
        if params["verbose"]:
            print('Fitting finished')
    
    # Analog comparison
    ## Stats analog
    if params["enhanced_distance"]:
        if params["period"] == 'both':
            stat_data = np.concatenate(((x_train_ind_pred-data_of_interest_pred).flatten(),(x_train_pre_pred-data_of_interest_pred).flatten()),axis=0)
        elif params["period"] == 'pre':
            stat_data = (x_train_pre_pred-data_of_interest_pred).flatten()
        else:
            stat_data = (x_train_ind_pred-data_of_interest_pred).flatten()
        stat_mean = np.abs(stat_data).mean()
        stat_std = np.abs(stat_data).std()
        stat_max = np.abs(stat_data).max()
        stat_min = np.abs(stat_data).min()
        print(f'Mean analog: {stat_mean}')
        print(f'Std analog: {stat_std}')
        print(f'Max anlog: {stat_max}')
        print(f'Min analog: {stat_min}')
        print(f'len analog: {len(stat_data)}')
        print(f'len th analog: {len(stat_data[np.abs(stat_data) < (stat_mean-0.3*stat_std)])}')

    ## Stats AE
    if params["enhanced_distance"]:
        if params["period"] == 'both':
            encoded = get_AE_stats(params["with_cpu"], params["use_VAE"], AE_pre, AE_ind, x_train_pre_pred, x_train_ind_pred, data_of_interest_pred)
        elif params["period"] == 'pre':
            encoded = get_AE_stats(with_cpu=params["with_cpu"], use_VAE=params["use_VAE"], AE_pre=AE_pre, pre_indust_pred=x_train_pre_pred, data_of_interest_pred=data_of_interest_pred, period=params["period"])
        else:
            encoded = get_AE_stats(with_cpu=params["with_cpu"], use_VAE=params["use_VAE"], AE_ind=AE_ind, indust_pred=x_train_ind_pred, data_of_interest_pred=data_of_interest_pred, period=params["period"])
        print(f'Mean: {encoded.mean()}')
        print(f'Std: {encoded.std()}')
        print(f'Max AE: {encoded.max()}')
        print(f'Min AE: {encoded.min()}')
        print(f'len AE: {len(encoded)}')
        print(f'len th AE: {len(encoded[encoded < (encoded.mean() - 0.3*encoded.std())])}')

    ## Time and interest region
    current = datetime.datetime.now()
    int_reg = params["interest_region"]
    if params["interest_region_type"] == "coord":
        if params["resolution"] == 'auto':
            int_reg = calculate_interest_region(params["interest_region"], (data_target.latitude.data, data_target.longitude.data), params["resolution"], is_teleg, params["secret_file"])
        else:
            int_reg = calculate_interest_region(params["interest_region"], [params["latitude_min"], params["latitude_max"], params["longitude_min"], params["longitude_max"]], params["resolution"], is_teleg, params["secret_file"])

    ## This is the threshold of difference between driver/predictor maps and target
    ## to be acepted as low difference
    ## Only used for the local proximity of enhanced distance
    threshold = 0
    threshold_AE = 0
    if params["enhanced_distance"]:
        threshold = stat_mean-0.3*stat_std
        threshold_AE = encoded.mean()-0.3*encoded.std()

    ## Encode
    if params["with_cpu"]:
        with tf.device("/cpu:0"):
            if params["period"] in ['both', 'pre']:
                pre_indust_pred_encoded = AE_pre.predict(pre_indust_pred)
                data_of_interest_pred_encoded_pre = AE_pre.predict(data_of_interest_pred)
            if params["period"] in ['both', 'post']:
                indust_pred_encoded = AE_ind.predict(indust_pred)
                data_of_interest_pred_encoded_ind = AE_ind.predict(data_of_interest_pred)
    else:
        if params["period"] in ['both', 'pre']:
            pre_indust_pred_encoded = AE_pre.predict(pre_indust_pred)
            data_of_interest_pred_encoded_pre = AE_pre.predict(data_of_interest_pred)
        if params["period"] in ['both', 'post']:
            indust_pred_encoded = AE_ind.predict(indust_pred)
            data_of_interest_pred_encoded_ind = AE_ind.predict(data_of_interest_pred)

    if params["verbose"]:
        print('Starting iterations')    
        ax = plt.subplot(1, 1, 1)
        if params["period"] in ['both', 'post']:
            plt.imshow(data_of_interest_pred_encoded_ind.reshape(square_dims(params["latent_dim"])))
        else:
            plt.imshow(data_of_interest_pred_encoded_pre.reshape(square_dims(params["latent_dim"])))
        plt.colorbar()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Encoded')
        plt.savefig(f'./encoded_latent{params["latent_dim"]}.png')
        plt.savefig(f'./encoded_latent{params["latent_dim"]}.pdf')
        plt.close()

    Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
    # Analog Pre
    if params["period"] in ['both', 'pre']:
        file_time_name = f'./comparison-csv/analogues-am-pre-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.npy'.replace(" ","").replace("'", "").replace(",","")
        analog_pre = analogSearch(params["p"], params["k"], pre_indust_pred, data_of_interest_pred, time_pre_indust_pred, pre_indust_target, params["enhanced_distance"], threshold=threshold, img_size=img_size, iter=params["iter"], replace_choice=params["replace_choice"], target_var_name=params["target_var_name"], file_time_name=file_time_name)

    # Analog Post
    if params["period"] in ['both', 'post']:
        file_time_name = f'./comparison-csv/analogues-am-post-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.npy'.replace(" ","").replace("'", "").replace(",","")
        analog_ind = analogSearch(params["p"], params["k"], indust_pred, data_of_interest_pred, time_indust_pred, indust_target, params["enhanced_distance"], threshold=threshold, img_size=img_size, iter=params["iter"], replace_choice=params["replace_choice"], target_var_name=params["target_var_name"], file_time_name=file_time_name)
    

    # AE Pre
    if params["period"] in ['both', 'pre']:
        file_time_name = f'./comparison-csv/analogues-ae-am-pre-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.npy'.replace(" ","").replace("'", "").replace(",","")
        latent_analog_pre = analogSearch(params["p"], params["k"], pre_indust_pred_encoded, data_of_interest_pred_encoded_pre, time_pre_indust_pred, pre_indust_target, params["enhanced_distance"], threshold=threshold_AE, img_size=img_size, iter=params["iter"], replace_choice=params["replace_choice"], target_var_name=params["target_var_name"], file_time_name=file_time_name)
    

    # AE Post
    if params["period"] in ['both', 'post']:
        file_time_name = f'./comparison-csv/analogues-ae-am-post-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.npy'.replace(" ","").replace("'", "").replace(",","")
        latent_analog_ind = analogSearch(params["p"], params["k"], indust_pred_encoded, data_of_interest_pred_encoded_ind, time_indust_pred, indust_target, params["enhanced_distance"], threshold=threshold_AE, img_size=img_size, iter=params["iter"], replace_choice=params["replace_choice"], target_var_name=params["target_var_name"], file_time_name=file_time_name)
    
    dict_stats = {}
    reconstruction_Pre_Analog = []
    reconstruction_Pre_AE = []
    reconstruction_Post_Analog = []
    reconstruction_Post_AE = []
    for i in range(params["iter"]):
        if params["period"] in ['both', 'pre']:
            dict_stats[f'WithoutAE-Pre{i}'] = [np.nansum(np.abs(data_of_interest_pred - analog_pre[0][i]))/img_size,
                                    np.nanmean(np.abs(data_of_interest_target[params["interest_var_name"]].data - analog_pre[1][i])[:,int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                    np.nanmean(analog_pre[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                    str(analog_pre[2][i].data)[:10]]
            reconstruction_Pre_Analog.append(analog_pre[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]])
        else:
            dict_stats[f'WithoutAE-Pre{i}'] = [np.nan, np.nan, np.nan]
        
        if params["period"] in ['both', 'post']:
            dict_stats[f'WithoutAE-Post{i}'] = [np.nansum(np.abs(data_of_interest_pred - analog_ind[0][i]))/img_size,
                                    np.nanmean(np.abs(data_of_interest_target[params["interest_var_name"]].data - analog_ind[1][i])[:,int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                    np.nanmean(analog_ind[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                    str(analog_ind[2][i].data)[:10]]
            reconstruction_Post_Analog.append(analog_ind[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]])
        else:
            dict_stats[f'WithoutAE-Post{i}'] = [np.nan, np.nan, np.nan]
        
        if params["period"] in ['both', 'pre']:
            dict_stats[f'WithAE-Pre-Pre{i}'] = [np.nansum(np.abs(data_of_interest_pred_encoded_pre - latent_analog_pre[0][i]))/img_size,
                                        np.nanmean(np.abs(data_of_interest_target[params["interest_var_name"]].data - latent_analog_pre[1][i])[:,int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                        np.nanmean(latent_analog_pre[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                        str(latent_analog_pre[2][i].data)[:10]]
            reconstruction_Pre_AE.append(latent_analog_pre[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]])
        else:
            dict_stats[f'WithAE-Pre-Pre{i}'] = [np.nan, np.nan, np.nan]

        if params["period"] in ['both', 'post']:
            dict_stats[f'WithAE-Post-Post{i}'] = [np.nansum(np.abs(data_of_interest_pred_encoded_ind - latent_analog_ind[0][i]))/img_size,
                                    np.nanmean(np.abs(data_of_interest_target[params["interest_var_name"]].data - latent_analog_ind[1][i])[:,int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                    np.nanmean(latent_analog_ind[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]),
                                    str(latent_analog_ind[2][i].data)[:10]]
            reconstruction_Post_AE.append(latent_analog_ind[1][i][int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]])
        else:
            dict_stats[f'WithAE-Post-Post{i}'] = [np.nan, np.nan, np.nan]

        dict_stats[f'Original{i}']=[0,0,np.nanmean(((data_of_interest_target[params["interest_var_name"]].data)[:,int_reg[0]:int_reg[1],int_reg[2]:int_reg[3]]))]
        
        if params["verbose"]:
            print(f'Iteration {i} finished')

    df_stats = pd.DataFrame.from_dict(dict_stats,orient='index',columns=['pred-diff','target-diff','target','time'])
    Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(f'./comparison-csv/{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}-analog-comparision-stats{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.csv'.replace(" ","").replace("'", "").replace(",",""))
    return reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE


def identify_heatwave_days(params: dict) -> Union[list, np.ndarray]:
    """
    identify_heatwave_days                             
    
    Method that perform the identifitacion of the heat wave period, following the definition from http://doi.org/10.1088/1748-9326/10/12/124003.
    
    Parameters
    ----------
    params: dict
        A dictionary with needed parameters and configuration. Mainly loaded from the configuration file, with some auxiliar parameters added by other functions.
    
    Returns
    ----------
    heatwave_period: list or ndarray
        A list of datetime that contains the heat wave period.
     
    Raises
    ------
    ValueError
        If percentile out of [0,100] range.
    
    Notes
    -----
    - Uses 90th percentile by default
    - Follows Russo et al. (2015) methodology
    - Generates validation plots in ./figures/
    - Handles both single and multi-day events
    """
    # Load data
    if not "ident_dataset" in params:
        params["ident_dataset"] = 'data/data_dailyMax_t2m_1940-2022.nc'
    if not "ident_var_name" in params:
        params["ident_var_name"] = 't2m_dailyMax'
    data_target=xr.load_dataset(params["ident_dataset"], engine='netcdf4') #Target Datasets
    if params["verbose"]:
        print('Data loaded')

    # Preprocess
    ## Change longitude coordinates from 0 - 359 to -180 - 179
    if data_target.longitude.min() == 0:
        data_target = data_target.assign_coords(longitude=(((data_target.longitude + 180) % 360) - 180))
    data_target = data_target.sortby(data_target.longitude)
    data_target = data_target.sortby(data_target.latitude)
    if params["verbose"]:
        print('Coord changed')

    ## Load Target Dataset
    data_target = data_target.sel(latitude=slice(params["latitude_min"],params["latitude_max"]),longitude=slice(params["longitude_min"],params["longitude_max"]))
    
    time_x = data_target.sel(time=slice(params["data_of_interest_init"],params["data_of_interest_end"])).get_index('time')

    ## Extract interest data
    idx_interest = params["interest_region"]
    if params["interest_region_type"] == "coord":
        if params["resolution"] == 'auto':
            idx_interest = calculate_interest_region(params["interest_region"], (data_target.latitude.data, data_target.longitude.data), params["resolution"], params["teleg"], params["secret_file"])
        else:
            idx_interest = calculate_interest_region(params["interest_region"], [params["latitude_min"], params["latitude_max"], params["longitude_min"], params["longitude_max"]], params["resolution"], params["teleg"], params["secret_file"])
    data_target = data_target.isel(latitude=slice(idx_interest[0], idx_interest[1]),longitude=slice(idx_interest[2], idx_interest[3]))
    data_target = data_target.mean(dim=['latitude', 'longitude'])
    
    
    ## Percentil
    which_percentile = 90
    if "percentile" in params.keys():
        which_percentile = params["percentile"]
    

    percentile_threshold = functions.hw_pctl(data_target[params["ident_var_name"]], ['1981', '2010'], which_percentile, params["ident_var_name"])
    data_target = data_target.sel(time=slice(params["data_of_interest_init"], params["data_of_interest_end"]))
    percentile_threshold = percentile_threshold.sel(dayofyear=slice(time_x[0].day_of_year, time_x[-1].day_of_year))
    surpass_threshold = functions.isHW_in_ds(data_target, percentile_threshold, params["ident_var_name"]).isHW.data
    amount_surpass_threshold = ((surpass_threshold).astype(np.int8)).sum()
    data_target_array = data_target[params["ident_var_name"]].data
    percentile_threshold_array = percentile_threshold.pctl_th.data
    
    plt.figure()
    plt.plot(time_x, (data_target_array - 273.15), marker='o', label='Days studied')
    plt.xticks(rotation=45)
    plt.plot(time_x, (percentile_threshold_array - 273.15), color='r', label=f'p{which_percentile} threshold')
    if params["verbose"]:
        print(f'Threshold: {percentile_threshold_array - 273.15}')
        print(f'Amount of days that surpass the threshold: {amount_surpass_threshold}' )

    if amount_surpass_threshold == len(data_target_array):
        heatwave_period = time_x#.astype('datetime64[D]')
    elif amount_surpass_threshold == 0:
        heatwave_period = []
    else:
        grouped_surpass = np.array([(val , sum(1 for i in val_to_count)) for val, val_to_count in groupby(surpass_threshold)])
        mask = grouped_surpass[:,0]==1
        idx = np.arange(len(grouped_surpass))[mask][np.argmax(grouped_surpass[mask], axis=0)[1]]
        cumsum = np.cumsum(grouped_surpass, axis=0)
        heatwave_period = time_x[(0 if idx == 0 else cumsum[idx-1][1]):cumsum[min(np.shape(cumsum)[0],idx)][1]]#.astype('datetime64[D]')
    Path("./figures").mkdir(parents=True, exist_ok=True)
    if len(heatwave_period) > 0:
        plt.title(f'Heatwave: {heatwave_period[0].date()} - {heatwave_period[-1].date()}')
        plt.axvspan(heatwave_period[0], heatwave_period[-1], label='Heatwave period', color='crimson', alpha=0.3)
    plt.ylabel('Target Dataset (ºC)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/percentile{which_percentile}-for-{params["season"]}{params["name"][:-1]}.png')
    plt.savefig(f'./figures/percentile{which_percentile}-for-{params["season"]}{params["name"][:-1]}.pdf')
    if params["verbose"]:
        print(f'Heat wave period is {heatwave_period}')
        plt.show()
    plt.close()

    return heatwave_period

def _get_post_information(params: dict):
    """
    _get_post_information

    Auxiliar function used to obtain target var long_name + units, and if is Mv or not.

    Parameters
    ----------
    params : dict
        Dictionary with the parameters.

    Returns
    -------
    tuple
        A tuple of Strings. First one contains "Mv" or "" if the pred dataset is multivariate
        or not. Second one contains long_name and units of target variable.
    """    
    data_pred = xr.open_dataset(params["pred_dataset"])
    is_Mv = "Mv" if len(data_pred.data_vars) > 1 else ""
    data_target = xr.open_dataset(params["target_dataset"])
    var_name = list(data_target.data_vars)[0]
    target_var = rf'{data_target[var_name].attrs["long_name"]} {data_target[var_name].attrs["units"]}'
    return is_Mv, target_var

def post_process(params_file: str, save_stats: bool = True, is_atribution: bool = False, compare_to_am: bool = True, target_stat: str = "max"):
    """
    post_process

    Function to perform the post-process after the execution of the main code. This method will save a comparative figure and a
    statistical summary of the resultss.

    Parameters
    ----------
    params_file : str
        Path to the parameters file.
    save_stats : bool, optional
        If the statistical summary needs to be saved or not, by default True. If False, stats are printed but not saved.
    is_atribution : bool, optional
        Flag in case you are performing Atribution and want to get a comparison between Pre/Post period results, by default False
    compare_to_am : bool, optional
        Falg in case you performed also the Classical AM and want compare between AE-AM and AM, by default True
    target_stat : str, optional
        How to obtain the target value, by mean, max, min, etc., by default "max".

    Returns
    ----------
    
    Raises
    ------
    FileNotFoundError
        If result files missing.
    
    Notes
    -----
    - Generates KDE comparison plots
    - Produces multi-method statistical summaries
    - Handles both attribution and detection modes
    - Supports parallel execution results
    """

    # Read params
    file_params = open(params_file)
    params = json.load(file_params)
    file_params.close()
    # Datasets information
    is_Mv, target_var = _get_post_information(params)
    current = datetime.datetime.now()
    # Perform post-process
    Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
    Path("./figures").mkdir(parents=True, exist_ok=True)
    path = f'./comparison-csv/*{params["name"]}*{params["latent_dim"]}*arch{params["arch"]}*.csv'
    files_interest = glob.glob(path)
    files_interest = sorted(files_interest)
    list_interest = [pd.read_csv(df) for df in files_interest]
    ## Obtain stats
    tar_fun = getattr(np, target_stat)
    target = tar_fun([elem.get('target')[4::5] if 'target' in elem.columns else elem.get('temp')[4::5] for elem in list_interest])
    AE_Pre = np.array([elem.get('target')[2::5]  if 'target' in elem.columns else elem.get('temp')[2::5] for elem in list_interest]).flatten()
    AE_Pre_preddiff = np.array([elem.get('pred-diff')[2::5] for elem in list_interest]).flatten()
    AE_Ind = np.array([elem.get('target')[3::5]  if 'target' in elem.columns else elem.get('temp')[3::5] for elem in list_interest]).flatten()
    AE_Ind_preddiff = np.array([elem.get('pred-diff')[3::5] for elem in list_interest]).flatten()
    if compare_to_am:
        analog_Pre = np.array([elem.get('target')[0::5] if 'target' in elem.columns else elem.get('temp')[0::5] for elem in list_interest]).flatten()
        analog_Pre_preddiff = np.array([elem.get('pred-diff')[0::5] for elem in list_interest]).flatten()
        analog_Ind = np.array([elem.get('target')[1::5] if 'target' in elem.columns else elem.get('temp')[1::5] for elem in list_interest]).flatten()
        analog_Ind_preddiff = np.array([elem.get('pred-diff')[1::5] for elem in list_interest]).flatten()
    ## Reduce dim
    is_execs = np.any(["exec" in file_name for file_name in files_interest])
    reduce_dim = params["iter"]
    if is_execs:
        if "n_execs" in params.keys():
            reduce_dim = params["n_execs"] * reduce_dim
        else:
            reduce_dim = 5 * reduce_dim
    AE_Pre = np.reshape(AE_Pre, (int(len(AE_Pre)/reduce_dim), reduce_dim))
    AE_Pre = np.mean(AE_Pre, axis=0)
    AE_Ind = np.reshape(AE_Ind, (int(len(AE_Ind)/reduce_dim), reduce_dim))
    AE_Ind = np.mean(AE_Ind, axis=0)
    if compare_to_am:
        analog_Pre = np.reshape(analog_Pre, (int(len(analog_Pre)/reduce_dim), reduce_dim))
        analog_Pre = np.mean(analog_Pre, axis=0)
        analog_Ind = np.reshape(analog_Ind, (int(len(analog_Ind)/reduce_dim), reduce_dim))
        analog_Ind = np.mean(analog_Ind, axis=0)
    ## Make plot
    if is_atribution:
        if compare_to_am:
            matrix_comp = np.array([analog_Pre, analog_Ind, AE_Pre, AE_Ind])
            df_comp = pd.DataFrame(matrix_comp.T, columns=[f'{is_Mv}AM in Pre', f'{is_Mv}AM in Post', f'{is_Mv}AE-AM in Pre', f'{is_Mv}AE-AM in Post'])
        else:
            matrix_comp = np.array([AE_Pre, AE_Ind])
            df_comp = pd.DataFrame(matrix_comp.T, columns=[f'{is_Mv}AE-AM in Pre', f'{is_Mv}AE-AM in Post'])
    else:
        if compare_to_am:
            matrix_comp = np.array([analog_Ind, AE_Ind])
            df_comp = pd.DataFrame(matrix_comp.T, columns=[f'{is_Mv}AM', f'{is_Mv}AE-AM'])
        else:
            matrix_comp = np.array([AE_Ind])
            df_comp = pd.DataFrame(matrix_comp.T, columns=[f'{is_Mv}AE-AM'])
    df_comp_melted = df_comp.melt()
    if is_atribution:
        a = sns.displot(df_comp_melted, y = 'value', hue = 'variable', kind='kde', fill=True, legend=False)
        children = plt.gca().get_children()
        l = plt.axhline(target, color='red')
        if compare_to_am:
            plt.legend(children[:4] + [l], [f'{is_Mv}AE-AM Post', f'{is_Mv}AE-AM Pre', f'{is_Mv}AM Post', f'{is_Mv}AM Pre', 'target'],
                    loc='upper right', bbox_to_anchor=(1.05,0.9))
        else:
            plt.legend(children[:2] + [l], [f'{is_Mv}AE-AM Post', f'{is_Mv}AE-AM Pre''target'],
                    loc='upper right', bbox_to_anchor=(1.05,0.9))
        plt.ylabel(target_var)
        ## Save plot
        plt.savefig((f'./figures/distribution-{params["season"]}-automated-functions-LatentSpace{params["latent_dim"]}-{params["k"]}.png').replace('*',''))
        plt.savefig((f'./figures/distribution-{params["season"]}-automated-functions-LatentSpace{params["latent_dim"]}-{params["k"]}.pdf').replace('*',''))
    else:
        a = sns.displot(df_comp_melted, y = 'value', hue = 'variable', kind='kde', fill=True, legend=False)
        children = plt.gca().get_children()
        l = plt.axhline(target, color='red')
        if compare_to_am:
            plt.legend(children[:2] + [l], [f'{is_Mv}AE-AM', f'{is_Mv}AM', 'target'], loc='upper right', bbox_to_anchor=(1.,0.9))
        else:
            plt.legend(children[:1] + [l], [f'{is_Mv}AE-AM', 'target'], loc='upper right', bbox_to_anchor=(1.,0.9))
        plt.ylabel(target_var)
        ## Save plot
        plt.savefig((f'./figures/distribution-{params["season"]}-automated-functions-LatentSpace{params["latent_dim"]}-{params["k"]}.png').replace('*',''))
        plt.savefig((f'./figures/distribution-{params["season"]}-automated-functions-LatentSpace{params["latent_dim"]}-{params["k"]}.pdf').replace('*',''))
    # if save_stats save stats
    stats_name = ["Mean", "Std", "Diff with Target", "Diff with Pred"]
    if save_stats:
        if is_atribution:
            if compare_to_am:
                stats_nd = np.array(
                    [
                        [np.round(np.mean(AE_Pre), decimals=4), np.round(np.mean(analog_Pre), decimals=4),
                        np.round(np.mean(AE_Ind), decimals=4), np.round(np.mean(analog_Ind), decimals=4), np.round(target, decimals=4)],
                        [np.round(np.std(AE_Pre), decimals=4), np.round(np.std(analog_Pre), decimals=4),
                        np.round(np.std(AE_Ind), decimals=4), np.round(np.std(analog_Ind), decimals=4), 0],
                        [np.round(target-np.mean(AE_Pre), decimals=4), np.round(target-np.mean(analog_Pre), decimals=4),
                        np.round(np.mean(target-AE_Ind), decimals=4), np.round(target-np.mean(analog_Ind), decimals=4), 0],
                        [np.round(np.mean(AE_Pre_preddiff), decimals=4), np.round(np.mean(analog_Pre_preddiff), decimals=4),
                        np.round(np.mean(AE_Ind_preddiff), decimals=4), np.round(np.mean(analog_Ind_preddiff), decimals=4),0],
                    ]
                )
                df_stats = pd.DataFrame(data=stats_nd.T, columns=stats_name, index=[f'{is_Mv}AE-AM in Pre', f'{is_Mv}AM in Pre', f'{is_Mv}AE-AM in Post', f'{is_Mv}AM in Post', 'Target'])
                Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
                df_stats.to_csv(f'./comparison-csv/stats-summary-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}-analog-comparision-stats{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.csv'.replace(" ","").replace("'", "").replace(",",""))
            else:
                stats_nd = np.array(
                    [
                        [np.round(np.mean(AE_Pre), decimals=4), np.round(np.mean(AE_Ind), decimals=4), np.round(target, decimals=4)],
                        [np.round(np.std(AE_Pre), decimals=4), np.round(np.std(AE_Ind), decimals=4), 0],
                        [np.round(target-np.mean(AE_Pre), decimals=4), np.round(np.mean(target-AE_Ind), decimals=4), 0],
                        [np.round(np.mean(AE_Pre_preddiff), decimals=4), np.round(np.mean(AE_Ind_preddiff), decimals=4), 0],
                    ]
                )
                df_stats = pd.DataFrame(data=stats_nd.T, columns=stats_name, index=[f'{is_Mv}AE-AM in Pre', f'{is_Mv}AE-AM in Post', 'Target'])
                Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
                df_stats.to_csv(f'./comparison-csv/stats-summary-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}-analog-comparision-stats{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.csv'.replace(" ","").replace("'", "").replace(",",""))
        else:
            if compare_to_am:
                stats_nd = np.array(
                    [
                        [np.round(np.mean(AE_Ind), decimals=4), np.round(np.mean(analog_Ind), decimals=4), np.round(target, decimals=4)],
                        [np.round(np.std(AE_Ind), decimals=4), np.round(np.std(analog_Ind), decimals=4), 0],
                        [np.round(np.mean(target-AE_Ind), decimals=4), np.round(target-np.mean(analog_Ind), decimals=4), 0],
                        [np.round(np.mean(AE_Ind_preddiff), decimals=4), np.round(np.mean(analog_Ind_preddiff), decimals=4),0],
                    ]
                )
                df_stats = pd.DataFrame(data=stats_nd.T, columns=stats_name, index=[f'{is_Mv}AE-AM', f'{is_Mv}AM', 'Target'])
                Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
                df_stats.to_csv(f'./comparison-csv/stats-summary-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}-analog-comparision-stats{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.csv'.replace(" ","").replace("'", "").replace(",",""))
            else:
                stats_nd = np.array(
                    [
                        [np.round(np.mean(AE_Ind), decimals=4), np.round(target, decimals=4)],
                        [np.round(np.std(AE_Ind), decimals=4), 0],
                        [np.round(np.mean(target-AE_Ind), decimals=4), 0],
                        [np.round(np.mean(AE_Ind_preddiff), decimals=4), 0],
                    ]
                )
                df_stats = pd.DataFrame(data=stats_nd.T, columns=stats_name, index=[f'{is_Mv}AE-AM', 'Target'])
                Path("./comparison-csv").mkdir(parents=True, exist_ok=True)
                df_stats.to_csv(f'./comparison-csv/stats-summary-{params["season"]}{params["name"]}x{params["iter"]}-{params["data_of_interest_init"]}-epoch{params["n_epochs"]}-latent{params["latent_dim"]}-k{params["k"]}-arch{params["arch"]}-{"VAE" if params["use_VAE"] else "noVAE"}-analog-comparision-stats{current.year}-{current.month}-{current.day}-{current.hour}-{current.minute}-{current.second}.csv'.replace(" ","").replace("'", "").replace(",",""))
    # else print stats
    else:
        if is_atribution:
            if compare_to_am:
                print(f'Pre {is_Mv}AE-AM mean: {np.mean(AE_Pre):.4f}')
                print(f'Pre {is_Mv}AE-AM std: {np.std(AE_Pre):.4f}')
                print(f'Pre Diff with Target: {target-np.mean(AE_Pre):.4f}')
                print(f'Pre Diff with Pred {np.mean(AE_Pre_preddiff):.4f}')
                print(f'Pre {is_Mv}AM mean: {np.mean(analog_Pre):.4f}')
                print(f'Pre {is_Mv}AM std: {np.std(analog_Pre):.4f}')
                print(f'Pre Diff with Target: {target-np.mean(analog_Pre):.4f}')
                print(f'Pre Diff with Pred {np.mean(analog_Pre_preddiff):.4f}\n')
                print(f'Post {is_Mv}AE-AM mean: {np.mean(AE_Ind):.4f}')
                print(f'Post {is_Mv}AE-AM std: {np.std(AE_Ind):.4f}')
                print(f'Post Diff with Target: {target-np.mean(AE_Ind):.4f}')
                print(f'Post Diff with Pred {np.mean(AE_Ind_preddiff):.4f}')
                print(f'Post {is_Mv}AM mean: {np.mean(analog_Ind):.4f}')
                print(f'Post {is_Mv}AM std: {np.std(analog_Ind):.4f}')
                print(f'Post Diff with Target: {target-np.mean(analog_Ind):.4f}')
                print(f'Post Diff with Pred {np.mean(analog_Ind_preddiff):.4f}\n')
                print(f'Target: {target:.4f}')
            else:
                print(f'Pre {is_Mv}AE-AM mean: {np.mean(AE_Pre):.4f}')
                print(f'Pre {is_Mv}AE-AM std: {np.std(AE_Pre):.4f}')
                print(f'Pre Diff with Target: {target-np.mean(AE_Pre):.4f}')
                print(f'Pre Diff with Pred {np.mean(AE_Pre_preddiff):.4f}')
                print(f'Post {is_Mv}AE-AM mean: {np.mean(AE_Ind):.4f}')
                print(f'Post {is_Mv}AE-AM std: {np.std(AE_Ind):.4f}')
                print(f'Post Diff with Target: {target-np.mean(AE_Ind):.4f}')
                print(f'Post Diff with Pred {np.mean(AE_Ind_preddiff):.4f}')
                print(f'Target: {target:.4f}')
        else:
            if compare_to_am:
                print(f'{is_Mv}AE-AM mean: {np.mean(AE_Ind):.4f}')
                print(f'{is_Mv}AE-AM std: {np.std(AE_Ind):.4f}')
                print(f'Diff with Target: {target-np.mean(AE_Ind):.4f}')
                print(f'Diff with Pred {np.mean(AE_Ind_preddiff):.4f}')
                print(f'{is_Mv}AM mean: {np.mean(analog_Ind):.4f}')
                print(f'{is_Mv}AM std: {np.std(analog_Ind):.4f}')
                print(f'Diff with Target: {target-np.mean(analog_Ind):.4f}')
                print(f'Diff with Pred {np.mean(analog_Ind_preddiff):.4f}\n')
                print(f'Target: {target:.4f}')
            else:
                print(f'{is_Mv}AE-AM mean: {np.mean(AE_Ind):.4f}')
                print(f'{is_Mv}AE-AM std: {np.std(AE_Ind):.4f}')
                print(f'Diff with Target: {target-np.mean(AE_Ind):.4f}')
                print(f'Diff with Pred {np.mean(AE_Ind_preddiff):.4f}')
                print(f'Target: {target:.4f}')
    return


def _step_loop(params, params_multiple, file_params_name, n_execs, ident, verb, teleg, token, chat_id, user_name, save_recons, args):
    """
      _step_loop
       
      Auxiliar method that handle the runs depending on the argsparse options selected.
        
      Parameters
      ----------
      params: dict
          Default parameters and configuration dictionary for most of the executions.
      params_multiple: dict
          Specific parameters and configuration dictionary that overwrite params for some methods.
      file_params_name: str
          The default name of the params/configuration file.
      n_execs: int
          The number of repeted executions for some methods.
      ident: bool
          Value of flag to performs the identification period task or not.
      verb: bool
          Value of flag that indicates if verbosity information should be show or not.
      teleg: bool
          Value of flag for sending Exceptions to Telegram bot.
      token: str
          Token of Telegram bot.
      chat_id: str
          ID of the chat where the Telegram bot will send the messages.
      user_name: str
          User name to mention in case of Exceptions.
      save_recons: bool
          Value of flag for saving or not the reconstrucion information in an .nc file.
      args: argsparse
          The argsparse object with the activated options from terminal. 
        
      Returns
      ----------
    """
    # Period
    period = 'both'
    if args.period is not None:
        period = args.period
    reconstructions_Pre_Analog = []
    reconstructions_Post_Analog = []
    reconstructions_Pre_AE = []
    reconstructions_Post_AE = []
    # Execution of method
    if args.method == 'day' or args.method is None:
        if ident:
            # Read file
            ## Default if configfile is not specified
            if args.conf is not None:
                file_params_name = args.conf
            ## Try-except-else to open file and re-write params
            try:
                file_params = open(file_params_name)
            except:
                message = OSError(f'File {file_params_name} not found. To identify the Heat wave period a configuration parameters file is needed.')
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
                    print(requests.get(url).json())
                raise message
            else:
                params = json.load(file_params)
                file_params.close()
            if verb is not None:
                params["verbose"] = verb
            params["teleg"] = teleg
            params["period"] = period
            if "p" not in params.keys():
                params["p"] = 2
            if "enhanced_distance" not in params.keys():
                params["enhanced_distance"] = False
            if "save_recons" not in params.keys():
                params["save_recons"] = save_recons
            if "secret_file" not in params.keys():
                params["secret_file"] = args.secret
            identify_heatwave_days(params)
            message = f"Indentify Heat wave period (flag -i   --identifyhw) for {params['name'][1:-1]} is not compatible with default 'method' ('day') and this will not be executed"
            if teleg:
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                print(requests.get(url).json())
            warnings.warn(message)
        else:
            runComparison(params)
        # Everything finished
        message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
        if teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
            requests.get(url).json()
        warnings.warn(message)
    elif args.method in ['days', 'seasons', 'execs', 'latents', 'seasons-execs', 'latents-execs', 'latents-seasons-execs']:
        # Read file
        ## Default if configfile is not specified
        if args.conf is not None:
            file_params_name = args.conf
        ## Try-except-else to open file and re-write params
        try:
            file_params = open(file_params_name)
        except:
            message = OSError(f'File {file_params_name} not found. Your method need a configuration parameters file.')
            if teleg:
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
                requests.get(url).json()
            raise message
        else:
            params_multiple = json.load(file_params)
            params_multiple["teleg"] = teleg
            params_multiple["period"] = period
            if "secret_file" not in params_multiple.keys():
                params_multiple["secret_file"] = args.secret
            if ident:
                heatwave_period = identify_heatwave_days(params_multiple)
                params_multiple["data_of_interest_init"] = heatwave_period
                params_multiple["data_of_interest_end"] = heatwave_period
            else:
                if params_multiple["per_what"] == "per_month":
                    heatwave_period = (pd.date_range(start=params_multiple["data_of_interest_init"], end=params_multiple["data_of_interest_end"], freq='MS')).to_numpy()
                elif params_multiple["per_what"] == "per_week":
                    heatwave_period = np.arange(datetime.datetime.strptime(params_multiple["data_of_interest_init"], '%Y-%m-%d'), datetime.datetime.strptime(params_multiple["data_of_interest_end"], '%Y-%m-%d')+datetime.timedelta(days=1), datetime.timedelta(weeks=1))
                else:
                    heatwave_period = np.arange(datetime.datetime.strptime(params_multiple["data_of_interest_init"], '%Y-%m-%d'), datetime.datetime.strptime(params_multiple["data_of_interest_end"], '%Y-%m-%d')+datetime.timedelta(days=1), datetime.timedelta(days=1))
                heatwave_period = np.array(list(map(pd.Timestamp, heatwave_period)))
                params_multiple["data_of_interest_init"] = heatwave_period
                params_multiple["data_of_interest_end"] = heatwave_period
            params = params_multiple.copy()
            file_params.close()
        if verb is not None:
            params["verbose"] = verb
        if "p" not in params.keys():
            params["p"] = 2
        if "enhanced_distance" not in params.keys():
            params["enhanced_distance"] = False
        if "save_recons" not in params.keys():
            params["save_recons"] = save_recons
        # Methods with configfile
        if args.method == 'days':
            for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                if idx == 0:
                    params["load_AE"] = False
                if idx > 0 and not params["load_AE"]:
                    params["load_AE"] = True
                params["data_of_interest_init"] = init
                params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                if params["save_recons"]:
                    reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                    reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                    reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                    reconstructions_Post_AE.append(reconstruction_Post_AE)
            save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
            # Exec finished
            message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
            if teleg:
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                requests.get(url).json()
            warnings.warn(message)
        elif args.method == 'seasons':
            for season in params_multiple["season"]:
                params["season"] = season
                for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                    if idx == 0:
                        params["load_AE"] = False
                    if idx > 0 and not params["load_AE"]:
                        params["load_AE"] = True
                    params["data_of_interest_init"] = init
                    params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                    reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                    if params["save_recons"]:
                        reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                        reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                        reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                        reconstructions_Post_AE.append(reconstruction_Post_AE)
                save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                # Exec finished
                message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                    requests.get(url).json()
                warnings.warn(message)
        elif args.method == 'execs':
            if 'n_execs' in params.keys():
                n_execs = params['n_execs']
            for i in range(n_execs):
                params["name"] = f'-exec{i}{params_multiple["name"]}'
                for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                    if idx == 0:
                        params["load_AE"] = False
                    if idx > 0 and not params["load_AE"]:
                        params["load_AE"] = True
                    params["data_of_interest_init"] = init
                    params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                    reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                    if params["save_recons"]:
                        reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                        reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                        reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                        reconstructions_Post_AE.append(reconstruction_Post_AE)
                save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                # Exec finished
                message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                    requests.get(url).json()
                warnings.warn(message)
        elif args.method == 'latents':
            for latent in params_multiple["latent_dim"]:
                params["latent_dim"] = latent
                for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                    if idx == 0:
                        params["load_AE"] = False
                    if idx > 0 and not params["load_AE"]:
                        params["load_AE"] = True
                    params["data_of_interest_init"] = init
                    params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                    reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                    if params["save_recons"]:
                        reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                        reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                        reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                        reconstructions_Post_AE.append(reconstruction_Post_AE)
                save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                # Exec finished
                message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                    requests.get(url).json()
                warnings.warn(message)
        elif args.method == 'seasons-execs':
            if 'n_execs' in params.keys():
                n_execs = params["n_execs"]
            for season in params_multiple["season"]:
                params["season"] = season
                for i in range(n_execs):
                    params["name"] = f'-exec{i}{params_multiple["name"]}'
                    for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                        if idx == 0:
                            params["load_AE"] = False
                        if idx > 0 and not params["load_AE"]:
                            params["load_AE"] = True
                        params["data_of_interest_init"] = init
                        params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                        reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                        if params["save_recons"]:
                            reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                            reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                            reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                            reconstructions_Post_AE.append(reconstruction_Post_AE)
                    save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                    # Exec finished
                    message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                    if teleg:
                        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                        requests.get(url).json()
                    warnings.warn(message)
        elif args.method == 'latents-execs':
            if 'n_execs' in params.keys():
                n_execs = params['n_execs']
            for latent in params_multiple["latent_dim"]:
                params["latent_dim"] = latent
                for i in range(n_execs):
                    params["name"] = f'-exec{i}{params_multiple["name"]}'
                    for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                        if idx == 0:
                            params["load_AE"] = False
                        if idx > 0 and not params["load_AE"]:
                            params["load_AE"] = True
                        params["data_of_interest_init"] = init
                        params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                        reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                        if params["save_recons"]:
                            reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                            reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                            reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                            reconstructions_Post_AE.append(reconstruction_Post_AE)
                    save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                    # Exec finished
                    message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                    if teleg:
                        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                        requests.get(url).json()
                    warnings.warn(message)
        elif args.method == 'latents-seasons-execs':
            if 'n_execs' in params.keys():
                n_execs = params['n_execs']
            for latent in params_multiple["latent_dim"]:
                params["latent_dim"] = latent
                for season in params_multiple["season"]:
                    params["season"] = season
                    for i in range(n_execs):
                        params["name"] = f'-exec{i}{params_multiple["name"]}'
                        for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                            if idx == 0:
                                params["load_AE"] = False
                            if idx > 0 and not params["load_AE"]:
                                params["load_AE"] = True
                            params["data_of_interest_init"] = init
                            params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                            reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                            if params["save_recons"]:
                                reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                                reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                                reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                                reconstructions_Post_AE.append(reconstruction_Post_AE)
                        save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                        # Exec finished
                        message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                        if teleg:
                            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                            requests.get(url).json()
                        warnings.warn(message)
    else:
        message = ValueError(f"Not recognized {args.method} method. The available methods are 'day' (default), 'days', 'seasons', 'execs', 'latents', 'seasons-execs', 'latents-execs' or 'latents-seasons-execs'")
        if teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
            requests.get(url).json()
        raise message
    # Post-process
    post_process(file_params_name, is_atribution = period == 'both')
    message = f'Post process of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
    if teleg:
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
        requests.get(url).json()
    warnings.warn(message)
    return

def _step_loop_without_args(params, params_multiple, file_params_name, n_execs, ident, verb, teleg, token, chat_id, user_name, save_recons, period, method):
    """
    _step_loop
     
    Auxiliar method that handle the runs depending on the options specified.
      
    Parameters
    ----------
    params: dict
        Default parameters and configuration dictionary for most of the executions.
    params_multiple: dict
        Specific parameters and configuration dictionary that overwrite params for some methods.
    file_params_name: str
        The default name of the params/configuration file.
    n_execs: int
        The number of repeted executions for some methods.
    ident: bool
        Value of flag to performs the identification period task or not.
    verb: bool
        Value of flag that indicates if verbosity information should be show or not.
    teleg: bool
        Value of flag for sending Exceptions to Telegram bot.
    token: str
        Token of Telegram bot.
    chat_id: str
        ID of the chat where the Telegram bot will send the messages.
    user_name: str
        User name to mention in case of Exceptions.
    save_recons: bool
        Value of flag for saving or not the reconstrucion information in an .nc file.
    period: str
        Specify the period where to perform the operation between `both` (default), `pre` or `post`.
    method: str
        Specify an method to execute between: `day` (default), `days`, `seasons`, `execs`, `latents`, `seasons-execs`, `latents-execs` or `latents-seasons-execs`
      
    Returns
    ----------
    """
    reconstructions_Pre_Analog = []
    reconstructions_Post_Analog = []
    reconstructions_Pre_AE = []
    reconstructions_Post_AE = []
    # Execution of method
    if method == 'day':
        if ident:
            # Read file
            ## Try-except-else to open file and re-write params
            try:
                file_params = open(file_params_name)
            except:
                message = OSError(f'File {file_params_name} not found. To identify the Heat wave period a configuration parameters file is needed.')
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
                    print(requests.get(url).json())
                raise message
            else:
                params = json.load(file_params)
                file_params.close()
            if verb is not None:
                params["verbose"] = verb
            params["teleg"] = teleg
            params["period"] = period
            if "p" not in params.keys():
                params["p"] = 2
            if "enhanced_distance" not in params.keys():
                params["enhanced_distance"] = False
            if "save_recons" not in params.keys():
                params["save_recons"] = save_recons
            if "secret_file" not in params.keys():
                params["secret_file"] = args.secret
            identify_heatwave_days(params)
            message = f"Indentify Heat wave period (flag -i   --identifyhw) for {params['name'][1:-1]} is not compatible with default 'method' ('day') and this will not be executed"
            if teleg:
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                print(requests.get(url).json())
            warnings.warn(message)
        else:
            runComparison(params)
        # Everything finished
        message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
        if teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
            requests.get(url).json()
        warnings.warn(message)
    elif method in ['days', 'seasons', 'execs', 'latents', 'seasons-execs', 'latents-execs', 'latents-seasons-execs']:
        # Read file
        ## Try-except-else to open file and re-write params
        try:
            file_params = open(file_params_name)
        except:
            message = OSError(f'File {file_params_name} not found. Your method need a configuration parameters file.')
            if teleg:
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
                requests.get(url).json()
            raise message
        else:
            params_multiple = json.load(file_params)
            params_multiple["teleg"] = teleg
            params_multiple["period"] = period
            if "secret_file" not in params_multiple.keys():
                params_multiple["secret_file"] = args.secret
            if ident:
                heatwave_period = identify_heatwave_days(params_multiple)
                params_multiple["data_of_interest_init"] = heatwave_period
                params_multiple["data_of_interest_end"] = heatwave_period
            else:
                if params_multiple["per_what"] == "per_month":
                    heatwave_period = (pd.date_range(start=params_multiple["data_of_interest_init"], end=params_multiple["data_of_interest_end"], freq='MS')).to_numpy()
                elif params_multiple["per_what"] == "per_week":
                    heatwave_period = np.arange(datetime.datetime.strptime(params_multiple["data_of_interest_init"], '%Y-%m-%d'), datetime.datetime.strptime(params_multiple["data_of_interest_end"], '%Y-%m-%d')+datetime.timedelta(days=1), datetime.timedelta(weeks=1))
                else:
                    heatwave_period = np.arange(datetime.datetime.strptime(params_multiple["data_of_interest_init"], '%Y-%m-%d'), datetime.datetime.strptime(params_multiple["data_of_interest_end"], '%Y-%m-%d')+datetime.timedelta(days=1), datetime.timedelta(days=1))
                heatwave_period = np.array(list(map(pd.Timestamp, heatwave_period)))
                params_multiple["data_of_interest_init"] = heatwave_period
                params_multiple["data_of_interest_end"] = heatwave_period
            params = params_multiple.copy()
            file_params.close()
        if verb is not None:
            params["verbose"] = verb
        if "p" not in params.keys():
            params["p"] = 2
        if "enhanced_distance" not in params.keys():
            params["enhanced_distance"] = False
        if "save_recons" not in params.keys():
            params["save_recons"] = save_recons
        # Methods with configfile
        if method == 'days':
            for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                if idx == 0:
                    params["load_AE"] = False
                if idx > 0 and not params["load_AE"]:
                    params["load_AE"] = True
                params["data_of_interest_init"] = init
                params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                if params["save_recons"]:
                    reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                    reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                    reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                    reconstructions_Post_AE.append(reconstruction_Post_AE)
            save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
            # Exec finished
            message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
            if teleg:
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                requests.get(url).json()
            warnings.warn(message)
        elif method == 'seasons':
            for season in params_multiple["season"]:
                params["season"] = season
                for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                    if idx == 0:
                        params["load_AE"] = False
                    if idx > 0 and not params["load_AE"]:
                        params["load_AE"] = True
                    params["data_of_interest_init"] = init
                    params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                    reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                    if params["save_recons"]:
                        reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                        reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                        reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                        reconstructions_Post_AE.append(reconstruction_Post_AE)
                save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                # Exec finished
                message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                    requests.get(url).json()
                warnings.warn(message)
        elif method == 'execs':
            if 'n_execs' in params.keys():
                n_execs = params['n_execs']
            for i in range(n_execs):
                params["name"] = f'-exec{i}{params_multiple["name"]}'
                for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                    if idx == 0:
                        params["load_AE"] = False
                    if idx > 0 and not params["load_AE"]:
                        params["load_AE"] = True
                    params["data_of_interest_init"] = init
                    params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                    reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                    if params["save_recons"]:
                        reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                        reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                        reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                        reconstructions_Post_AE.append(reconstruction_Post_AE)
                save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                # Exec finished
                message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                    requests.get(url).json()
                warnings.warn(message)
        elif method == 'latents':
            for latent in params_multiple["latent_dim"]:
                params["latent_dim"] = latent
                for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                    if idx == 0:
                        params["load_AE"] = False
                    if idx > 0 and not params["load_AE"]:
                        params["load_AE"] = True
                    params["data_of_interest_init"] = init
                    params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                    reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                    if params["save_recons"]:
                        reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                        reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                        reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                        reconstructions_Post_AE.append(reconstruction_Post_AE)
                save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                # Exec finished
                message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                if teleg:
                    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                    requests.get(url).json()
                warnings.warn(message)
        elif method == 'seasons-execs':
            if 'n_execs' in params.keys():
                n_execs = params["n_execs"]
            for season in params_multiple["season"]:
                params["season"] = season
                for i in range(n_execs):
                    params["name"] = f'-exec{i}{params_multiple["name"]}'
                    for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                        if idx == 0:
                            params["load_AE"] = False
                        if idx > 0 and not params["load_AE"]:
                            params["load_AE"] = True
                        params["data_of_interest_init"] = init
                        params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                        reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                        if params["save_recons"]:
                            reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                            reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                            reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                            reconstructions_Post_AE.append(reconstruction_Post_AE)
                    save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                    # Exec finished
                    message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                    if teleg:
                        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                        requests.get(url).json()
                    warnings.warn(message)
        elif method == 'latents-execs':
            if 'n_execs' in params.keys():
                n_execs = params['n_execs']
            for latent in params_multiple["latent_dim"]:
                params["latent_dim"] = latent
                for i in range(n_execs):
                    params["name"] = f'-exec{i}{params_multiple["name"]}'
                    for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                        if idx == 0:
                            params["load_AE"] = False
                        if idx > 0 and not params["load_AE"]:
                            params["load_AE"] = True
                        params["data_of_interest_init"] = init
                        params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                        reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                        if params["save_recons"]:
                            reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                            reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                            reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                            reconstructions_Post_AE.append(reconstruction_Post_AE)
                    save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                    # Exec finished
                    message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                    if teleg:
                        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                        requests.get(url).json()
                    warnings.warn(message)
        elif method == 'latents-seasons-execs':
            if 'n_execs' in params.keys():
                n_execs = params['n_execs']
            for latent in params_multiple["latent_dim"]:
                params["latent_dim"] = latent
                for season in params_multiple["season"]:
                    params["season"] = season
                    for i in range(n_execs):
                        params["name"] = f'-exec{i}{params_multiple["name"]}'
                        for idx, init in enumerate(params_multiple["data_of_interest_init"]):
                            if idx == 0:
                                params["load_AE"] = False
                            if idx > 0 and not params["load_AE"]:
                                params["load_AE"] = True
                            params["data_of_interest_init"] = init
                            params["data_of_interest_end"] = params_multiple["data_of_interest_end"][idx]
                            reconstruction_Pre_Analog, reconstruction_Post_Analog, reconstruction_Pre_AE, reconstruction_Post_AE = runComparison(params)
                            if params["save_recons"]:
                                reconstructions_Pre_Analog.append(reconstruction_Pre_Analog)
                                reconstructions_Post_Analog.append(reconstruction_Post_Analog)
                                reconstructions_Pre_AE.append(reconstruction_Pre_AE)
                                reconstructions_Post_AE.append(reconstruction_Post_AE)
                        save_reconstruction(params, reconstructions_Pre_Analog, reconstructions_Post_Analog, reconstructions_Pre_AE, reconstructions_Post_AE)
                        # Exec finished
                        message = f'Execution of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
                        if teleg:
                            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
                            requests.get(url).json()
                        warnings.warn(message)
    else:
        message = ValueError(f"Not recognized {method} method. The available methods are 'day' (default), 'days', 'seasons', 'execs', 'latents', 'seasons-execs', 'latents-execs' or 'latents-seasons-execs'")
        if teleg:
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(message).__name__+'</b>] '+user_name+': '+str(message)}"
            requests.get(url).json()
        raise message
    # Post-process
    post_process(file_params_name, is_atribution = period == 'both')
    message = f'Post process of method {args.method} for {params["name"]} with arch {params["arch"]} and latent dim {params["latent_dim"]} has finished successfully'
    if teleg:
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={'[WARN]: '+message}"
        requests.get(url).json()
    warnings.warn(message)
    return

def va_am(ident:bool=False, method:str='day', config_file:str='params.json', secret_file:str='secrets.txt', verbose:bool=False, teleg:bool=False, period:str='both', save_recons:bool=False):
    """
    va_am

    Equivalent to main function. Its scope is to provide a way to perform the same procedures as `main` function, but by importing it in another python code. 
    
    Parameters
    ----------
    ident: bool
        Value of flag to performs the identification period task or not.
    method: str
        Specify an method to execute between: `day` (default), `days`, `seasons`, `execs`, `latents`, `seasons-execs`, `latents-execs` or `latents-seasons-execs`
    config_file: str
        The default name of the params/configuration file.
    secret_file: str
        The default name of the Telegram bot informatin file.
    verbose: bool
        Value of flag that indicates if verbosity information should be show or not.
    teleg: bool
        Value of flag for sending Exceptions to Telegram bot.
    period: str
        Specify the period where to perform the operation between `both` (default), `pre` or `post`.
    save_recons: bool
        Value of flag for saving or not the reconstrucion information in an .nc file.
        
    Returns
    ----------
    """
    # Default parameters
    params = {
        "season":                   "all",
        "name":                     "-per_day-france2003",
        "latitude_min":             32,
        "latitude_max":             70,
        "longitude_min":             -30,
        "longitude_max":              30,
        "pre_init":                 '1851-01-06',
        "pre_end":                  '1950-12-31',
        "post_init":                '1951-01-01',
        "post_end":                 '2014-12-28',
        "data_of_interest_init":    '2003-08-09',
        "data_of_interest_end":     '2003-08-09',
        "load_AE":                  True,
        "load_AE_pre":              True, # only for AE post comparison
        "file_AE_pre":              './models/AE_pre.h5',
        "file_AE_post":             './models/AE_post.h5',
        "latent_dim":               600,
        "use_VAE":                  True,
        "with_cpu":                 False,
        "n_epochs":                 300,
        "k":                        20,
        "iter":                     20,
        "interest_region":          [4,8,12,19],
        "resolution":               2,
        "interest_region_type":     "idx", # idx or coord
        "per_what":                 "per_day", # per_day or per_week
        "remove_year":              False,
        "replace_choice":           False,
        "arch":                     5,
        "teleg":                    False,
        "verbose":                  False
    }
    params_multiple = None
    file_params_name = config_file
    n_execs = 5
    verb = None
    token = None
    chat_id = None
    user_name = None
    if teleg:
        with open(secret_file) as f:
            token = f.readline().strip()
            chat_id = f.readline().strip()
            user_name = f.readline().strip()
        f.close()
    try:
        _step_loop_without_args(params, params_multiple, file_params_name, n_execs, ident, verb, teleg, token, chat_id, user_name, save_recons, period, method)
    except Exception as ex:
        if teleg:
            message = traceback.format_exc().replace("<","").replace(">","")
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(ex).__name__+'</b>] '+user_name+': '+str(message)}"
            requests.get(url).json()
        raise ex

def main():
    """
    Main

    Function prepared for runing and managing the program functionality. It use the argparse module to manage the execution of va_am.py as a bash function. To see help use: 
    
    .. code-block:: python
    
        python va_am.py -h


    """

    # Parser initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--identifyhw", dest='ident', action="store_true", help="Flag. If true, first, identify the heatwave period and, then, apply the 'method' if is one of: \n 'days', 'seasons', 'execs', 'latents', 'seasons-execs', 'latents-execs' or 'latents-seasons-execs'")
    parser.add_argument("-m", "--method", dest='method', help="Specify an method to execute between: \n 'day' (default), 'days', 'seasons', 'execs', 'latents', 'seasons-execs', 'latents-execs' or 'latents-seasons-execs'")
    parser.add_argument("-f", "--configfile", dest='conf', help="JSON file with configuration of parameters. If not specified and 'method' require the file, it will be searched at 'params.json'")
    parser.add_argument("-sf", "--secretfile", dest='secret', help="Path to TXT file with needed information of the Telegram bot to use to WARN and advice about Exceptions. If not specified and 'method' require the file, it will be searched at 'secret.txt'")
    parser.add_argument("-v", "--verbose", dest='verb', action="store_true", help="Flag. If true, overwrite verbose param.")
    parser.add_argument("-t", "--teleg", dest='teleg', action="store_true", help="Flag. If true, exceptions and warnings will be sent to Telegram Bot.")
    parser.add_argument("-p", "--period", dest='period', help="Specify the period where to perform the operations between: \n 'both' (default), 'pre' or 'post'")
    parser.add_argument("-sr", "--savereconstruction", dest='save_recons', action="store_true", help="Flag. If true, the reconstruction per iteration would be saved in ./data/ folder as an reconstruction-[name]-[day]-[period]-[AM/VA-AM].nc file.")
    args = parser.parse_args()
    
    # Default parameters
    params = {
        "season":                   "all",
        "name":                     "-per_day-france2003",
        "latitude_min":             32,
        "latitude_max":             70,
        "longitude_min":             -30,
        "longitude_max":              30,
        "pre_init":                 '1851-01-06',
        "pre_end":                  '1950-12-31',
        "post_init":                '1951-01-01',
        "post_end":                 '2014-12-28',
        "data_of_interest_init":    '2003-08-09',
        "data_of_interest_end":     '2003-08-09',
        "load_AE":                  True,
        "load_AE_pre":              True, # only for AE post comparison
        "file_AE_pre":              './models/AE_pre.h5',
        "file_AE_post":             './models/AE_post.h5',
        "latent_dim":               600,
        "use_VAE":                  True,
        "with_cpu":                 False,
        "n_epochs":                 300,
        "k":                        20,
        "iter":                     20,
        "interest_region":          [4,8,12,19],
        "resolution":               2,
        "interest_region_type":     "idx", # idx or coord
        "per_what":                 "per_day", # per_day or per_week
        "remove_year":              False,
        "replace_choice":           False,
        "arch":                     5,
        "teleg":                    False,
        "verbose":                  False
    }
    params_multiple = None
    file_params_name = 'params.json'
    secret_file = 'secret.txt'
    n_execs = 5
    ident = False
    verb = None
    teleg = False
    save_recons = False

    # Identify Heat wave period
    if args.ident is not None:
        ident = args.ident
    if args.verb is not None:
        verb = args.verb
    if args.teleg is not None:
        teleg = args.teleg
    if args.save_recons is not None:
        save_recons = args.save_recons
    if args.secret is not None:
        secret_file = args.secret
    else:
        args.secret = secret_file
    token = None
    chat_id = None
    user_name = None
    if teleg:
        with open(secret_file) as f:
            token = f.readline().strip()
            chat_id = f.readline().strip()
            user_name = f.readline().strip()
        f.close()
    try:
        _step_loop(params, params_multiple, file_params_name, n_execs, ident, verb, teleg, token, chat_id, user_name, save_recons, args)
    except Exception as ex:
        if teleg:
            message = traceback.format_exc().replace("<","").replace(">","")
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&parse_mode=HTML&text={'[<b>'+type(ex).__name__+'</b>] '+user_name+': '+str(message)}"
            requests.get(url).json()
        raise ex


if __name__ == "__main__":
    main()
