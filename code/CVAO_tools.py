#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:52:32 2020

@author: mjr583
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.offsetbox import AnchoredText
import netCDF4
import datetime     


def get_dataset_from_merge(d, timestep='M'):
    filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
    filen = filepath+'20191007_CV_Merge.csv'
    dtf = pd.read_csv(filen, index_col=0,dtype={'Airmass':str, 'New_Airmass':str})
    dtf.index = pd.to_datetime(dtf.index,format='%d/%m/%Y %H:%M')

    filen=filepath+'cv_ovocs_2018_M_Rowlinson.csv'
    odf = pd.read_csv(filen, index_col=0)
    odf.index = pd.to_datetime(odf.index,format='%d/%m/%Y %H:%M')
    
    cols=list(dtf) ; ocols = list(odf)
    for col in cols:
        try:
            dtf[col] = dtf[col].loc[~(dtf[col] <= 0. )]
        except:
            pass
    for col in ocols:
        odf = odf.loc[~(odf[col] <= 0.)]
    cols=cols+ocols
        
    df=dtf.resample('H').mean()
    odf=odf.resample('H').mean()
    dtf=pd.concat([df,odf], axis=1, sort=False)
    dtf = dtf[d['merge_pref']+d['species']+d['merge_suff']]
    dtf = pd.DataFrame(dtf)
    dtf.columns = [d['species']]

    df = dtf.resample(timestep).mean()
    dates = df.index
    return df, dates


def get_dataset_as_df(D, timestep='M'):    
    dataset = netCDF4.Dataset(D['ebas_url'])
    time = dataset.variables['time'][:]
    time=np.array(timestamp_to_date(time))
    mean = dataset.variables[D['ebas_var_name']][:] 
    
    dtf = pd.DataFrame(mean)
    dtf.index = time
    dtf.columns = [D['species']]

    df = dtf.resample(timestep).mean()
    dates=df.index
    
    return dataset, df, dates


def get_nox_data(d, timestep='M'):
    filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
    filen = filepath+'Hourly_NOx_2007-2018_with_flags.csv'
    df = pd.read_csv(filen, index_col=0)
    df.index = pd.to_datetime(df.index,format='%Y-%m-%d %H:%M:%S')
    
    cols=list(df) 
    for col in cols:
        try:
            df[col] = df[col].loc[~(df[col] < 0. )]
        except:
            pass

    if d['species'] == 'NOx':
        no_flag = df.loc[df['NO_Ozone_corrected_Flag'] == 0.0]
        no = no_flag['NO_pptV_Ozone_corrected']
    
        no2_flag = df.loc[df['NO2_Ozone_corrected_Flag'] == 0.0]
        no2 = no2_flag['NO2_pptV_Ozone_corrected']
        
        df = pd.DataFrame(no+no2)
    else:
        df = df.loc[df[d['species']+'_Ozone_corrected_Flag'] == 0.0]
        df = df[d['species']+'_pptV_Ozone_corrected']
        df = pd.DataFrame(df)
    df.columns = [d['species']]
    df = df.resample(timestep).mean()
    dates = df.index
    return df, dates


def get_start_year(dataset, d):
    
    try:
        start_year = eval(dataset.comment)['Startdate'][:4]
    except:
        start_year=d['start_year'] 
    from datetime import datetime
    end_year=datetime.today().strftime('%Y')
    years=np.arange(int(start_year),int(end_year))
    return start_year, end_year, years


def get_start_year_merge(df):
    start_year = df.index[0].strftime('%Y')
    end_year = df.index[-1].strftime('%Y')
    years=np.arange(int(start_year),int(end_year))
    return start_year, end_year, years


def timestamp_to_date(times):
    new_date=[]
    for t, tt in enumerate(times):
        x = (datetime.datetime(1900,1,1,0,0) + datetime.timedelta(tt-1))
        new_date.append(x)
    return new_date


def curve_fit_function(df,X,Y, start, timestep='monthly'):
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    a = np.mean(Y.resample('A').mean())
    b = z
    c2 = .00001
    A1 = 5.1 
    A2 = 0.5
    B1 = 0.1
    B2 = 0.5
    s1 = 1/8760 * 2*np.pi
    s2 = 5000/8760 * 2*np.pi   
    s3 = 12/24 * np.pi
    s4 = 2/24 * np.pi
    if timestep == 'monthly' or timestep == 'Monthly' or timestep =='M':
        def re_func(t,a,b,c2,A1,s1,A2,s2):
            return a + b*t + c2*t**2 + A1*np.sin(t/12*2*np.pi + s1) + A2*np.sin(2*t/12*2*np.pi + s2)
        guess = np.array([a, b, c2, A1, s1,A2,s2])
        c,cov = curve_fit(re_func, X, Y, guess)
        n = len(X)
        y = np.empty(n)
        for i in range(n):
            y[i] = re_func(X[i],c[0],c[1],c[2],c[3],c[4],c[5],c[6])

    elif timestep == 'hourly' or timestep == 'Hourly' or timestep == 'H':
        def re_func(t,a,b,c2,A1,s1,A2,s2, B1, s3, B2, s4):
            return a + b*t + c2*t**2 + A1*np.sin(t/8760*2*np.pi + s1) + A2*np.sin(2*t/8760*2*np.pi + s2) + B1*np.sin(t/24*2*np.pi + s3) + B2*np.sin(2*t/24*2*np.pi + s4)
        guess = np.array([a, b, c2, A1, s1,A2,s2,B1,s3,B2,s4])
        c,cov = curve_fit(re_func, X, Y, guess, maxfev=500000)
        n = len(X)
        y = np.empty(n)
        for i in range(n):
            y[i] = re_func(X[i],c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10])
    else:
        raise Exception('Invalid timestep argument, must be daily ("D") or monthly ("M")')
    var = c[0] + c[1]*X + c[2]*X**2
    rmse = np.round(np.sqrt(mean_squared_error(Y,re_func( X, *c))),2)
    r2 = np.round(r2_score(Y,re_func( X, *c))*100,1)
    return y, var, z, rmse, r2, c

def plot_trend_breakdown(d, X, Y, c, start_year,times,timestep,savepath=''):
    detrended = [Y[i] - c[1]*i for i in range(0, len(X))]
    detrended2 = np.array([detrended[i] - c[2]*i**2 for i in range(0, len(X))])
    plt.plot(times,Y, label='Obervations')
    plt.plot(times,detrended,label='Detrend b')
    plt.plot(times,detrended2,label='Detrend b + c')
    plt.ylabel(d['abbr']+' ('+d['unit']+')')
    plt.legend()
    plt.savefig(savepath+d['species']+'_trend_breakdown.png', dpi=300)
    plt.close()
    return


def plot_fitted_curve(d,X,Y,output,times,timestep, savepath=''):
    if timestep=='H':
        t='hourly'
        s=0.05
        l=0.2
    elif timestep=='M':
        t='monthly'
        s=0.5
        l=2
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(times, Y, 'ro', markersize=s)
    ax1.plot(times, output[0], '--', linewidth=l )
    ax1.plot(times, output[1], 'k')
    ax1.set_ylabel(d['abbr']+' ('+d['unit']+')')
    ax1.set_yscale(d['yscale'])
    if output[2]>0.: # moves legend based on direction of trend
        loc=2
    elif output[2]<0.:
        loc=1
    txt = AnchoredText('RMSE='+str(output[3])+' '+d['unit']+'\n$r^2$='+str(output[4])+'%', loc=loc)
    ax1.add_artist(txt)
    plt.savefig(savepath+d['species']+'_'+t+'_mean.png', dpi=300)
    plt.close()
    return


def plot_residual(d,X,Y,output,times,timestep, savepath='', nbins=25):
    if timestep=='H':
        t='hourly'
    elif timestep=='M':
        t='monthly'
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(times, Y-output[0], 'g', label='Residual')
    ax1.set_ylabel(d['abbr']+' ('+d['unit']+')')
    
    ax2 = fig.add_subplot(212)
    ax2.hist( Y-output[0], bins=nbins, density=True, alpha=0.6, color='g')
    ax2.set_xlabel(d['abbr']+' ('+d['unit']+')')
    ax2.set_ylabel('freq')
    fig.suptitle('Residual - '+t+' means')
    plt.savefig(savepath+d['species']+'_'+t+'_residual.png', dpi=300)
    plt.close()
    return


def remove_nan_rows(df,times):
    '''
    Remove nan values from dataset.

    Parameters
    ----------
    df (array): Variable dataset
    times (array): Index/timestep for df

    Returns
    -------
    X (array) : Nan corrected array of time integers
    Y (pd.Series) : Nan corrected concnetrations
    time (array) : Nan corrected datetime array
    Notes
    '''
    XX = np.arange(len(df))
    idx = np.isfinite(df)    
    Y = df[idx]
    X = XX[idx]
    time = times[idx]
    return X, Y, time


def plot_trend_with_func_from_dict(d, timestep='M', force_merge=False,\
                                   savepath='/users/mjr583/scratch/NCAS_CVAO/trend_plots/'):
    '''
    Get CVAO for data in dict and plot over time with curved fit.
    Can extract data from merge_file or url. 

    Parameters
    ----------
    d (dict): Dictionary of desired species with relevant fields
    Timestep (str): Timestep for trend fit, hourly ("H") or monthly ("M")
    force_merge (bool): Takes all datasets from merge file, default is to attmpet url first.
    savepath (str): path to save directory for plots.

    Returns
    -------
    (plots)

    Notes
    '''
    for i in d:
        if force_merge == False:
            if i == 'NO' or i=='NO2' or i=='NOx':
                df, time = get_nox_data(d[i], timestep=timestep)
                start, end, years = get_start_year_merge(df)  
            else:
                try:
                    dataset, df, time = get_dataset_as_df(d[i], timestep=timestep)
                    start, end, years = get_start_year(dataset, d[i]) 
                except:
                    df, time = get_dataset_from_merge(d[i], timestep=timestep)
                    start, end, years = get_start_year_merge(df)                
        elif force_merge== True:
            df, time = get_dataset_from_merge(d[i], timestep=timestep)
            start, end, years = get_start_year_merge(df) 
            plt.plot(df)
            
        X, Y, time = remove_nan_rows(df[i], time)
        if X.size == 0:
            print('No values for ', df.columns[0])
            pass
        elif Y.size <=5:
            print('Insufficient data for ', i)
            pass
        else:
            output = curve_fit_function(df, X, Y, start, timestep=timestep)
            plot_fitted_curve(d[i],X, Y, output, times=time, timestep=timestep, savepath=savepath)
            plot_residual(d[i],X, Y, output, times=time, timestep=timestep, savepath=savepath)
            if timestep=='M':
                plot_trend_breakdown(d[i], X, Y, output[5], start, times=time, timestep=timestep,savepath=savepath)
            print(i, 'done')
    return 