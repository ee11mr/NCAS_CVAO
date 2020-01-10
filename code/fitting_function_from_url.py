#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:54:35 2019

@author: mjr583"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,4)
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.offsetbox import AnchoredText
import netCDF4
import datetime        
def timestamp_to_date(times):
    new_date=[]
    for t, tt in enumerate(times):
        x = (datetime.datetime(1900,1,1,0,0) + datetime.timedelta(tt-1))
        new_date.append(x)
    return new_date
filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'

d = {
     'O3' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc',
                'longname' : 'ozone',
                'abbr' : '$O_3$',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'mean_name' : 'ozone_nmol_per_mol_amean'
                },
    'CO' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20160101000000.20190425083930.online_crds.carbon_monoxide.air.3y.1h.GB12L_CVO_Picarro_G2401.GB12L_Picarro.lev2.nc',
                'longname' : 'carbon monoxide',
                'abbr' : 'CO',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2016',
                'mean_name' : 'carbon_monoxide_amean'
                }
     }

for s in d:
    species = d[s]['longname']
    url = d[s]['url']
    dataset = netCDF4.Dataset(url)
    
    try:
        start_year = eval(dataset.comment)['Startdate'][:4]
    except:
        start_year=d[s]['start_year']
    
    time = dataset.variables['time'][:]
    new_date=np.array(timestamp_to_date(time))
    mean = dataset.variables[d[s]['mean_name']][:] 
    dtf = pd.DataFrame(mean)
    dtf.index = new_date
    dtf.columns = [s]
    df = dtf.resample('M').mean()
    
    XX = np.arange(len(df))
    idx = np.isfinite(df[s])    
    Y = df[idx][s]
    X = XX[idx]

    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    a = np.nanmean(df[s][start_year].resample('A').mean())  #31.08
    b = z
    c2 = .00001
    A1 = 5.1 
    A2 = 0.5
    s1 = 1/12 * 2*np.pi
    s2 = 7/12 * 2*np.pi   
    def new_func(x,m,c,c0):
        return a + b*x + c2*x**2 + A1*np.sin(x/12*2*np.pi + s1) + A2*np.sin(2*x/12*2*np.pi + s2)
            
    target_func = new_func
    #stop
    popt, pcov = curve_fit(target_func, X, Y)#, maxfev=20000)
    rmse = np.round(np.sqrt(mean_squared_error(Y,target_func( X, *popt))),2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, 'ro', markersize=0.5)
    ax1.plot(X, target_func(X, *popt), '--')
    plt.text(.75,.1, 'RMSE='+str(rmse), fontsize=14,transform=ax1.transAxes,)#, boxstyle='round,pad=1'))
    years=np.arange(int(start_year),2020)
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.close()
    
    ''' With curve fitting to minimise error '''
    def re_func(t,a,b,c2,A1,s1,A2,s2):
        return a + b*t + c2*t**2 + A1*np.sin(t/12*2*np.pi + s1) + A2*np.sin(2*t/12*2*np.pi + s2)
    
    guess = np.array([a, b, c2, A1, s1,A2,s2])
    c,cov = curve_fit(re_func, X, Y, guess)
    
    n = len(X)
    y = np.empty(n)
    for i in range(n):
      y[i] = re_func(X[i],c[0],c[1],c[2],c[3],c[4],c[5],c[6])
    
    var = c[0] + c[1]*X + c[2]*X**2
    rmse = np.round(np.sqrt(mean_squared_error(Y,re_func( X, *c))),2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, 'ro', markersize=0.5)
    ax1.plot(X, y, '--')
    ax1.plot(X, var)
    plt.text(.75,.1, 'RMSE='+str(rmse), fontsize=14,transform=ax1.transAxes,)#, boxstyle='round,pad=1'))
    plt.xticks(np.arange(0, len(X), 12), years)
    #plt.close()
    stop
    detrended = [Y[i] - c[1]*i for i in range(0, len(X))]
    detrended2 = np.array([detrended[i] - c[2]*i**2 for i in range(0, len(X))])
    
    plt.plot(X,Y, label='Obervations')
    plt.plot(X,detrended,label='Detrend b')
    plt.plot(X,detrended2,label='Detrend b + c')
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.legend()
    plt.savefig(savepath+d[s]['longname']+'_trend_breakdown.png')
    plt.close()

    ds = pd.DataFrame(detrended2[:])
    ds.index = df[idx].index
    seas = ds.groupby(ds.index.month).mean()
    
    ####### Redo with monthly means replacing NaNs ###########
    df = dtf[start_year:].resample('M').mean() 
    for i,ii in enumerate(df.index.month):
        if df[s][i] != df[s][i]:
            year=df.index[i].year
            try:
                val = ( df[s][str(year-1)][ii] + df[s][str(year+1)][ii] ) / 2
                df[s][i] = val
            except:
                try:
                    val = df[s][str(year-1)][ii] 
                except:
                    try:
                        val = df[s][str(year+1)][ii]  
                    except:
                        pass
    
    idx = np.isfinite(df[s])
    Y = df[s][idx] 
    X = np.arange(len(Y))
    #pp = np.linspace(1,5.1,np.round(len(Y)*0.75),0) ; qq = np.linspace(5.1,0.9,np.round(len(Y)*0.25,0)) ; rr = np.concatenate((pp,qq))
    #Y = Y * rr
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    
    a = np.nanmean(df[s][start_year].resample('A').mean())  #31.08
    b = z
    
    def new_func(x,m,c,c0):
        return a + b*x + c2*x**2 + A1*np.sin(x/12*2*np.pi + s1) + A2*np.sin(2*x/12*2*np.pi + s2)
            
    target_func = new_func
    popt, pcov = curve_fit(target_func, X, Y)#, maxfev=20000)
    rmse = np.round(np.sqrt(mean_squared_error(Y,target_func( X, *popt))),2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, 'ro', markersize=0.5)
    ax1.plot(X, target_func(X, *popt), '--')
    plt.text(.75,.1, 'RMSE='+str(rmse), fontsize=14,transform=ax1.transAxes,)#, boxstyle='round,pad=1'))
    years=np.arange(int(start_year),2020)
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.close()
    
    ''' With curve fitting to minimise error '''
    def re_func(t,a,b,c2,A1,s1,A2,s2):
        return a + b*t + c2*t**2 + A1*np.sin(t/12*2*np.pi + s1) + A2*np.sin(2*t/12*2*np.pi + s2)
    guess = np.array([a, b, c2, A1, s1,A2,s2])
    c,cov = curve_fit(re_func, X, Y, guess)
    
    n = len(X)
    y = np.empty(n)
    for i in range(n):
      y[i] = re_func(X[i],c[0],c[1],c[2],c[3],c[4],c[5],c[6])
    
    if np.nanmean(y[:12]) < np.nanmean(y[-12:]):
        h=.05
    else:
        h=.8
    var = c[0] + c[1]*X + c[2]*X**2
    rmse = np.round(np.sqrt(mean_squared_error(Y,re_func( X, *c))),2)
    r2 = np.round(r2_score(Y,re_func( X, *c))*100,1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, 'ro', markersize=0.5)
    ax1.plot(X, y, '--')
    ax1.plot(X, var)
    ax1.set_ylabel(d[s]['unit'])
    txt = AnchoredText('RMSE='+str(rmse)+' '+d[s]['unit']+'\n$r^2$='+str(r2)+'%', loc=2)
    ax1.add_artist(txt)
    #plt.text(.75,h, 'RMSE='+str(rmse)+' '+unit[2:-2]+'\n$r^2$='+str(r2)+'%', fontsize=12,transform=ax1.transAxes,)#, boxstyle='round,pad=1'))
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.savefig(savepath+d[s]['longname']+'_nonlin_regression.png')
    plt.close()
    
    detrended = [Y[i] - c[1]*i for i in range(0, len(X))]
    detrended2 = np.array([detrended[i] - c[2]*i**2 for i in range(0, len(X))])
    
    plt.plot(X,Y, label='Observations')
    plt.plot(X,detrended,label='Detrend b term')
    plt.plot(X,detrended2,label='Detrend b + c terms')
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.legend()
    plt.show()
    plt.close()
    
    ds = pd.DataFrame(detrended2[:])
    ds.index = df[idx].index
    seas = ds.groupby(ds.index.month).mean()
    print(s, 'done')