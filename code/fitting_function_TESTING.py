#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:54:35 2019

@author: mjr583
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,4)
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.offsetbox import AnchoredText

filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'

filen = filepath+'20191007_CV_Merge.csv'
dtf = pd.read_csv(filen, index_col=0,dtype={'Airmass':str})
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
hourly=dtf.resample('M').mean()
ohourly=odf.resample('M').mean()
dtf=pd.concat([hourly,ohourly], axis=1, sort=False)

S = ['O3','CO','CH4','CO2','NO','propane','ethane','ethene','acetone','meoh','acetaldehyde',\
     'CHBr3', 'CH2Br2', 'CCl4', 'CH2I2', 'CH3I', 'CH2ICl',\
     'CHCl3', 'CH2BrCl', 'CH2IBr', 'CHBr2Cl',]
#S = ['acetone']
for n,species in enumerate(S):
    print(species)
    dtf['2009-07-01' : '2009-09-30'] = np.nan 
    if species == 'O3':
        print('a')
        suff = '' ; unit=' (ppbv)' ; start_year='2007' ## O3 data begins in 2007
    elif species == 'CO':
        print('b')
        suff = ' (ppbV)' ; unit=suff ; start_year='2008' ## CO data begins in 2008
    elif species == 'CO2':
        print('c')
        suff = ' (ppmV)' ; unit=suff ; start_year='2015' ## CO2 data begins in 2015
    elif species == 'CH4':
        print('d')
        suff = ' all (ppbV)' ; unit=suff[-7:] ; start_year='2007' ## Regular CH4 data begins in 2015
    elif species == 'NO':
        print('e')
        suff = ' (pptV)' ; unit = suff ; start_year='2008' 
    elif species=='acetone': 
        suff = '' ; unit = ' (pptV)' ; start_year='2015' 
    elif species=='meoh': 
        suff = '' ; unit = ' (pptV)' ; start_year='2015' 
    elif species=='acetaldehyde': 
        suff = '' ; unit = ' (pptV)' ; start_year='2015' 
    elif n>10:
        suff = ' (pptV)' ; unit = suff ; start_year='2015' 
    elif 'ethane' or 'ethene' or 'propane' in species:
        print('f')
        suff = ' (pptV)' ; unit = suff ; start_year='2010' 
    else:
        start_year='2007' ## CVAO dataset begins in 2007. For specific species add if statement to adjust start_year
        unit='ppb'
    
    spec = species+suff
    df = dtf[start_year:]
    df[spec] = df[spec].fillna(method='bfill')
    
    idx = np.isfinite(df[spec])
    Y = df[spec][idx] 
    X = np.arange(len(Y))
    
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    a = np.nanmean(df[spec][start_year].resample('A').mean())  #31.08
    b = z
    c2 = .00001
    A1 = 5.1 
    A2 = 0.5
    s1 = 1/12 * 2*np.pi
    s2 = 7/12 * 2*np.pi   
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
    
    var = c[0] + c[1]*X + c[2]*X**2
    rmse = np.round(np.sqrt(mean_squared_error(Y,re_func( X, *c))),2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, 'ro', markersize=0.5)
    ax1.plot(X, y, '--')
    ax1.plot(X, var)
    plt.text(.75,.1, 'RMSE='+str(rmse), fontsize=14,transform=ax1.transAxes,)#, boxstyle='round,pad=1'))
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.close()
    
    detrended = [Y[i] - c[1]*i for i in range(0, len(X))]
    detrended2 = np.array([detrended[i] - c[2]*i**2 for i in range(0, len(X))])
    
    plt.plot(X,Y, label='Obervations')
    plt.plot(X,detrended,label='Detrend b')
    plt.plot(X,detrended2,label='Detrend b + c')
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.legend()
    plt.close()
    
    ds = pd.DataFrame(detrended2[:])
    ds.index = df[idx].index
    seas = ds.groupby(ds.index.month).mean()
    
    ####### Redo with monthly means replacing NaNs ###########
    df = dtf[start_year:].resample('M').mean() 
    for i,ii in enumerate(df.index.month):
        if df[spec][i] != df[spec][i]:
            year=df.index[i].year
            try:
                val = ( df[spec][str(year-1)][ii] + df[spec][str(year+1)][ii] ) / 2
                df[spec][i] = val
            except:
                try:
                    val = df[spec][str(year-1)][ii] 
                except:
                    try:
                        val = df[spec][str(year+1)][ii]  
                    except:
                        pass
    
    idx = np.isfinite(df[spec])
    Y = df[spec][idx] 
    X = np.arange(len(Y))
    #pp = np.linspace(1,5.1,np.round(len(Y)*0.75),0) ; qq = np.linspace(5.1,0.9,np.round(len(Y)*0.25,0)) ; rr = np.concatenate((pp,qq))
    #Y = Y * rr
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    
    a = np.nanmean(df[spec][start_year].resample('A').mean())  #31.08
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
    ax1.set_ylabel(species+unit)
    txt = AnchoredText('RMSE='+str(rmse)+' '+unit[2:-2]+'\n$r^2$='+str(r2)+'%', loc=2)
    ax1.add_artist(txt)
    #plt.text(.75,h, 'RMSE='+str(rmse)+' '+unit[2:-2]+'\n$r^2$='+str(r2)+'%', fontsize=12,transform=ax1.transAxes,)#, boxstyle='round,pad=1'))
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.savefig(savepath+species+'_nonlin_regression.png')
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
    print(species, 'done')