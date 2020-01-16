#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:40:27 2020

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

import CVAO_tools as CV
start='1988' ; timestep = 'H' ; i='O3' ; savepath='/users/mjr583/scratch/MC_'

def curve_fit_function(df,X,Y, start, timestep='monthly'):
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    a = np.mean(Y[start].resample('A').mean())
    print(a)
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
        c,cov = curve_fit(re_func, X, Y,guess)
        print(c)
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


url='http://thredds.nilu.no/thredds/dodsC/ebas/IE0031R.19880401000000.20140101000000.uv_abs.ozone.air.25y.1h.GB02L_uv_abs_31.GB02L_uv_abs..nc'
dataset = netCDF4.Dataset(url)
time = dataset.variables['time'][:]
time=np.array(CV.timestamp_to_date(time))
mean = dataset.variables['ozone'][:] / 1.96
dtf = pd.DataFrame(mean)
dtf.index = time
dtf.columns = [i]

url='http://thredds.nilu.no/thredds/dodsC/ebas/IE0031R.20130101000000.20190724000000.uv_abs.ozone.air.6y.1h.GB02L_uv_IE0031.GB02L_uv_abs..nc'
dataset = netCDF4.Dataset(url)
time = dataset.variables['time'][:]
time=np.array(CV.timestamp_to_date(time))
mean = dataset.variables['ozone'][:] / 1.96
dtf2 = pd.DataFrame(mean)
dtf2.index = time
dtf2.columns = [i]

df = pd.concat([dtf,dtf2])


df = df[start:].resample(timestep).mean()
dates=df.index

X, Y, time = CV.remove_nan_rows(df[i], dates)  
d = { 'O3' : {'species': 'O3',
  'longname': 'ozone',
  'abbr': '$O_3$',
  'unit': 'ppbv',
  'scale': '1e9',
  'yscale': 'linear',
  'start_year': '2006',
  'merge_pref': '',
  'merge_suff': '',
  'instrument': ''}}


output = CV.curve_fit_function(df, X, Y, start, timestep=timestep)
CV.plot_fitted_curve(d[i],X, Y, output, times=time, timestep=timestep, savepath='/users/mjr583/scratch/MC_')
CV.plot_residual(d[i],X, Y, output, times=time, timestep=timestep, savepath=savepath)
if timestep=='M':
    CV.plot_trend_breakdown(d[i], X, Y, output[5], start, times=time, timestep=timestep,savepath=savepath)
print(i, 'done')
