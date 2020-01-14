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

def get_dataset_as_df(D, timestep='M'):    
    dataset = netCDF4.Dataset(D['url'])
    time = dataset.variables['time'][:]
    new_date=np.array(timestamp_to_date(time))
    mean = dataset.variables[D['var_name']][:] 

    
    dtf = pd.DataFrame(mean)#, columns=d['species'])
    dtf.index = new_date
    dtf.columns = [D['species']]
    df = dtf.resample(timestep).mean()
    
    return dataset, df


def get_start_year(dataset, d):
    try:
        start_year = eval(dataset.comment)['Startdate'][:4]
    except:
        start_year=d['start_year'] 
    from datetime import datetime
    end_year=datetime.today().strftime('%Y')
    years=np.arange(int(start_year),int(end_year))
    return start_year, end_year, years


def timestamp_to_date(times):
    new_date=[]
    for t, tt in enumerate(times):
        x = (datetime.datetime(1900,1,1,0,0) + datetime.timedelta(tt-1))
        new_date.append(x)
    return new_date


def curve_fit_function(df,X,Y, start):
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    a = np.nanmean(df[start].resample('A').mean())  #31.08
    b = z
    c2 = .00001
    A1 = 5.1 
    A2 = 0.5
    s1 = 1/12 * 2*np.pi
    s2 = 7/12 * 2*np.pi   
    def new_func(x,m,c,c0):
        return a + b*x + c2*x**2 + A1*np.sin(x/12*2*np.pi + s1) + A2*np.sin(2*x/12*2*np.pi + s2)
            
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
    r2 = np.round(r2_score(Y,re_func( X, *c))*100,1)
    
    return y, var, z, rmse, r2, c

def plot_trend_breakdown(d, X, Y, c,start_year,\
                         savepath='/users/mjr583/scratch/NCAS_CVAO/plots/'):
    years=np.arange(int(start_year),2020)
    detrended = [Y[i] - c[1]*i for i in range(0, len(X))]
    detrended2 = np.array([detrended[i] - c[2]*i**2 for i in range(0, len(X))])
    plt.plot(X,Y, label='Obervations')
    plt.plot(X,detrended,label='Detrend b')
    plt.plot(X,detrended2,label='Detrend b + c')
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.legend()
    plt.savefig(savepath+d['species']+'_trend_breakdown.png')
    plt.close()
    return


def plot_fitted_curve(d, X,Y,output):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, 'ro', markersize=0.5)
    ax1.plot(X, output[0], '--')
    ax1.plot(X, output[1])
    ax1.set_ylabel(d['abbr']+' '+d['unit'])
    if output[2]>0.: # moves legend based on direction of trend
        loc=2
    elif output[2]<0.:
        loc=1
    txt = AnchoredText('RMSE='+str(output[3])+' '+d['unit']+'\n$r^2$='+str(output[4])+'%', loc=loc)
    ax1.add_artist(txt)
    plt.xticks(np.arange(0, len(X), 12), years)
    plt.savefig(savepath+d['species']+'_nonlin_regression.png')
    plt.close()
    return


def remove_nan_rows(df):
    XX = np.arange(len(df))
    idx = np.isfinite(df)    
    Y = df[idx]
    X = XX[idx]
    return X, Y


filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'
d = {
     'O3' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc',
                'species' : 'O3',
                'longname' : 'ozone',
                'abbr' : '$O_3$',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'var_name' : 'ozone_nmol_per_mol_amean'
                },
    'CO' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20160101000000.20190425083930.online_crds.carbon_monoxide.air.3y.1h.GB12L_CVO_Picarro_G2401.GB12L_Picarro.lev2.nc',
                'species' : 'CO',
                'longname' : 'carbon_monoxide',
                'abbr' : 'CO',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'var_name' : 'carbon_monoxide_amean'
                },
    'NO' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20170101000000.20191024143042.chemiluminescence_photolytic..air.2y.1h.GB12L_CVO_AQD_Nox.GB12L_AQD_NOx.lev2.nc',
                'species' : 'NO',
                'longname' : 'nitrogen_monoxide',
                'abbr' : 'NO',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'var_name' : 'nitrogen_monoxide_nmol_per_mol'
                },
    'NO2' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20170101000000.20191024143042.chemiluminescence_photolytic..air.2y.1h.GB12L_CVO_AQD_Nox.GB12L_AQD_NOx.lev2.nc',
                'species' : 'NO2',
                'longname' : 'nitrogen_dioxide',
                'abbr' : '$NO_2$',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'var_name' : 'nitrogen_dioxide_nmol_per_mol'
                },
    }
'''
    'NOx' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20170101000000.20191024143042.chemiluminescence_photolytic..air.2y.1h.GB12L_CVO_AQD_Nox.GB12L_AQD_NOx.lev2.nc',
                'species' : 'NOx',
                'longname' : 'nitrogen_oxides',
                'abbr' : '$NO_x$',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'var_name_a' : 'nitrogen_monoxide_nmol_per_mol',
                'var_name_b' : 'nitrogen_dioxide_nmol_per_mol'
                },
     }
'''

for i in d:
    dataset, df = get_dataset_as_df(d[i])
    start, end, years = get_start_year(dataset, d[i]) 
    '''
    XX = np.arange(len(df))
    idx = np.isfinite(df[i])    
    Y = df[idx][i]
    X = XX[idx]
    '''
    X, Y = remove_nan_rows(df[i])
    output = curve_fit_function(df, X, Y, start)
    
    plot_fitted_curve(d[i],X, Y, output)
    plot_trend_breakdown(d[i], X, Y, output[5], start)
    print(i, 'done')