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

def get_dataset_from_merge():
    filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
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
    
    df=dtf.resample('T').mean()
    odf=odf.resample('T').mean()
    
    dtf=pd.concat([df,odf], axis=1, sort=False)
    return dtf

stop
def get_dataset_as_df(D, timestep='M'):    
    dataset = netCDF4.Dataset(D['url'])
    time = dataset.variables['time'][:]
    new_date=np.array(timestamp_to_date(time))
    mean = dataset.variables[D['var_name']][:] 

    
    dtf = pd.DataFrame(mean)#, columns=d['species'])
    dtf.index = new_date
    dtf.columns = [D['species']]
    df = dtf.resample(timestep).mean()
    
    return dataset, df, new_date


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


def curve_fit_function(df,X,Y, start, timestep='monthly'):
    ''' Guess of polynomial terms '''
    z, p = np.polyfit(X, Y, 1)
    a = np.nanmean(df[start].resample('A').mean())  #31.08
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
    #c[0]=32.49
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


def plot_fitted_curve(d, X,Y,output, times):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(times, Y, 'ro', markersize=0.5)
    ax1.plot(times, output[0], '--')
    #ax1.plot(X, output[1])
    ax1.set_ylabel(d['abbr']+' '+d['unit'])
    if output[2]>0.: # moves legend based on direction of trend
        loc=2
    elif output[2]<0.:
        loc=1
    txt = AnchoredText('RMSE='+str(output[3])+' '+d['unit']+'\n$r^2$='+str(output[4])+'%', loc=loc)
    ax1.add_artist(txt)
    #plt.xticks(np.arange(0, len(X), len(X)/len(years)), years)
    plt.savefig(savepath+d['species']+'_nonlin_regression.png')
    plt.close()
    return


def remove_nan_rows(df,times):
    XX = np.arange(len(df))
    idx = np.isfinite(df)    
    Y = df[idx]
    X = XX[idx]
    time = times[idx]
    return X, Y, time


filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'
d = {
     'O3' : {   'ceda_url' : 'http://dap.ceda.ac.uk/thredds/dodsC/badc/capeverde/data/cv-tei-o3/2019/ncas-tei-49i-1_cvao_20190101_o3-concentration_v1.nc',
                'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc',
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
    timestep='H'
    dataset, df, time = get_dataset_as_df(d[i], timestep=timestep)
    start, end, years = get_start_year(dataset, d[i]) 
    X, Y, time = remove_nan_rows(df[i], time)
    output = curve_fit_function(df, X, Y, start, timestep=timestep)
    plot_fitted_curve(d[i],X, Y, output, times=time)
    plot_trend_breakdown(d[i], X, Y, output[5], start)
    print(i, 'done')
    stop