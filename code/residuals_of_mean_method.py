#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:55:50 2019
Script for comparing residual of means deseasonalisation 
methods with varying bins
@author: mjr583
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (7, 7)
filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'

filen = filepath+'20191007_CV_Merge.csv'
df = pd.read_csv(filen, index_col=0,dtype={'Airmass':str})
df.index = pd.to_datetime(df.index,format='%d/%m/%Y %H:%M')

filen = filepath+'cv_ovocs_2018_M_Rowlinson.csv'
odf = pd.read_csv(filen, index_col=0)
odf.index = pd.to_datetime(odf.index,format='%d/%m/%Y %H:%M')

cols = list(df) ; ocols = list(odf)
for col in cols:
    try:
        df[col] = df[col].loc[~(df[col] <= 0.)]
    except:
        pass
for col in ocols:
    odf = odf.loc[~(odf[col] <= 0.)]
    
hourly = df.resample('H').mean()
daily = df.resample('D').mean()
monthly = df.resample('M').mean()
yearly= df.resample('Y').mean()

ohourly = odf.resample('H').mean()
odaily = odf.resample('D').mean()
omonthly = odf.resample('M').mean()
oyearly= odf.resample('Y').mean()

hourly = pd.concat([hourly,ohourly],axis=1,sort=False)
daily = pd.concat([daily,odaily],axis=1,sort=False)
monthly = pd.concat([monthly,omonthly],axis=1,sort=False)
yearly = pd.concat([yearly,oyearly],axis=1,sort=False)

years = np.arange(2006, 2006+len(yearly.index))
colors = ['#e5f5e0','#c7e9c0','R','#74c476','#41ab5d',\
              '#7fcdbb','#41b6c4','#1d91c0',\
              '#225ea8','b','#253494','#081d58','k']

## in test case just use Ozone data for now
start_year='2007'
data = daily['O3'][start_year:]
freq = 12 
#data = monthly['O3']['2015':]
years = (data.resample('Y').mean()).index.year

## Residual of mean method 

## current setup
mean = data.mean()
std = data.std()
monmean = data.groupby(data.index.month).mean()
deseas_factor = monmean / std
ds1=np.zeros(len(data))
for n,m in enumerate(data.index.month):
    ds1[n] = data[n] + (mean - monmean[m])
ds1a = pd.DataFrame(ds1[:])
ds1a.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds1 = ds1a.resample('M').mean()

## basic version - just remove the monthly mean
ds2mean = data.mean()
std = data.std()
monmean = data.groupby(data.index.month).mean()
deseas_factor = monmean / std
ds2=np.zeros(len(data)) ; ds2a=np.zeros(len(data))
for n,m in enumerate(data.index.month):
    ds2[n] = data[n] + (mean - monmean[m])
    ds2a[n] = data[n] - monmean[m]
ds2a = pd.DataFrame(ds2a[:])
ds2a.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds2b = ds2a.resample('M').mean()

## Weekly version - same but with weekly bins
data1 = daily['O3'][start_year:]
ds3mean = data1.mean()
std = data1.std()
weekmean = data1.groupby(data1.index.week).mean()
deseas_factor = weekmean / std
ds3=np.zeros(len(data1)) ; ds3a=np.zeros(len(data1))
for n,m in enumerate(data1.index.month):
    ds3[n] = data[n] + (mean - monmean[m])
    ds3a[n] = data1[n] - weekmean[m]
ds3a = pd.DataFrame(ds3a[:])
ds3a.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds3b = ds3a.resample('M').mean()

## Daily version - same but with daily bins
data1 = daily['O3'][start_year:]
ds4mean = data1.mean()
std = data1.std()
daymean = data1.groupby(data1.index.day).mean()
deseas_factor = daymean / std
ds4a=np.zeros(len(data1))
for n,m in enumerate(data1.index.month):
    ds4a[n] = data1[n] - daymean[m]
ds4a = pd.DataFrame(ds4a[:])
ds4a.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds4b = ds4a.resample('M').mean()

## Quarterly version - same but with quarterly bins
data1 = daily['O3'][start_year:]
ds5mean = data1.mean()
std = data1.std()
daymean = data1.groupby(data1.index.quarter).mean()
deseas_factor = daymean / std
ds5a=np.zeros(len(data1))
for n,m in enumerate(data1.index.quarter):
    ds5a[n] = data1[n] - daymean[m]
ds5a = pd.DataFrame(ds5a[:])
ds5a.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds5b = ds5a.resample('M').mean()

## second plot
plt.rcParams['figure.figsize'] = (12, 4)
plt.plot(ds2b.index, ds5b,label='residuals of mean, 4 bins')
plt.plot(ds2b.index, ds2b,label='residuals of mean, 12 bins')
plt.plot(ds2b.index, ds3b, label='residuals of mean, 52 bins')
plt.plot(ds2b.index, ds4b, label='residuals of mean, 365 bins')
plt.legend()
plt.ylabel('$O_3$ (ppbv)')
plt.savefig(savepath+'/residual_of_means_comp.png')
plt.close()