#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:47:37 2019

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
stop
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

## Quarterly bins 
mean = data.mean()
std = data.std()
quartmean = data.groupby(data.index.quarter).mean()
deseas_factor = quartmean / std
ds1=np.zeros(len(data)) 
for n,m in enumerate(data.index.quarter):
    ds1[n] = data[n] + (mean - quartmean[m])

## Monthly bins
ds2mean = data.mean()
std = data.std()
monmean = data.groupby(data.index.month).mean()
deseas_factor = monmean / std
ds2=np.zeros(len(data)) ; ds2a=np.zeros(len(data))
for n,m in enumerate(data.index.month):
    ds2[n] = data[n] + (mean - monmean[m])


## Weekly bins - same but with weekly bins
ds3mean = data.mean()
std = data.std()
weekmean = data.groupby(data.index.week).mean()
deseas_factor = weekmean / std
ds3=np.zeros(len(data)) ; ds3a=np.zeros(len(data))
for n,m in enumerate(data.index.week):
    ds3[n] = data[n] + (mean - weekmean[m])

## Daily version - same but with daily bins
ds4mean = data.mean()
std = data.std()
daymean = data.groupby(data.index.day).mean()
deseas_factor = daymean / std
ds4=np.zeros(len(data))
for n,m in enumerate(data.index.day):
    ds4[n] = data[n] - (mean - daymean[m])

## first plot
labels = ['Observations','Residuals of mean, quarterly bins',\
          'Residuals of mean, monthly bins',\
          'Residuals of mean, weekly bins',\
          'Residuals of mean, daily bins']
plt.rcParams['figure.figsize'] = (8, 8)
fig = plt.figure()

data = data.resample('M').mean()
ds1 = pd.DataFrame(ds1[:])
ds1.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds1 = ds1.resample('M').mean()[0]
ds2 = pd.DataFrame(ds2[:])
ds2.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds2 = ds2.resample('M').mean()[0]
ds3 = pd.DataFrame(ds3[:])
ds3.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds3 = ds3.resample('M').mean()[0]
ds4 = pd.DataFrame(ds4[:])
ds4.index = pd.to_datetime(daily['2007':].index,format='%d/%m/%Y')
ds4 = ds4.resample('M').mean()[0]

ds = [data,ds1,ds2,ds3,ds4]
for i in range(len(ds)):
    ax = fig.add_subplot(len(ds),1,i+1)
    x = data.index[-len(ds[i]):]
    plt.plot(x, data, color='darkgrey', linestyle='--')
    ax.plot(x, ds[i])
    ax.set_ylabel('$O_3$ (ppbv)')
    y = np.array(ds[i])
    xx = np.arange(len(x))
    idx=np.isfinite(y)
    
    from scipy.stats.stats import pearsonr
    z = np.polyfit(xx[idx],y[idx],1)
    p = np.poly1d(z)
    pcc, xxxx = pearsonr(xx[idx],y[idx])
    
    ax.set_title(labels[i]+': '+str(np.round(z[0]*1e3,2))+' ppt $yr^{-1}$')
    ax.plot(x,p(xx),'r--')
    plt.xlim(data.index[0], data.index[-1])
    
    from scipy.stats import linregress
    xx = linregress(xx[idx],y[idx])
    
plt.tight_layout()
plt.savefig(savepath+'/residual_of_means.png')
plt.close()