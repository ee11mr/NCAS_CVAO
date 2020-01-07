# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:31:45 2019
Script to examine different deseasonalisation techniques. 
@author: ee11mr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (7, 7)
filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'

filen = filepath+'20191007_CV_Merge.csv'
df = pd.read_csv(filen, index_col=0,dtype={'Airmass':str},engine='python')
df.index = pd.to_datetime(df.index,format='%d/%m/%Y %H:%M')

filen = filepath+'cv_ovocs_2018_M_Rowlinson.csv'
odf = pd.read_csv(filen, index_col=0,engine='python')
odf.index = pd.to_datetime(odf.index,format='%d/%m/%Y %H:%M')

cols = list(df) ; ocols = list(odf)
for col in cols:
    try:
        df[col] = df[col].loc[~(df[col] <= 0.)]
    except:
        pass
for col in ocols:
    odf = odf.loc[~(odf[col] <= 0.)]
cols  = cols+ocols
   
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
style=['-','--','-.',':']
## Choose the species - one at a time currently
species = 'O3'
## chose period to plot, starts at 2006, runs to 2019
start_year = '2007'
if species == 'O3' or species == 'meoh' or 'ace' in species:
    unit = ''
elif species == 'CO' or species == 'CH4':
    unit = ' (ppbV)'
else:
    unit = ' (pptV)'
data = monthly[species+unit][start_year:]
years = len((data.resample('Y').mean()).index.year)

## method one - centered moving average
ds1 = data.rolling(window=12,center=True).mean()
## method two - seasonal differencing 
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
diff = data.diff()
ds2 = data[12:] + difference(data, 12)
## method three - subtracting monthly average for each observation
mean = data.mean()
std = data.std()
monmean = data.groupby(data.index.month).mean()
deseas_factor = monmean / std
ds3=np.zeros(len(data))
for n,m in enumerate(data.index.month):
    ds3[n] = data[n] + (mean - monmean[m])
## method 4 - divide series by seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
idx = np.isfinite(data)
result = seasonal_decompose(data[idx], model='multiplicative',freq=12)
ds4 = result.observed / result.seasonal
## method five - will fail when there is insifficient data, failsafe added

try:
    a = data / result.trend
    iters = int(np.ceil(years / (14./3.)))
    for iter in range(iters):
        idx= np.isfinite(a)
        result = seasonal_decompose(a[idx], model='multiplicative', freq=12)
        b = data / result.seasonal
        idx = np.isfinite(b)
        result = seasonal_decompose(b[idx], model='multiplicative', freq=12)
        a = data / result.trend
    ds5 =  b
    func=1
    if ds5.max() > data.max() or ds5.min() < data.min():
        func=0
except:
    func=0

## plotting comparison of each method
labels = ['Observations','Moving average','Seasonal differencing',\
          'Removing period seasonal mean','Normalised by seasonal decomposition',\
          'X-12 ARIMA method']
plt.rcParams['figure.figsize'] = (8, 8)
fig = plt.figure()
if func==1:
    ds = [data, ds1,ds2,ds3,ds4,ds5]
else:
    ds = [data, ds1,ds2,ds3,ds4]
for i in range(len(ds)):
    ax = fig.add_subplot(len(ds),1,i+1)
    x = data.index[-len(ds[i]):]
    plt.plot(data.index, data, color='darkgrey',linestyle='--')
    ax.plot(x, ds[i])
    ax.set_title(labels[i])
    plt.xlim(data.index[0], data.index[-1])
plt.tight_layout()
#plt.savefig(filepath+'plots/deseasonalisation_'+species+'.png')
plt.close()
## plotting results of all methods on one plot
plt.rcParams['figure.figsize'] = (12, 4)
plt.plot(data, label='Observations',color='darkgrey',linestyle='--')
plt.plot(ds1, label='Moving average', linestyle=style[0])
#plt.plot(ds1.index[-len(ds2):], ds2, label='Seasonal differencing',linestyle=style[0])
plt.plot(ds1.index[-len(ds3):], ds3, label='Residuals of mean',linestyle=style[2])
plt.plot(ds4, label='Normalised by seasonal decomposition', linestyle=style[3])
if func==1:
    plt.plot(ds5, label='X-12 ARIMA method', linestyle=style[0])
else:
    print('Insufficent data points for X-12 ARIMA deseasonalisation')
plt.ylabel(species+unit)
plt.legend()
plt.savefig(savepath+species+'_seasonality.png')
plt.close()