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
data = df['O3']#['2015':]
freq = 12 
#data = monthly['O3']['2015':]
years = len((data.resample('Y').mean()).index.year)

## method one - centered moving average
ds1 = data.rolling(window=freq,center=True).mean()

## method two - seasonal differencing 
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
diff = data.diff()
ds2 = data[freq:] + difference(data, freq)

## method three - subtracting monthly average for each observation
mean = data.mean()
std = data.std()
monmean = data.groupby(data.index.month).mean()
deseas_factor = monmean / std
ds3=np.zeros(len(data))
for n,m in enumerate(data.index.month):
    ds3[n] = data[n] + (mean - monmean[m])

## method 4 - divide series by seasonal decomposition
#fig = plt.figure()
from statsmodels.tsa.seasonal import seasonal_decompose
idx = np.isfinite(data)
result = seasonal_decompose(data[idx], model='multiplicative',freq=freq)

hold= [result.trend,result.seasonal,result.resid,result.observed, result.observed / result.seasonal]
ds4 = result.observed / result.seasonal

## method five - 
a = data / result.trend
'''
idx = np.isfinite(a)
result = seasonal_decompose(a[idx], model='multiplicative', freq=12)
b = data / result.seasonal

idx = np.isfinite(b)
result = seasonal_decompose(b[idx], model='multiplicative', freq=12)
a = data / result.trend
idx = np.isfinite(a)
result = seasonal_decompose(a[idx], model='multiplicative', freq=12)
b = data / result.seasonal

idx = np.isfinite(b)
result = seasonal_decompose(b[idx], model='multiplicative', freq=12)
a = data / result.trend
idx = np.isfinite(a)
result = seasonal_decompose(a[idx], model='multiplicative', freq=12)
b = data / result.seasonal
'''
iters = int(np.ceil(years / (14./3.)))
for iter in range(iters):
    idx= np.isfinite(a)
    result = seasonal_decompose(a[idx], model='multiplicative', freq=freq)
    b = data / result.seasonal
    idx = np.isfinite(b)
    result = seasonal_decompose(b[idx], model='multiplicative', freq=freq)
    a = data / result.trend
ds5 =  b

labels = ['Observations','Moving average','Seasonal differencing',\
          'Residuals of mean (monthly bins)','Normalised by seasonal decomposition',\
          'X-12 ARIMA method']
plt.rcParams['figure.figsize'] = (8, 8)
fig = plt.figure()
ds = [data, ds1,ds2,ds3,ds4,ds5]
for i in range(len(ds)):
    ax = fig.add_subplot(len(ds),1,i+1)
    x = data.index[-len(ds[i]):]
    plt.plot(data.index, data, color='darkgrey',linestyle='--')
    ax.plot(x, ds[i])
    ax.set_title(labels[i])
    ax.set_ylabel('$O_3$ (ppbv)')

    plt.xlim(data.index[0], data.index[-1])
    #plt.ylim(13,44)
plt.tight_layout()
plt.savefig(savepath+'/deseasonalisation_techniques.png')
plt.close()

plt.rcParams['figure.figsize'] = (12, 4)
plt.plot(data, label='Observations',color='darkgrey',linestyle='--')
plt.plot(ds1, label='Moving average')
#plt.plot(ds1.index[-len(ds2):], ds2, label='Seasonal differencing')
plt.plot(ds1.index[-len(ds3):], ds3, label='Residuals of mean')
plt.plot(ds4, label='Normalised by seasonal decomposition', linestyle='-.')
plt.plot(ds5, label='X-12 ARIMA method', linestyle='--')
plt.legend()
plt.ylabel('$O_3$ (ppbv)')
plt.savefig(savepath+'/deseasonalisation_comparison.png')
plt.close()

from scipy.stats.stats import pearsonr
data = data
#data['2009-07-01' : '2009-09-30'] = np.nan
y = np.array(data)
xx = np.arange(len(y))
idx=np.isfinite(y)


z = np.polyfit(xx[idx],y[idx],1)
p = np.poly1d(z)
pcc, xxxx = pearsonr(xx[idx],y[idx])

print(str(np.round(z[0]*1e3,2)))