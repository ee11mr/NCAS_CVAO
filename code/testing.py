#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:54:35 2019

@author: mjr583
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy

filepath  = '/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/'
savepath  = '/users/mjr583/scratch/NCAS_CVAO/plots/'

filen = filepath+'20191007_CV_Merge.csv'
df = pd.read_csv(filen, index_col=0,dtype={'Airmass':str})
df.index = pd.to_datetime(df.index,format='%d/%m/%Y %H:%M')

cols = list(df) 
for col in cols:
    try:
        df[col] = df[col].loc[~(df[col] <= 0.)]
    except:
        pass

spec = 'O3'
df['2009-07-01' : '2009-09-30'] = np.nan 
df = df.resample('D').mean()

idx = np.isfinite(df[spec])
Y = df[spec][idx] 
X = np.arange(len(Y))

#seasonal_decompose(Y, model='additive', freq=24).plot()
result = seasonal_decompose(Y, model='additive', freq=12)
stop
#plt.show() ; plt.close()

detrended = result.observed / result.trend
plt.plot(detrended)
plt.show()

z,p = np.polyfit(X,Y,1)
print(z)

ran = np.arange(len(Y))
detrend = ran * z

plt.plot( Y - detrend )

# Number of samplepoints
N = 600
N = len(Y )
# sample spacing
T = 1/12#1.0 / 80.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(detrended)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()