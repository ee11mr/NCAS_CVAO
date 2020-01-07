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
from sklearn.metrics import mean_squared_error
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
#df['2009'] = np.nan 
start_year='2007'
df = df[start_year:].resample('M').mean() 
#df[spec] = df[spec].fillna(df[spec].mean())
df[spec] = df[spec].fillna(method='bfill')
#pp = np.linspace(1,1.1,40) ; qq = np.linspace(1.1,0.9,8) ; rr = np.concatenate((pp,qq))
no = len(df[spec])
idx = np.isfinite(df[spec])
Y = df[spec][idx] #* rr
X = np.arange(len(Y))
nn = len(Y)

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
plt.show()
plt.close()

''' With curvie fitting to minimise error '''
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
plt.show()
plt.close()


detrended = [Y[i] - c[1]*i for i in range(0, len(X))]
detrended2 = np.array([detrended[i] - c[2]*i**2 for i in range(0, len(X))])

plt.plot(X,Y, label='Obervations')
plt.plot(X,detrended,label='Detrend b')
plt.plot(X,detrended2,label='Detrend b + c')
plt.xticks(np.arange(0, len(X), 12), years)
plt.legend()
plt.show()
plt.close()

ds = pd.DataFrame(detrended2[:])
ds.index = df[idx].index
seas = ds.groupby(ds.index.month).mean()


i=0 ; j=12 ; irange=[]
for k in range(int(start_year),2018):
    plt.plot(X[:12], detrended2[i:j])
    irange.append(detrended2[i:j].max() - detrended2[i:j].min())
    i = i+12
    j = j+12
plt.plot(X[:12], seas, 'k--')
plt.show()


'''
N = 600
N = len(Y)
# sample spacing
T = 1/12#1.0 / 80.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
from scipy import fftpack
yf = fftpack.fft(detrended)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()
'''