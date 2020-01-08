#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:45:58 2020
@author: mjr583
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,4)
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.offsetbox import AnchoredText
import datetime
import netCDF4
def timestamp_to_date(times):
    new_date=[]
    for t, tt in enumerate(times):
        x = (datetime.datetime(1900,1,1,0,0) + datetime.timedelta(tt-1))
        new_date.append(x)
    return new_date

##------------------Script-----------------------------------------------------
url = 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc'
dataset = netCDF4.Dataset(url)
species='ozone'
spec=species

time = dataset.variables['time'][:]
new_date=np.array(timestamp_to_date(time))
mean = dataset.variables['ozone_ug_per_m3_amean'][:]
df = pd.DataFrame(mean)
df.index = new_date
df.columns = [species]

df = df.fillna(method='bfill')
idx = np.isfinite(df)
Y = df[idx] 
X = np.arange(len(Y))

''' Guess of polynomial terms '''
z, p = np.polyfit(X, Y, 1)
a = np.nanmean(df.resample('A').mean())  #31.08
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
start_year='2006'
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

idx = np.isfinite(df)
Y = df[idx] 
X = np.arange(len(Y))
#pp = np.linspace(1,5.1,np.round(len(Y)*0.75),0) ; qq = np.linspace(5.1,0.9,np.round(len(Y)*0.25,0)) ; rr = np.concatenate((pp,qq))
#Y = Y * rr
''' Guess of polynomial terms '''
z, p = np.polyfit(X, Y, 1)

a = np.nanmean(df.resample('A').mean())  #31.08
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
years=np.arange(int(2006),2020)
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