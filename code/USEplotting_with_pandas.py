# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:54:04 2019
Script for reading 

@author: ee11mr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (14, 14)
##------------------------------------------------------------------------------##
def moving_average(a,n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
##------------------------------------------------------------------------------##
def cen_moving_average(a,n=3):
    ret = np.nancumsum(a, dtype=float)
    b = n/2
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
##------------------------ MAIN SCRIPT ----------------------------------------
filepath  = '/Users/ee11mr/Documents/York/'

filen = filepath+'20191007_CV_Merge.csv'
df = pd.read_csv(filen, index_col=0,dtype={'Airmass':str})
df.index = pd.to_datetime(df.index,format='%d/%m/%Y %H:%M')

filen = filepath+'cv_ovocs_2018_M_Rowlinson.csv'
odf = pd.read_csv(filen, index_col=0)
odf.index = pd.to_datetime(odf.index,format='%d/%m/%Y %H:%M')

cols = list(df) ; ocols = list(odf)
for col in cols:
    df[col] = df[col].loc[~(df[col] <= 0.)]
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


years = np.arange(2006, 2006+len(yearly.index))#['2010','2011','2012','2013','2014']
colors = ['#e5f5e0','#c7e9c0','R','#74c476','#41ab5d',\
              '#7fcdbb','#41b6c4','#1d91c0',\
              '#225ea8','b','#253494','#081d58','k']

metkeys = cols[17:28]             
amkey = cols[:14]
spkey = cols[28:44]

ctrlkey = 'O3'
JASdaily = daily[np.in1d(daily.index.month, [7,8,9])]

fig = plt.figure()
for k,key in enumerate(metkeys):
    ax = fig.add_subplot(3,4,k+1)
    for y,year in enumerate(yearly.index.year[1:]):#ly.index:
        year=str(year)
        ax.scatter(JASdaily[ctrlkey][year], JASdaily[key][year], label=year,color=colors[y])
    ax.set_ylabel(key[:25])
    ax.set_xlabel(ctrlkey)
    plt.tight_layout()
ax.legend()
plt.savefig(filepath+'plots/metkey_'+ctrlkey+'_2009.png')
plt.close()    

fig = plt.figure()
for k,key in enumerate(amkey):
    ax = fig.add_subplot(3,5,k+1)
    for y,year in enumerate(yearly.index.year[1:]):#ly.index:
        year=str(year)
        ax.scatter(JASdaily[ctrlkey][year], JASdaily[key][year], label=year,color=colors[y])
    ax.set_ylabel(key[:25])
    ax.set_xlabel(ctrlkey)
    plt.tight_layout()
ax.legend()
plt.savefig(filepath+'plots/amkey_'+ctrlkey+'_2009.png')
plt.close()

fig = plt.figure()
for k,key in enumerate(spkey):
    ax = fig.add_subplot(4,4,k+1)
    for y,year in enumerate(yearly.index.year[1:]):#ly.index:
        year=str(year)
        ax.scatter(JASdaily[ctrlkey][year], JASdaily[key][year], label=year,color=colors[y])
    ax.set_ylabel(key[:25])
    ax.set_xlabel(ctrlkey)
    plt.tight_layout()
ax.legend()
plt.savefig(filepath+'plots/spkey_'+ctrlkey+'_2009.png')
plt.close()


from scipy.stats.stats import pearsonr
plt.rcParams['figure.figsize'] = (9, 9)
key = 'Isoprene (pptV)'
key2='O3'

if 'V)' in key: ## just to make save file look nicer
    savkey=key[:-6]
else:
    savkey=key
if 'V)' in key2:
    savkey2=key2[:-6]
else:
    savkey2=key2
    
for y,year in enumerate(yearly.index.year[1:]): ## skips 2006 due to incomplete data
    year=str(year)     
    xs = JASdaily[key][year]
    ys = JASdaily[key2][year]
    
    idx = np.isfinite(xs) & np.isfinite(ys)
    if not any(idx) == True: ## skips year if no data at all
        pass
    else: ## calculates trend, pearson R and plots data
        ab = np.polyfit(xs[idx], ys[idx], 1)
        p = np.poly1d(ab)
        plt.plot(xs,p(xs),color=colors[y], linestyle='--')
        
        pcc,no = pearsonr(xs[idx],ys[idx])
        plt.scatter(xs, ys, label=year+': r='+str(np.round(pcc,2)),color=colors[y])
plt.xlabel(key)    
plt.ylabel(key2[:25])
plt.legend()
plt.savefig(filepath+'/plots/'+savkey+'_'+savkey2+'_JAS.png')
plt.close()