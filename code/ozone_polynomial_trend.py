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

cols = list(df) 
for col in cols:
    try:
        df[col] = df[col].loc[~(df[col] <= 0.)]
    except:
        pass

## in test case just use Ozone data for now
start_year='2007'
years = np.arange(2007, 2020)

data = df['O3'][start_year:]
data['2009-07-01' : '2009-09-30'] = np.nan
nyears = len((data.resample('Y').mean()).index.year)

## method three - subtracting monthly average for each observation
mean = data.mean()
std = data.std()
monmean = data.groupby(data.index.month).mean()
deseas_factor = monmean / std
ds3=np.zeros(len(data))
for n,m in enumerate(data.index.month):
    ds3[n] = data[n] + (mean - monmean[m])
ds3 = pd.DataFrame(ds3[:])
ds3.index = pd.to_datetime(df[start_year:].index,format='%d/%m/%Y')
Mean = ds3.resample('M').mean()[0]
Max = ds3.resample('M').max()[0]
Min = ds3.resample('M').min()[0]
per75 = ds3.resample('M').quantile(.75)[0]
per25 = ds3.resample('M').quantile(.25)[0]

labels = ['raw','mean','max','min','75th percentile','25th percentile']
plt.rcParams['figure.figsize'] = (12, 7)
fig = plt.figure()
data_ = data.resample('M').median()
ds = [data_, Mean,Max, Min, per75, per25]
for i in range(len(ds)):
    ax = fig.add_subplot(len(ds),1,i+1)
    x = data_.index[-len(ds[i]):]
    plt.plot(data_.index, data_, color='darkgrey',linestyle='--')
    ax.plot(x, ds[i])
    ax.set_title(labels[i])
    ax.set_ylabel('$O_3$ (ppbv)')
    
    from scipy.stats.stats import pearsonr
    y = np.array(ds[i])
    xx = np.arange(len(x))
    idx=np.isfinite(y)
    
    z = np.polyfit(xx[idx],y[idx],1)
    p = np.poly1d(z)
    pcc, xxxx = pearsonr(xx[idx],y[idx])
    
    ax.set_title(labels[i]+': '+str(np.round(z[0]*1e3,2))+' ppt $yr^{-1}$')
    ax.plot(x,p(xx),'r--')

    plt.xlim(data.index[0], data.index[-1])
    #plt.ylim(13,44)
plt.tight_layout()
plt.savefig(savepath+'/trends_ozone.png')
plt.close()

data = df['O3']['2007':'2018']
data['2009-07-01' : '2009-09-30'] = np.nan
nyears = len((data.resample('Y').mean()).index.year)
data_ = data.resample('M').mean()
interseas=[] ; i=0 ; j=12
for y in years[:-1]:
    Ma = np.nanmax(data_[i:j])
    Mi = np.nanmin(data_[i:j])
    interseas.append(Ma-Mi)
    i=i+12 ; j=j+12

from sklearn.linear_model import LinearRegression  
import operator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures 
plt.rcParams['figure.figsize'] = (13, 6)
for dn,d in enumerate(ds):
    idx=np.isfinite(ds[dn])
    y = np.array(ds[dn][idx])
    x = np.arange(len(ds[dn][idx].index))
    
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    
    model = LinearRegression()
    model.fit(x,y)
    z = np.round(model.coef_[0][0] * 12,2)
    #print(labels[dn], z)
    y_pred = model.predict(x)
    
    #plt.scatter(x,y, s=10)
    #plt.plot(x, y_pred, color='r')
    
    #plt.ylabel('Ozone (ppbv)')
    #years=np.arange(2007,2020)
    #plt.xticks(np.arange(0, len(x), 12), years)
    
    #plt.savefig(savepath+'ozone_trends/'+labels[dn][:4]+'_linear_trend.png')
    #plt.close()
    
   
    polynomial_features = PolynomialFeatures(degree=4)
    x_poly = polynomial_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly,y)
    y_poly_pred = model.predict(x_poly)
    
    rmse = np.round(np.sqrt(mean_squared_error(y,y_poly_pred)),2)
    r2 = np.round(r2_score(y,y_poly_pred),2)
    
    fig, ax = plt.subplots()
    ax.scatter(x,y, s=10)
    
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    ax.plot(x, y_poly_pred, color='m')
    ax.plot(x, y_pred, 'r--')
    
    ax.set_ylabel('Monthly '+labels[dn]+' $O_3$ (ppbv)')
    years=np.arange(2007,2020)
    plt.xticks(np.arange(0, len(x), 12), years)
    
    plt.text(0.7,0.04, '$r^2$='+str(r2)+'% \nRMSE='+str(rmse)+\
             '% \nLinear trend='+str(z)+' ppbv $yr^{-1}$', fontsize=14,transform=ax.transAxes, \
             bbox=dict(facecolor='none', edgecolor='lightgrey'))#, boxstyle='round,pad=1'))

    plt.savefig(savepath+'ozone_trends/'+labels[dn][:4]+'_poly_trend.png')
    plt.close()
    
hold = [] ; hold2=[]
for i in range(2,20):
    polynomial_features = PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly,y)
    y_poly_pred = model.predict(x_poly)
    
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)
    #print(i, ':', rmse)
    hold.append(rmse)
    hold2.append(r2)
## highest r2 value when degree=15
## lowest RMSE wehn degree=15