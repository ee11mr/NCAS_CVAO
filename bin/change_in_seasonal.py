# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:33:36 2019

@author: ee11mr
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from datetime import datetime, timedelta
from scipy.interpolate import InterpolatedUnivariateSpline 
from pandas import  DataFrame
from scipy import stats
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (11, 7)
##------------------- Get time and dates for plotting -------------------------
def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta
##------------------------------------------------------------------------------##
def moving_average(a,n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
##------------------------ MAIN SCRIPT ----------------------------------------
## columns are as follows:
#0=ozone, 1=CO, 2=ch4, 3=ethane, 4=propane, 5=, 6=, 7=, 
#
species = ['O3','CO']
spec = 'CO' ; r=1

for r,spec in enumerate(species[:1]):
    print(spec)
    time=[]
    start_date = datetime(2006, 10, 2, 16, 00)
    end_date = datetime(2019, 8, 27, 21, 00)
    for single_date in daterange(start_date, end_date):
        time.append(single_date)
    filepath  = '/users/mjr583/scratch/cape_verde/'
    filen = filepath+'test_merge.txt'
    
    data = np.loadtxt(filen,skiprows=1)
    blank_fltr = np.where(data == -999.99)
    data[blank_fltr] = np.nan
    zero_fltr = np.where(data == 0.0)
    data[zero_fltr] = np.nan
    
    n=len(data)
    d = data[:,r][2168:107360]
    time=time[2168:107360]
    
    msk = np.where(np.isfinite(d))
    df = d[msk]

    plt.plot(time[:n],d)
    plt.savefig(filepath+'plots/'+spec+'_timeseries.png')
    plt.close()

    years = ['2007','2008','2009','2010','2011','2012','2013','2014','2015',\
                 '2016','2017','2018']
    months=['01','02','03','04','05','06','07','08','09','10','11','12']
    all_monthly = np.zeros(((len(months)),len(d))) ; all_monthly[:]=np.nan
    for m,mon in enumerate(months):
        for n,i in enumerate(time):
            j = str(i)
            if mon in j:
                all_monthly[m,n]=d[n]
            else:
                pass
            
    ann_monthly = np.zeros((len(years),len(months),len(d))) ; ann_monthly[:]=np.nan
    for y, year in enumerate(years):
        for m,mon in enumerate(months):
            for n,i in enumerate(time):
                j = str(i)
                if year in j and mon in j:
                    ann_monthly[y,m,n]=d[n]
                else:
                    pass
    
    all_seas = np.nanmean(all_monthly, axis=1)
    ann_seas = np.nanmean(ann_monthly, axis=2)
    
    month_names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct',\
                     'Nov','Dec']        
    colors = ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d',\
                  '#7fcdbb','#41b6c4','#1d91c0',\
                  '#225ea8','#253494','#081d58']
    
    for i in range(len(ann_monthly)):
        plt.plot(month_names,ann_seas[i], label=years[i], color=colors[i])      
    plt.plot(month_names,all_seas,label='2007-2018 mean',color='k',linewidth=3)
    plt.legend(ncol=2)
    plt.savefig(filepath+'plots/'+spec+'_seasonal.png')
    plt.close()
    
    all_seas = np.nanmean(all_monthly, axis=1)
    one_seas = np.nanmean(np.nanmean(ann_monthly[:3], axis=2),0)
    two_seas = np.nanmean(np.nanmean(ann_monthly[3:6], axis=2),0)
    thr_seas = np.nanmean(np.nanmean(ann_monthly[6:9], axis=2),0)
    fou_seas = np.nanmean(np.nanmean(ann_monthly[9:], axis=2),0)
    
    #colors=['#fee5d9','#fcae91','#fb6a4a','#cb181d']
    periods=['2007-2009','2010-2012','2013-2015','2016-2018']
    colors=['yellow','orange','red','darkred']
    plt.plot(month_names,one_seas, label=periods[0], color=colors[0])      
    plt.plot(month_names,two_seas, label=periods[1], color=colors[1])      
    plt.plot(month_names,thr_seas, label=periods[2], color=colors[2])      
    plt.plot(month_names,fou_seas, label=periods[3], color=colors[3])      
    
    plt.plot(month_names,all_seas,label='2007-2018 mean',color='k',linewidth=3)
    plt.legend()
    
    plt.savefig(filepath+'plots/'+spec+'_period_seasonal.png')
    plt.close()
    
