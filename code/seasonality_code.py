# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:45:28 2019
Script for examinign seasonality of OVOCs species
Data good from ~2015 - 2018
@author: ee11mr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-darkgrid')
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
species = ['acetone','meoh','acetaldehyde','O3','CO','CH4 all','CO2']
## chose period to plot, starts at 2006, runs to 2019
start_year = '2015'
for spec in species:
    if spec == 'O3':
        suff = '' ; unit='(ppbV)'
    elif spec == 'meoh' or 'ace' in spec:
        suff = '' ; unit='(pptV)'
    elif spec == 'CO' or 'CH4' in spec:
        suff = ' (ppbV)' ; unit=suff
    elif 'CO2' in spec:
        suff = ' (ppmV)' ; unit=suff
    else:
        suff = ' (pptV)' ; unit=suff
    data = monthly[spec+suff][start_year:]
    years = (data.resample('Y').mean()).index.year
    
    mean = data.mean()
    std = data.std()
    monmean = data.groupby(data.index.month).mean()
    
    plt.rcParams['figure.figsize'] = (10, 6)
    x = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    #x = data['2015'].index.month_name()
    for year in years:
        if len(data[str(year)]) != 12:
            pass
        else:
            annmon= data[str(year)]
            plt.plot(x, annmon, label=year)
    plt.plot(x, monmean, color='darkgrey', linestyle='--', label='Period mean')
    plt.ylabel(spec+' '+unit)
    plt.legend()
    plt.savefig(savepath+spec+'_seasonality.png')
    plt.show()
    plt.close()