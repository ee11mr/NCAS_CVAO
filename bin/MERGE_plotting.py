# -*- coding: utf-8 -*-
"""
Created by Matthew Rowlinson on Tue Oct  8 11:12:36 2019

Python script to plot all data from MERGE.txt file. 

Currently for simple plots but can be adapted for deseasonalising, 
plotting diurnals, trends etc. 
@author: Matthew Rowlinson (matthew.rowlinson@york.ac.uk)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

##------------------- Get time and dates for plotting -------------------------
def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta

time=[]
start_date = datetime(2006, 10, 2, 16, 00)
end_date = datetime(2019, 8, 27, 21, 00)
for single_date in daterange(start_date, end_date):
    time.append(single_date)

##------------------------ MAIN SCRIPT ----------------------------------------
filepath  = '/Users/ee11mr/Documents/York/'
filen = filepath+'MERGE.txt'

print(filen)
#converters = {0: lambda s: float(s.strip('"')})
#xx = np.loadtxt(filen,skiprows=1,converter=converters)
xx = np.genfromtxt(filen,skip_header=1,usecols=np.arange(0,10))#,converter=converters)
print('done')
fltr = np.where(xx == -999.99)

xx[fltr] = np.nan
n=len(xx)
plt.plot(time[:n],xx[:,0])

