#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:45:58 2020

@author: mjr583
"""
import numpy as np
import datetime
import netCDF4
import matplotlib.pyplot as plt
def timestamp_to_date(times):

    new_date=[]
    for t, tt in enumerate(times):
        x = (datetime.datetime(1900,1,1,0,0) + datetime.timedelta(tt-1))
        new_date.append(x)
    return new_date
url = 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc'
dataset = netCDF4.Dataset(url)

time = dataset.variables['time'][:]
new_date=np.array(timestamp_to_date(time))
mean = dataset.variables['ozone_ug_per_m3_amean'][:]

plt.plot(new_date,mean)
plt.show()