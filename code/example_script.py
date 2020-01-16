#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:54:57 2020

@author: mjr583
"""
import CVAO_tools as CV
from CVAO_dict import CVAO_dict as d

''' EXAMPLE DICTIONARY '''
'''
d = {
    'SO2' : {'species' : 'SO2',
                     'ebas_url': '',
                     'ceda_url' : '',
                     'ebas_var_name' : '',
                     'ceda_var_name' : '',
                     'longname' : 'Sulphur dioxide',
                     'abbr' : '$SO_2$',
                     'unit': 'ppbv',
                     'scale' : '1e12',
                     'yscale' : 'linear',
                     'start_year' : '2006',
                     'merge_pref' : '',
                     'merge_suff' : ' (ppbV)',
                     'instrument' : ''
                }
    }
'''
CV.plot_trend_with_func_from_dict(d, force_merge=False, timestep='H')
CV.plot_trend_with_func_from_dict(d, force_merge=False, timestep='M')