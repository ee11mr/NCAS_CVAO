#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:54:57 2020

@author: mjr583
"""
import CVAO_tools as CV
from CVAO_dict import CVAO_dict as d

CV.plot_trend_with_func_from_dict(d, force_merge=False, timestep='M')