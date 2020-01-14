#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:56:33 2020

@author: mjr583
"""

CVAO_dict = {
     'O3' : {   'ceda_url' : 'http://dap.ceda.ac.uk/thredds/dodsC/badc/capeverde/data/cv-tei-o3/2019/ncas-tei-49i-1_cvao_20190101_o3-concentration_v1.nc',
                'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc',
                'species' : 'O3',
                'longname' : 'ozone',
                'abbr' : '$O_3$',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'merge_suff' : '',
                'var_name' : 'ozone_nmol_per_mol_amean'
                },
    'CO' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20160101000000.20190425083930.online_crds.carbon_monoxide.air.3y.1h.GB12L_CVO_Picarro_G2401.GB12L_Picarro.lev2.nc',
                'species' : 'CO',
                'longname' : 'carbon_monoxide',
                'abbr' : 'CO',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'merge_suff' : ' (ppbV)',
                'var_name' : 'carbon_monoxide_amean'
                },
    'NO' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20170101000000.20191024143042.chemiluminescence_photolytic..air.2y.1h.GB12L_CVO_AQD_Nox.GB12L_AQD_NOx.lev2.nc',
                'species' : 'NO',
                'longname' : 'nitrogen_monoxide',
                'abbr' : 'NO',
                'unit': 'ptbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'merge_suff' : ' (pptV)',
                'var_name' : 'nitrogen_monoxide_nmol_per_mol'
                },
    'NO2' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20170101000000.20191024143042.chemiluminescence_photolytic..air.2y.1h.GB12L_CVO_AQD_Nox.GB12L_AQD_NOx.lev2.nc',
                'species' : 'NO2',
                'longname' : 'nitrogen_dioxide',
                'abbr' : '$NO_2$',
                'unit': 'pptv',
                'scale' : '1e9',
                'start_year' : '2006',
                'merge_suff' : ' (pptV)',
                'var_name' : 'nitrogen_dioxide_nmol_per_mol'
                },
    'propane' : {'species' : 'propane',
                'longname' : 'propane',
                'abbr' : '$NO_2$',
                'unit': 'pptv',
                'scale' : '1e12',
                'start_year' : '2006',
                'merge_suff' : ' (pptV)',
                'var_name' : 'nitrogen_dioxide_nmol_per_mol'
                },
    }
'''
    'NOx' : {'url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20170101000000.20191024143042.chemiluminescence_photolytic..air.2y.1h.GB12L_CVO_AQD_Nox.GB12L_AQD_NOx.lev2.nc',
                'species' : 'NOx',
                'longname' : 'nitrogen_oxides',
                'abbr' : '$NO_x$',
                'unit': 'ppbv',
                'scale' : '1e9',
                'start_year' : '2006',
                'var_name_a' : 'nitrogen_monoxide_nmol_per_mol',
                'var_name_b' : 'nitrogen_dioxide_nmol_per_mol'
                },
     }
'''