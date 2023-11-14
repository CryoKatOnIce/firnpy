""" This code is for GRACE data plotting.

Created on: 09/02/2023

@author: katse
"""

"""Libraries"""
from firnpy import load, save, analise, plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import calendar

"""Paths"""
gpath = '/Users/kat/DATA/Firn_data/IMBIE_GRIS_timeseries_BertWouters.txt'

outpath = '/Users/kat/DATA/Firn_data/'

ts = '201101'
te = '201712'
em = 31 #end of month of end date

"""Load data and adjust time"""

gdata = pd.read_csv(gpath, sep=",", header=None)
gdata.columns = ["author", "source", "method", "area", "id1", "id2", "time", "cumulative_mass", "2std"]


dtime = [datetime(int(x), 1, 1) + timedelta(days = (x % 1) * (366 if calendar.isleap(int(x)) else 365)) for x in gdata['time']]

gtime = [t.date() for t in dtime]

df = pd.DataFrame({'time': gtime})

gdata['time'] = df['time']

mdata_z = gdata.loc[gdata['method'] == 'Zwally']
mdata_r = gdata.loc[gdata['method'] == 'Rignot']

mdata_z.set_index(mdata_z['time'], drop=True, inplace=True)
mdata_r.set_index(mdata_r['time'], drop=True, inplace=True)


"""Extract time period, ploting and yearly rate"""

mdata_z = mdata_z.truncate(before = datetime(int(ts[:4]),int(ts[4:]),1).date()).truncate(after = datetime(int(te[:4]),int(te[4:]),em).date())
mdata_r = mdata_r.truncate(before = datetime(int(ts[:4]),int(ts[4:]),1).date()).truncate(after = datetime(int(te[:4]),int(te[4:]),em).date())

refmass_z = [m - mdata_z['cumulative_mass'].iloc[0] for m in mdata_z['cumulative_mass']]
refmass_r = [m - mdata_r['cumulative_mass'].iloc[0] for m in mdata_r['cumulative_mass']]

mdata_z['ref_mass'] = refmass_z
mdata_r['ref_mass'] = refmass_r

mdata_z['std'] = mdata_z['2std'].div(2)
mdata_r['std'] = mdata_r['2std'].div(2)

plot.grace_series_output(mdata_r, mdata_z, 'GRACE_Rignot', 'GRACE_Zwally', ts, te, 'Total mass change for GIS from Gravimetry', 'Gt', 'grace_series_check', outpath)

plot.grace_series(mdata_z,'GRACE', ts, te, 'Mass change', 'Gt', 'GRACE', outpath)


