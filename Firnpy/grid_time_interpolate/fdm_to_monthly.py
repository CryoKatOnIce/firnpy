
"""
This code is for changing the 10-daily resolution of IMAU-FDM products to a monthly average.
Re-coded after a previous file lost.

Created on: 05/12/2022

@author: katse
"""


"""Libraries"""


from netCDF4 import Dataset
import datetime
import numpy as np


"""Load file"""
#path = '/Users/kat/DATA/FIRNPY_DATA/FDM_dens_32cm_FGRN055_1957-2020_GrIS_GIC.nc'
path = '/Users/kat/DATA/FIRNPY_DATA/FDM_Rho0_FGRN055_1957-2020_GrIS_GIC.nc'

dfile = Dataset(path, "r", format="NETCDF4")
x = np.array(dfile.variables['lon'][:])
y = np.array(dfile.variables['lat'][:])
var = np.array(dfile.variables['Rho0'][:]) # dens_32cm #zs #dens_1m #vfc
#time = np.array(dfile.variables['time'][:]) #days since 1957-10-11 00:00:00
#projection = dfile.variables['rotated_pole'].proj_parameters

dfile.close()

path = '/Users/kat/DATA/FIRNPY_DATA/FDM_vfc_FGRN055_1957-2020_GrIS_GIC.nc'

dfile = Dataset(path, "r", format="NETCDF4")
time = np.array(dfile.variables['time'][:]) #days since 1957-10-11 00:00:00
projection = dfile.variables['rotated_pole'].proj_parameters

dfile.close()


"""Converting to polar stereographic"""

from pyproj import Transformer

inProj = projection
outProj = 'epsg:3413'
polar_transformer = Transformer.from_crs(inProj, outProj)

px = x.copy()
py = y.copy()

for i,v in enumerate(x):
    x_trans, y_trans  = polar_transformer.transform(x[i],y[i])
    px[i] = x_trans
    py[i] = y_trans

"""Creating time stamps"""

start_FDM = datetime.datetime(1957,10,11,00,00,00) #1957-10-11 00:00:00 

fdm_time = []
fdm_datetime = []
fdm_months = []
for t in time:
    
    delta = datetime.timedelta(days=int(t))
    date = start_FDM + delta
    ds = date.strftime("%Y%m%d")
    dm = date.strftime("%Y%m")
    
    fdm_time.append(ds)
    fdm_datetime.append(date)
    fdm_months.append(dm)

fdm_months = np.unique(np.array(fdm_months))
fdm_date = [f+str(15) for f in fdm_months]

"""Function for converting to average"""

def to_monthly(fdm_datetime, fdm_months, var):
    
    import numpy as np
    import datetime
    
    fdm_grids = []
    fdm_nangrids = []
    for d in fdm_months:
        
        year = int(d[:4])
        month = int(d[4:])
        
        if month<12:
            length_month = (datetime.date(year, month+1, 1) - datetime.date(year, month, 1)).days
        else:
            length_month = (datetime.date(year+1, 1, 1) - datetime.date(year, month, 1)).days
        
        group_dates = []
        group_indices = []
        for i,t in enumerate(fdm_datetime):
            
            if t.year == year and t.month == month:
                group_dates.append(t)
                group_indices.append(i)
        
        month_days = [d.day for d in group_dates]
        extra_days = length_month - month_days[-1]
        group_weights = list([month_days[0]]) + list(np.diff(month_days)) + list([extra_days])
        
        
        if d == fdm_months[-1]:
            if extra_days >0:
                group_indices.append(group_indices[-1])

            month_grid = np.nansum([np.array(var[n])*group_weights[i] for i,n in enumerate(group_indices)], axis=0)/length_month
            month_nangrid = month_grid.copy()
            month_nangrid[month_nangrid==month_nangrid[0,0]]=np.nan

        else:
            if extra_days >0:
                group_indices.append(group_indices[-1]+1)

            month_grid = np.nansum([np.array(var[n])*group_weights[i] for i,n in enumerate(group_indices)], axis=0)/length_month
            month_nangrid = month_grid.copy()
            month_nangrid[month_nangrid==month_nangrid[0,0]]=np.nan

        fdm_grids.append(month_grid)
        fdm_nangrids.append(month_nangrid)
        
    fdm_grids = np.array(fdm_grids)
    fdm_nangrids = np.array(fdm_nangrids)
        
    return fdm_grids, fdm_nangrids

"""Apply function"""

fdm_grids, fdm_nangrids = to_monthly(fdm_datetime, fdm_months, var)

"""Save data to new file"""

path_to_outfile = '/Users/kat/DATA/FIRNPY_DATA/FDM_dens_surface_monthlyK_FGRN055_1957-2020_GrIS_GIC.nc'

import numpy as np
from netCDF4 import Dataset
      
data_file = Dataset(path_to_outfile, 'w', format='NETCDF4_CLASSIC')

lat_dim = data_file.createDimension('lat', x.shape[0])     # latitude axis
lon_dim = data_file.createDimension('lon', x.shape[1])    # longitude axis
time_dim = data_file.createDimension('time', len(fdm_date))


var_monthly = data_file.createVariable('denssurface_monthly', np.float64, ('time', 'lat', 'lon',)) #zs_monthly #dens1m_monthly #compaction_monthly
time = data_file.createVariable('time', np.float64, ('time',))
lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

var_monthly.description = 'dnesity averaged over top 32 cm'
var_monthly.units = 'kg/m3'
var_monthly.source = 'Original 10-day IMAU FDM data averaged to one month data by KM Sejan'
var_monthly.projection = 'epsg:3413'

var_monthly[:] = fdm_grids
time[:] = fdm_date
lat[:] = py
lon[:] = px

data_file.close()
