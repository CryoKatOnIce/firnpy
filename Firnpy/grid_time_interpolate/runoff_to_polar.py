"""
@author: katse
"""


"""Libraries"""


from netCDF4 import Dataset
import datetime
import numpy as np


"""Load file"""
#path = '/Users/kat/DATA/FIRNPY_DATA/runoff.FGRN055_1957-2020_BN_RACMO2.3p2_Monthly.nc'
path = '/Users/kat/DATA/FIRNPY_DATA/snowfall.FGRN055_1957-2020_BN_RACMO2.3p2_Monthly.nc'


dfile = Dataset(path, "r", format="NETCDF4")
x = np.array(dfile.variables['lon'][:])
y = np.array(dfile.variables['lat'][:])
#height = np.array(dfile.variables['height'][:])
var = np.array(dfile.variables['snowfall'][:]) # mm w.e. per month #runoff
ftime = np.array(dfile.variables['time'][:]) # hours since 1957-09-15 12:00:00 #days since 1950-01-01
projection = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5'

dfile.close()


"""Creating time stamps"""

start_SMB = datetime.datetime(1957,9,15,12,00,00) #1957-09-15 12:00:00 

smb_time = []
smb_datetime = []
smb_months = []
for t in ftime:
    
    delta = datetime.timedelta(hours=int(t)) #days
    date = start_SMB + delta
    ds = date.strftime("%Y%m%d")
    dm = date.strftime("%Y%m")
    
    smb_time.append(ds)
    smb_datetime.append(date)
    smb_months.append(dm)

smb_months = np.unique(np.array(smb_months))
smb_date = [f+str(15) for f in smb_months]


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


"""Save data to new file"""

path_to_outfile = '/Users/kat/DATA/FIRNPY_DATA/snowfall_monthly_polar_FGRN055_1957-2020_BN_RACMO23p2.nc'

import numpy as np
from netCDF4 import Dataset
      
data_file = Dataset(path_to_outfile, 'w', format='NETCDF4_CLASSIC')

lat_dim = data_file.createDimension('lat', x.shape[0])     # latitude axis
lon_dim = data_file.createDimension('lon', x.shape[1])    # longitude axis
time_dim = data_file.createDimension('time', len(ftime))


var_monthly = data_file.createVariable('snowfall', np.float64, ('time', 'lat', 'lon',))
time = data_file.createVariable('time', np.float64, ('time',))
lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

var_monthly.description = 'Snowfall from RACMO'
var_monthly.units = 'mm w.e. per month'
var_monthly.source = 'Original snowfall from RACMO, coordinates converted to polar by KM Sejan'
var_monthly.projection = 'epsg:3413'

var_monthly[:] = var
time[:] = smb_date
lat[:] = py
lon[:] = px

data_file.close()

"""
path_to_outfile = '/Users/kat/DATA/FIRNPY_DATA/runoffheight_monthly_polar_FGRN055_1957-2020_BN_RACMO23p2.nc'

import numpy as np
from netCDF4 import Dataset
      
data_file = Dataset(path_to_outfile, 'w', format='NETCDF4_CLASSIC')

lat_dim = data_file.createDimension('lat', x.shape[0])     # latitude axis
lon_dim = data_file.createDimension('lon', x.shape[1])    # longitude axis
time_dim = data_file.createDimension('time', len(ftime))


var_monthly = data_file.createVariable('runoff_height', np.float64, ('time', 'lat', 'lon',))
time = data_file.createVariable('time', np.float64, ('time',))
lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

var_monthly.description = 'Runoff height from RACMO'
var_monthly.units = 'meters'
var_monthly.source = 'Original runoff height from RACMO, coordinates converted to polar by KM Sejan'
var_monthly.projection = 'epsg:3413'

var_monthly[:] = height
time[:] = smb_date
lat[:] = py
lon[:] = px

data_file.close()
"""