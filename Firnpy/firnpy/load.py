"""
This code is for loading data.
Re-coded after a previous file lost.

Created on: 06/12/2022

@author: katse
"""

def CS2(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    time_str = []
    for t in time:
        year = int(t)
        days_delta = int((t*365) % 365)
        new_time = (datetime.datetime(year, 1, 1) + datetime.timedelta(days = days_delta)).strftime("%Y%m%d")
        
        time_str.append(new_time)
    
    start_index = [i for i,t in enumerate(time_str) if t.startswith(start)][0]
    end_index = [i for i,t in enumerate(time_str) if t.startswith(end)][0]
    
    time_str = time_str[start_index:end_index+1]
    
    x = np.array(dfile.variables['x'][:]) #lon
    y = np.array(dfile.variables['y'][:]) #lat
    var = np.array(dfile.variables['dh'][start_index:end_index+1])
    err = np.array(dfile.variables['std'][start_index:end_index+1])
    projection = dfile.variables['polar_stereographic'].spatial_ref
    
    dfile.close()

    xgrid = var[0].copy()
    for i,q in enumerate(y):
        xgrid[i, :] = x
    
    ygrid = var[0].copy()
    for i,q in enumerate(x):
        ygrid[:, i] = y
    
    return xgrid, ygrid, time_str, var, err, projection

def GRACE(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    g_start = datetime.datetime(1858,11,17,00,00,00) 
    
    g_time = []
    g_datetime = []
    g_months = []
    for t in time:
        
        delta = datetime.timedelta(days=int(t))
        date = g_start + delta
        ds = date.strftime("%Y%m%d")
        dm = date.strftime("%Y%m")
        
        g_time.append(ds)
        g_datetime.append(date)
        g_months.append(dm)
    
    g_months = np.unique(np.array(g_months))
    g_date = [f+str(15) for f in g_months]
    
    start_index = [i for i,t in enumerate(g_date) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(g_date) if str(t).startswith(end)][0]

    
    errortrend = np.array(dfile.variables['sigma_dmdt'][:])
    trend = np.array(dfile.variables['dmdt'][:])
    area = np.array(dfile.variables['area'][:]) 
    regions = np.array(dfile.variables['regions'][:])
    error = np.array(dfile.variables['sigma_dm'][:])
    data = np.array(dfile.variables['dm'][:])

    berror = []
    bdata = []
    for i,f in enumerate(regions):
        e = error[i][start_index:end_index+1]
        d = data[i][start_index:end_index+1]

        berror.append(e)
        bdata.append(d)
    
    dfile.close()
    
    g_date = g_date[start_index:end_index]
    
    return g_date, bdata, berror, trend, errortrend, area, regions



def FDM_zs(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]

    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables['zs_monthly'][start_index:end_index+1])
    projection = dfile.variables['zs_monthly'].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection


def FDM_rho(start, end, path, var_name):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables[var_name][start_index:end_index+1])
    projection = dfile.variables[var_name].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    var[var==0] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection


def FDM_vfc(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables['compaction_monthly'][start_index:end_index+1])
    projection = dfile.variables['compaction_monthly'].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection



def SMB(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables['compaction_monthly'][start_index:end_index+1]) #this is smb data but saved it with wrong variable name
    projection = dfile.variables['compaction_monthly'].projection #this is smb data but saved it with wrong variable name
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection


def RUNOFF(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables['runoff'][start_index:end_index+1])
    projection = dfile.variables['runoff'].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection


def RUNOFFheight(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables['runoff_height'][start_index:end_index+1])
    projection = dfile.variables['runoff_height'].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection


def SNOW(start, end, variable, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables[variable][start_index:end_index+1])
    projection = dfile.variables[variable].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection



def iVEL(start, end, path):
    
    from netCDF4 import Dataset
    import datetime
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    time = np.array(dfile.variables['time'][:])
    
    start_index = [i for i,t in enumerate(time) if str(t).startswith(start)][0]
    end_index = [i for i,t in enumerate(time) if str(t).startswith(end)][0]
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables['compaction_monthly'][start_index:end_index+1])
    projection = dfile.variables['compaction_monthly'].projection
    
    dfile.close()
    
    var[var==9.969209968386869e+36] = np.nan
    time = time[start_index:end_index+1]
    time = np.array([str(t) for t in time])
    
    return x, y, time, var, projection

    
    
    return



def fit(path, grid):
    
    from netCDF4 import Dataset
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables[grid][:])
    unit = dfile.variables[grid].units
    projection = dfile.variables[grid].projection
    error = np.array(dfile.variables['std'][:])
    
    dfile.close()
        
    return x, y, var, error, unit, projection

    
def mask(path, grid):
    
    from netCDF4 import Dataset
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var = np.array(dfile.variables[grid][:])
    
    dfile.close()
        
    return x, y, var


def error(path, error_grids, error_series):

    from netCDF4 import Dataset
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    var1 = np.array(dfile.variables[error_grids][:])
    unit1 = dfile.variables[error_grids].units
    var2 = np.array(dfile.variables[error_series][:])
    unit2 = dfile.variables[error_series].units

    projection = dfile.variables[error_grids].projection
    
    dfile.close()
        
    return x, y, var1, unit1, var2, unit2, projection


def grid_series(path, grid):
    
    from netCDF4 import Dataset
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    
    x = np.array(dfile.variables['lon'][:]) #x
    y = np.array(dfile.variables['lat'][:]) #y
    time = np.array(dfile.variables['time'][:])
    var = np.array(dfile.variables[grid][:])
    unit = dfile.variables[grid].units
    projection = dfile.variables[grid].projection

    dfile.close()
        
    return x, y, time, var, unit, projection
