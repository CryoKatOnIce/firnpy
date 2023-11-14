""" This code is for analysing data.
Re-coded after a previous file lost.

Created on: 08/12/2022

@author: katse
"""


def select_icesheet(data, icesheet_grid):
    
    import numpy as np
    
    for g in data:
        for i,v in np.ndenumerate(g):
            if np.isnan(icesheet_grid[i]): #== True:
                g[i] = np.nan

    return data


def select_icesheet_onegrid(data, icesheet_grid):
    
    import numpy as np
    
    for i,v in np.ndenumerate(data):
        if np.isnan(icesheet_grid[i]): #== True:
            data[i] = np.nan

    return data


def reference(data):
    
    import numpy as np
    
    data_ref = data.copy()
    for j,g in enumerate(data):
        for i,v in np.ndenumerate(g):
            
            data_ref[j][i] = g[i] - data[0][i]
    
    return data_ref


def referenced_cumulative(data, refdata):
    
    import numpy as np
    
    reference = np.nanmean(refdata, axis = 0)
    
    referenced_data = data - reference
    
    data_cumulative = np.cumsum(referenced_data, axis= 0)
    
    return data_cumulative


def remove_seasonality(data):
    
    import numpy as np
    from scipy.signal import butter,filtfilt
    
    data_noseason = data.copy()

    for i,v in np.ndenumerate(data[0]):
        
        timeseries = data[:, i[0], i[1]]
        
        T = len(timeseries)/12       #s  
        fs = 12.0       #Hz
        order = 2       
        n = int(T * fs) 
        cutoff = 0.5     #Hz
        nyq = 0.5 * fs  #Hz
        normal_cutoff = cutoff / nyq    #normalized between 0 and 1
    
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, timeseries)

        data_noseason[:, i[0], i[1]] = y
    
    return data_noseason


def remove_seasonality_timeseries(timeseries):
    
    import numpy as np
    from scipy.signal import butter,filtfilt
    
       
    T = len(timeseries)/12       #s  
    fs = 12.0       #Hz
    order = 2       
    n = int(T * fs) 
    cutoff = 0.5     #Hz
    nyq = 0.5 * fs  #Hz
    normal_cutoff = cutoff / nyq    #normalized between 0 and 1

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_series = filtfilt(b, a, timeseries)
    
    return filtered_series




def fit_grid(data):
    
    from scipy import stats
    import numpy as np
    
    data_fit = data[0].copy()
    data_fiterror = data[0].copy()
    for i,v in np.ndenumerate(data[0]):
        
        cell = data[:, i[0], i[1]]
    
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(range(len(cell)), cell)
        fit = slope*12
        fite = stderr*12
    
        data_fit[i[0], i[1]] = fit
        data_fiterror[i[0], i[1]] = fite
    
    return data_fit, data_fiterror


def fit_series(series):
    
    from scipy import stats
    import numpy as np
    
    
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(range(len(series)), series)
    series_fit = slope*12
    series_errorfit = stderr*12
    
    
    return series_fit, series_errorfit


def fit_grace(series, start, end):
    
    from scipy import stats
    import numpy as np
    
    years = int(end[:4]) - int(start[:4])
    months = ((int(end[4:]) - int(start[4:])) % 12 )/12
    steps = len(series)/(years + months)
    
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(range(len(series)), series)
    series_fit = slope*steps
    series_errorfit = stderr*steps
    
    
    return series_fit, series_errorfit



def fit_ebandgrid(data, edata):
    
    import numpy as np
    
    edata_fit = edata[0].copy()
    for i,v in np.ndenumerate(edata[0]):
        
        ecell = edata[:, i[0], i[1]]
        
        cell = data[:, i[0], i[1]]
        
        smax = (cell[0]+ecell[0]) - (cell[-1]-ecell[-1])
        smin = (cell[-1]+ecell[-1]) - (cell[0]-ecell[0])

        error = abs(smax-smin)/2

        fit = error/(len(cell)/12)   #error per year     
            
        edata_fit[i[0], i[1]] = fit
    
    return edata_fit

def detrend(series):
    
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = [i for i in range(0, len(series))]
    X = np.reshape(X, (len(X), 1))
    trend = LinearRegression().fit(X, series).predict(X)
    detrended = [series[i]-trend[i] for i,s in enumerate(series)]

    return detrended

def ablation_mask(data):
    
    import numpy as np
    
    ablation_only = data.copy()
    
    for i,v in np.ndenumerate(data):
        
        if v >= 0:
            ablation_only[i] = np.nan
        elif np.isnan(v):
            ablation_only[i] = np.nan
        else:
            ablation_only[i] = 1


    ablation_accumulation = data.copy()
    
    for i,v in np.ndenumerate(data):
        
        if v >= 0:
            ablation_accumulation[i] = 0
        elif np.isnan(v):
            ablation_accumulation[i] = np.nan            
        else:
            ablation_accumulation[i] = 1

    
    return ablation_only, ablation_accumulation


def accumulation_mask(data):
    
    import numpy as np
    
    accumulation_only = data.copy()
    
    for i,v in np.ndenumerate(data):
        
        if v < 0:
            accumulation_only[i] = np.nan
        elif np.isnan(v):
            accumulation_only[i] = np.nan
        else:
            accumulation_only[i] = 1


    accumulation_ablation = data.copy()
    
    for i,v in np.ndenumerate(data):
        
        if v < 0:
            accumulation_ablation[i] = 0
        elif np.isnan(v):
            accumulation_ablation[i] = np.nan            
        else:
            accumulation_ablation[i] = 1

    
    return accumulation_only, accumulation_ablation

def facies_mask(smb, melt, runoff):
    
    import numpy as np
    
    mask = smb.copy()
    
    for i,v in np.ndenumerate(smb):
        
        if v >= 0 and melt[i]<0 and runoff[i]<0: #dry
            mask[i] = 5
        elif v >= 0 and melt[i]>=0 and runoff[i]<0: #percolation
            mask[i] = 4
        elif v >= 0 and melt[i]>=0 and runoff[i]>=0: #wet
            mask[i] = 3
        elif v < 0 and melt[i]>=0 and runoff[i]>=0: #ablation
            mask[i] = 1 #leaving 2 for dynamic areas from Slater mask                       
        elif np.isnan(v): #not ice sheet
            mask[i] = np.nan
        else:
            mask[i] = np.nan

    
    return mask


def annual_positives(time, data):
    #more that 50% of annual avergaes are positive
    
    import numpy as np
    import pandas as pd
    
    years = [s[:4] for s in time]
    mask = data[0].copy()
    for i,v in np.ndenumerate(data[0]):
        
        cell = data[:, i[0], i[1]] 
        
        #get average yearly 
        df = pd.DataFrame({'year': years, 'cell': cell})
        df2 = (df.groupby(['year']).mean('cell'))
        
        m = [1 for c in df2.cell if c>0] 
        
        if np.nansum(m)>=len(df2.cell)/2: #more than 50% of the vlues are positive
            mask[i] = 1
        elif np.isnan(v): #not ice sheet
            mask[i] = np.nan
        else:
            mask[i] = -1
    
    return mask


def positives(time, data):
    #more that 50% of annual avergaes are positive
    
    import numpy as np

    mask = data[0].copy()
    for i,v in np.ndenumerate(data[0]):
        
        cell = data[:, i[0], i[1]] 
                
        m = [1 for c in cell if c>0] 
        
        if np.nansum(m)>=len(cell)/2: #more than 50% of the vlues are positive
            mask[i] = 1                        
        elif np.isnan(v): #not ice sheet
            mask[i] = np.nan
        else:
            mask[i] = -1
    
    return mask



def function_Hurkmans(grid, correlation_distance): #applied to one grid

    import numpy as np    

    grid_resolution = 1.5 #[km]
    cell_area = grid_resolution**2 #[km2]
            
    icesheet_area = np.sum(~np.isnan(grid)) * cell_area #[km2]        
            
    #CORRELATED CS2 MEASUREMENT+INTERPOLATION ERROR
    std_regions = int(icesheet_area/(np.pi*correlation_distance**2)) #number of correlation regions    
       
    #from Hurkmans:
    std_mean_correlated_error = (np.sqrt(np.nansum((grid)**2)*cell_area)/np.sqrt(std_regions*cell_area)) / np.sqrt(7) # additionally to Hurkmans we divide by np.sqrt(7) because we smooth the data (remove the seasonality)

    error = np.round(std_mean_correlated_error, 4)    

    return error


def error_CS2(stdgrid, correlation_distance): #correlation_distance = 3 #[km]
        
    
    error_timeseries = [] #timeseries of the error the length of time period
    for std_grid in stdgrid:
        
        error = function_Hurkmans(std_grid, correlation_distance)
        
        error_timeseries.append(error)
    
    
    error_grids = stdgrid.copy()
        
    
    return error_grids, error_timeseries 



def error_FDM(periodgfdm, refperiodfdm):

    from scipy import stats
    import numpy as np

    
    shift = 12
    chunks = [refperiodfdm[i + shift:i + shift + (12*20)] for i in range(0, len(refperiodfdm), shift)]
    
    periods = []
    for p in chunks:
        if len(p) == 240:
            periods.append(p)
            
    periods = np.array(periods)
    
        
    "Error grids"
    
    error_grids = periodgfdm.copy()
    for i,v in np.ndenumerate(periodgfdm[0]):
    
        cslopes = []
        for p in periods:
            
            trend = p[:, i[0], i[1]]
            
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(range(len(trend)), trend)
            
            slope_trend = np.array([slope*range(0, len(periodgfdm))])
        
            cslopes.append(slope_trend)


        error_series = np.nanstd(cslopes, axis=0)
    
        error_grids[:, i[0], i[1]] = error_series

    "Timeseries error"
    
    error_band = [np.nanmean(e) for e in error_grids]
    
    return error_grids, error_band

def station(grids, gridx, gridy, coords, areasize): #where area size is expressed in km as a side of a square area
    
    import numpy as np
    
    ix = (np.abs(gridx - coords[0])).argmin()
    iy = int(((np.abs(gridy - coords[1])).argmin())/gridx.shape[1])
    
    cell_size = 1.5
    expand = round(((areasize/cell_size) - (cell_size/2))/2)
    
    stationarea = grids[:, iy-expand:iy+expand, ix-expand:ix+expand]
    
    trend = np.array([np.nanmean(sg) for sg in stationarea])
    
    return trend

    
def statione_CS(grids, gridx, gridy, coords, areasize, correlation_distance): #where area size is expressed in km as a side of a square area
    
    import numpy as np
    
    ix = (np.abs(gridx - coords[0])).argmin()
    iy = int(((np.abs(gridy - coords[1])).argmin())/gridx.shape[1])
    
    cell_size = 1.5
    expand = round(((areasize/cell_size) - (cell_size/2))/2)
    
    stationarea = grids[:, iy-expand:iy+expand, ix-expand:ix+expand]
    
    trend = np.array([function_Hurkmans(sg, correlation_distance) for sg in stationarea])
    
    return trend
    

def basin(grids, gridx, gridy, bgrid, basin): #where area size is expressed in km as a side of a square area
    
    import numpy as np
    
    area = []
    for g in grids:
        
        bg = g[bgrid==basin]
        
        area.append(bg)
    
    trend = np.array([np.nanmean(sg) for sg in area])
    
    return trend


def basine_CS(grids, gridx, gridy, bgrid, basin, correlation_distance): #where area size is expressed in km as a side of a square area
    
    import numpy as np
    
    area = []
    for g in grids:
        
        bg = g[bgrid==basin]
        
        area.append(bg)
    
    trend = np.array([function_Hurkmans(sg, correlation_distance) for sg in area])
    
    return trend

def basin_n1(path):
    
    from netCDF4 import Dataset
    import numpy as np

    dfile = Dataset(path, "r", format="NETCDF4")

    bx = np.array(dfile.variables['lon'][:]) #x
    by = np.array(dfile.variables['lat'][:]) #y
    basins = np.array(dfile.variables['Basins'][:])

    dfile.close()

    #basins[basins==0] = np.nan

    n1_mask = basins.copy()

    for i,b in np.ndenumerate(basins):
        
        if b == 5:
            n1_mask[i] = 1
        elif b == 6:
            n1_mask[i] = 1
        else:
            n1_mask[i] = 0
            
    
    return n1_mask, bx, by

def basins_gracedresden(path):

    from netCDF4 import Dataset
    import numpy as np
    
    dfile = Dataset(path, "r", format="NETCDF4")
    
    bx = np.array(dfile.variables['lon'][:]) #x
    by = np.array(dfile.variables['lat'][:]) #y
    basins = np.array(dfile.variables['Basins'][:])
    
    dfile.close()
    
    #basins[basins==0] = np.nan
    
    grace_mask = basins.copy()
    
    for i,b in np.ndenumerate(basins):
    
        if 0 < b < 5:
            grace_mask[i] = 1
        elif  4 < b < 7:
            grace_mask[i] = 2
        elif  6 < b < 11:
            grace_mask[i] = 3
        elif  10 < b < 13:
            grace_mask[i] = 4
        elif  12 < b < 15:
            grace_mask[i] = 5
        elif  14 < b < 17:
            grace_mask[i] = 6
        elif  16 < b < 19:
            grace_mask[i] = 7
        elif  18 < b < 23:
            grace_mask[i] = 8
        else:
            grace_mask[i] = 0
    
    return grace_mask, bx, by

def basin_subgrid(basin_mask, facies_mask, topomask, csgridx, csgridy, mass_grids, elev_grids, snow_grids, basin, x1, x2, y1, y2, fx1, fx2, fy1, fy2):
    
    import numpy as np
    
    no_mask = basin_mask.copy()
    no_mask[no_mask!=basin] = np.nan
    no_mask[no_mask==basin] = 1
    nogridx = csgridx[x1:x2, y1:y2]
    nogridy = csgridy[x1:x2, y1:y2]

    f_mask = facies_mask.copy()
    f_mask[f_mask==0] = np.nan
    f_mask = f_mask[fx1:fx2, fy1:fy2]
    fgridx = csgridx[fx1:fx2, fy1:fy2]
    fgridy = csgridy[fx1:fx2, fy1:fy2]

    t_mask = topomask.copy()
    t_mask = t_mask[x1:x2, y1:y2]

    #AO
    no_ao = mass_grids[0] * no_mask
    no_ao[no_ao==0] = np.nan
    no_ao = no_ao[x1:x2, y1:y2]


    #Kappelsberger
    no_k = mass_grids[1] * no_mask
    no_k[no_k==0] = np.nan
    no_k = no_k[x1:x2, y1:y2]


    #McMillan
    no_mcm = mass_grids[2] * no_mask
    no_mcm[no_mcm==0] = np.nan
    no_mcm = no_mcm[x1:x2, y1:y2]


    #IMAU
    no_imau = mass_grids[3] * no_mask
    no_imau[no_imau==0] = np.nan
    no_imau = no_imau[x1:x2, y1:y2]

    no_massgrids = [no_ao, no_k, no_mcm, no_imau]
    
    #Elevation grids
    no_cs = elev_grids[0]*no_mask
    no_cs = no_cs[x1:x2, y1:y2]

    no_fdm = elev_grids[1]*no_mask
    no_fdm = no_fdm[x1:x2, y1:y2]

    no_smb = elev_grids[2]*no_mask
    no_smb = no_smb[x1:x2, y1:y2]

    no_comp = elev_grids[3]*no_mask
    no_comp = no_comp[x1:x2, y1:y2]

    no_elevgrids = [no_cs, no_fdm, no_smb, no_comp]
    
    no_m = snow_grids[0]*no_mask
    no_m = no_m[x1:x2, y1:y2]

    no_sf = snow_grids[1]*no_mask
    no_sf = no_sf[x1:x2, y1:y2]

    no_snowgrids = [no_m, no_sf]
    
    return nogridx, nogridy, no_massgrids, no_elevgrids, no_snowgrids, fgridx, fgridy, f_mask, t_mask