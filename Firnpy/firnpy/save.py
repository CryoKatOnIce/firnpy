"""
This code is for loading data.
Re-coded after a previous file lost.

Created on: 06/12/2022

@author: katse
"""

def data(data, x, y, time, variable_name, unit, description, file_name, path):
    
    import numpy as np
    from netCDF4 import Dataset
          
    data_file = Dataset(path + '/' + file_name, 'w', format='NETCDF4_CLASSIC')

    lat_dim = data_file.createDimension('lat', data.shape[1])     # latitude axis
    lon_dim = data_file.createDimension('lon', data.shape[2])    # longitude axis
    time_dim = data_file.createDimension('time', data.shape[0])


    var = data_file.createVariable(variable_name, np.float64, ('time', 'lat', 'lon',))
    tim = data_file.createVariable('time', np.float64, ('time',))
    lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
    lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

    var.description = variable_name + ' for the study time period ' +str(time[0]) + ' to ' + str(time[-1]) + ' ' + description + '.'
    var.units = unit
    var.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var.projection = 'epsg:3413'

    var[:] = data
    tim[:] = time
    lat[:] = y
    lon[:] = x

    data_file.close()

    
def fit(data, error, x, y, variable_name, unit, description, file_name, path):
    
    import numpy as np
    from netCDF4 import Dataset
          
    data_file = Dataset(path + '/' + file_name, 'w', format='NETCDF4_CLASSIC')

    lat_dim = data_file.createDimension('lat', data.shape[0])     # latitude axis
    lon_dim = data_file.createDimension('lon', data.shape[1])    # longitude axis

    var = data_file.createVariable(variable_name, np.float64, ('lat', 'lon',))
    err = data_file.createVariable('std', np.float64, ('lat', 'lon',))
    lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
    lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

    var.description = variable_name + ' linear fit per year in the study time period ' + description + '.'
    var.units = unit
    var.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var.projection = 'epsg:3413'

    var[:] = data
    err[:] = error
    lat[:] = y
    lon[:] = x

    data_file.close()


def mask(data1, data2, x, y, variable_name1, variable_name2, file_name, path):
    
    import numpy as np
    from netCDF4 import Dataset
          
    data_file = Dataset(path + '/' + file_name, 'w', format='NETCDF4_CLASSIC')

    lat_dim = data_file.createDimension('lat', data1.shape[0])     # latitude axis
    lon_dim = data_file.createDimension('lon', data1.shape[1])    # longitude axis

    var1 = data_file.createVariable(variable_name1, np.float64, ('lat', 'lon',))
    var2 = data_file.createVariable(variable_name2, np.float64, ('lat', 'lon',))
    lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
    lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

    var1.description = variable_name1 + 'mask based on mean SMB for the time period from RACMO.'
    var1.units = '1 for ' + variable_name1
    var1.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var1.projection = 'epsg:3413'
    
    var2.description = variable_name2 + ' mask based on mean SMB for the time period from RACMO.'
    var2.units = '1 and 0 for ' + variable_name2 + ' respectevily.'
    var2.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var2.projection = 'epsg:3413'


    var1[:] = data1
    var2[:] = data2
    lat[:] = y
    lon[:] = x

    data_file.close()

def facies_mask(data1, x, y, variable_name1, file_name, path):
    
    import numpy as np
    from netCDF4 import Dataset
          
    data_file = Dataset(path + '/' + file_name, 'w', format='NETCDF4_CLASSIC')

    lat_dim = data_file.createDimension('lat', data1.shape[0])     # latitude axis
    lon_dim = data_file.createDimension('lon', data1.shape[1])    # longitude axis

    var1 = data_file.createVariable(variable_name1, np.float64, ('lat', 'lon',))
    lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
    lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

    var1.description = variable_name1 + 'mask based on mean SMB, runoff and snowmelt for the time period from RACMO.'
    var1.units = '1 for ablation, 2 for dynamic, 3 for wet, 4 for percolation, 5 for dry'
    var1.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var1.projection = 'epsg:3413'
    

    var1[:] = data1
    lat[:] = y
    lon[:] = x

    data_file.close()

def basin_mask(data1, x, y, variable_name1, file_name, path):
    
    import numpy as np
    from netCDF4 import Dataset
          
    data_file = Dataset(path + '/' + file_name, 'w', format='NETCDF4_CLASSIC')

    lat_dim = data_file.createDimension('lat', data1.shape[0])     # latitude axis
    lon_dim = data_file.createDimension('lon', data1.shape[1])    # longitude axis

    var1 = data_file.createVariable(variable_name1, np.float64, ('lat', 'lon',))
    lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
    lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

    var1.description = variable_name1 + 'mask of a basin following basin N1 of Kappelsberger, i.e. basins 5 and 6 from Mouginiot.'
    var1.units = '1 for basin'
    var1.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var1.projection = 'epsg:3413'
    

    var1[:] = data1
    lat[:] = y
    lon[:] = x

    data_file.close()


def error(datags, datat, x, y, time, variable1_name, variable2_name, unit, description, file_name, path):
    
    import numpy as np
    from netCDF4 import Dataset
          
    data_file = Dataset(path + '/' + file_name, 'w', format='NETCDF4_CLASSIC')

    lat_dim = data_file.createDimension('lat', datags.shape[1])     # latitude axis
    lon_dim = data_file.createDimension('lon', datags.shape[2])    # longitude axis
    time_dim = data_file.createDimension('time', datags.shape[0])


    var1 = data_file.createVariable(variable1_name, np.float64, ('time', 'lat', 'lon',))
    var2 = data_file.createVariable(variable2_name, np.float64, ('time',))
    tim = data_file.createVariable('time', np.float64, ('time',))
    lat = data_file.createVariable('lat', np.float64, ('lat', 'lon',))
    lon = data_file.createVariable('lon', np.float64, ('lat', 'lon',))

    var1.description = variable1_name + ' for the study time period ' +str(time[0]) + ' to ' + str(time[-1]) + ' ' + description + '.'
    var1.units = unit
    var1.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var1.projection = 'epsg:3413'
    
    var2.description = variable2_name + ' is a timeseries in the study time period ' + description + '.'
    var2.units = unit
    var2.source = 'data processed by KM Sejan for use in altimetry to mass conversion'
    var2.projection = 'epsg:3413'


    var1[:] = datags
    var2[:] = datat
    tim[:] = time
    lat[:] = y
    lon[:] = x

    data_file.close()

def rates_txt(data, period_start, period_end, name, path): # rate_data, ts, te, 'GIS_rates.txt', figpath)

    f = open(path + name, 'w')
    f.write('Study period: ' + period_start + '-' + period_end)
    f.write('\n')

    f.write('rho_i = ' + str(data[0]) + 'kg/m3')
    f.write('\n')
    f.write('rho_w = ' + str(data[1]) + 'kg/m3')
    f.write('\n')
    f.write('cell_area = ' + str(data[2]) + '**2 m2')
    f.write('\n')
    f.write('\n')

    f.write('Total mass rates for GIS: ')
    f.write('\n')
    f.write('Altimetry Only: ' + str(data[3]) + ' +/- ' + str(data[4]) + ' Gt/yr')
    f.write('\n')
    f.write('Kapplesberger: ' + str(data[5]) + ' +/- ' + str(data[6]) + ' Gt/yr')
    f.write('\n')
    f.write('IMAU: ' + str(data[7]) + ' +/- ' + str(data[8]) + ' Gt/yr')
    f.write('\n')
    f.write('McMillan: ' + str(data[9]) + ' +/- ' + str(data[10]) + ' Gt/yr')
    f.write('\n')
    f.write('\n')

    f.write('Other rates: ')
    f.write('\n')
    f.write('Average ice sheet CS2 rate: ' + str(data[11]) + ' +/- ' + str(data[12]) + ' m/yr')
    f.write('\n')
    f.write('Average ice sheet FDM rate: ' + str(data[13]) + ' +/- ' + str(data[14]) + ' m/yr')
    f.write('\n')
    f.write('Average ice sheet SMB rate: ' + str(data[15]) + ' +/- ' + str(data[16]) + ' m w.e./yr')
    f.write('\n')
    f.write('Average rho Kappelsberger: ' + str(data[17]) + ' +/- ' + str(data[18]) + ' kg/m3')
    f.write('\n')
    f.write('Average rho McMillan: ' + str(data[19]) + ' +/- ' + str(data[20]) + ' kg/m3')
    f.write('\n')
    f.write('Average compaction: ' + str(data[21]) + ' +/- ' + str(data[22]) + ' m/yr')
    f.write('\n')
    f.write('\n')

    f.write('Total SMB mass rate for GIS: ' + str(data[23]) + ' +/- ' + str(data[24]) + ' Gt/yr')
    f.write('\n')
    f.write('Total Compaction mass rate for GIS: ' + str(data[25]) + ' +/- ' + str(data[26]) + ' Gt/yr')
    f.write('\n')
    f.write('Total FDM mass rate for GIS: ' + str(data[27]) + ' +/- ' + str(data[28]) + ' Gt/yr')
    f.write('\n')
    f.write('\n')

    f.write('Masks: ')
    f.write('\n')
    f.write('GIS area: ' + str(data[29]) + ' km2')
    f.write('\n')
    f.write('Ablation area: ' + str(data[30]) + ' km2')
    f.write('\n')
    f.write('Accumulation area: ' + str(data[31]) + ' km2')
    f.write('\n')
    f.write('Dynamic area: ' + str(data[32]) + ' km2')
    f.write('\n')
    f.write('\n')

    f.write('Ablation %: ' + str(data[33]))
    f.write('\n')
    f.write('Accumulation %: ' + str(data[34]))
    f.write('\n')
    f.write('Dynamic %: ' + str(data[35]))
    f.write('\n')
    f.write('Percolation %: ' + str(data[36]))
    f.write('\n')
    f.write('Wet snow %: ' + str(data[37]))
    f.write('\n')
    f.write('Dry snow %: ' + str(data[38]))
    f.write('\n')
    f.write('Percolation % of accumulation: ' + str(data[39]))
    f.write('\n')
    f.write('Wet snow % of accumulation: ' + str(data[40]))
    f.write('\n')
    f.write('Dry snow % of accumulation: ' + str(data[41]))
    f.write('\n')


    f.close()

def basin_rates_txt(data, title,  period_start, period_end, name, path): # rate_data, ts, te, 'GIS_rates.txt', figpath)

    f = open(path + name, 'w')
    f.write('Study period: ' + period_start + '-' + period_end)
    f.write('\n')

    f.write(title)
    f.write('\n')
    
    f.write('Altimetry Only: ' + str(data[0]) + ' +/- ' + str(data[1]) + ' Gt/yr')
    f.write('\n')
    f.write('Kapplesberger: ' + str(data[2]) + ' +/- ' + str(data[3]) + ' Gt/yr')
    f.write('\n')
    f.write('McMillan: ' + str(data[4]) + ' +/- ' + str(data[5]) + ' Gt/yr')
    f.write('\n')
    f.write('IMAU: ' + str(data[6]) + ' +/- ' + str(data[7]) + ' Gt/yr')
    f.write('\n')

    f.close()
