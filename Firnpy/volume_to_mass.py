""" This code is for altimetry to mass conversion.
Re-coded after a previous file lost.

Created on: 02/01/2023

@author: katse
"""

"""Libraries"""
from firnpy import load, save, analise, plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import calendar
from sklearn.metrics import mean_squared_error
from scipy import signal

"""Paths"""
ts = '201101'
te = '201712'
em = 31 #end of month of end date for GRACE data

period = 'Jan 2011 \n - Dec 2017' #this is for plotting

figpath = '/Users/kat/DATA/Firn_plotsandresults_2011_2017/'


cs2_path = '/Users/kat/DATA/Firn_data/cs_dhdt_gridfit_' + ts + '_' + te + '.nc'
cs2series_path = '/Users/kat/DATA/Firn_data/cs_dh_noseason_' + ts + '_' + te + '.nc'
cs2e_path = '/Users/kat/DATA/Firn_data/cs_error_correlated_at3km_' + ts + '_' + te + '.nc'

fdm_path = '/Users/kat/DATA/Firn_data/fdm_dhdt_gridfit_' + ts + '_' + te + '.nc'
fdmseries_path = '/Users/kat/DATA/Firn_data/fdm_dh_noseason_' + ts + '_' + te + '.nc'
fdme_path = '/Users/kat/DATA/Firn_data/fdm_error_' + ts + '_' + te + '.nc'

smb_path = '/Users/kat/DATA/Firn_data/smb_dhdt_gridfit_' + ts + '_' + te + '.nc'
smbseries_path = '/Users/kat/DATA/Firn_data/smb_dh_noseason_' + ts + '_' + te + '.nc'

rho_path = '/Users/kat/DATA/Firn_data/rho_model_' + ts + '_' + te + '.nc'
rho2_path = '/Users/kat/DATA/Firn_data/rho32cm_model_' + ts + '_' + te + '.nc'
rhos_path = '/Users/kat/DATA/Firn_data/rhosurf_model_' + ts + '_' + te + '.nc'

compaction_path = '/Users/kat/DATA/Firn_data/compaction_model_' + ts + '_' + te + '.nc'
compactionseries_path = '/Users/kat/DATA/Firn_data/compaction_dh_noseason_' + ts + '_' + te + '.nc'

runoff_path = '/Users/kat/DATA/Firn_data/runoff_dhdt_gridfit_' + ts + '_' + te + '.nc'
runoffseries_path = '/Users/kat/DATA/Firn_data/runoff_dh_noseason_' + ts + '_' + te + '.nc'
snowmelt_path = '/Users/kat/DATA/Firn_data/snowmelt_dhdt_gridfit_' + ts + '_' + te + '.nc'
snowmeltseries_path = '/Users/kat/DATA/Firn_data/snowmelt_dh_noseason_' + ts + '_' + te + '.nc'

snowfall_path = '/Users/kat/DATA/Firn_data/snowfall_dhdt_gridfit_' + ts + '_' + te + '.nc'
snowfallseries_path = '/Users/kat/DATA/Firn_data/snowfall_dh_noseason_' + ts + '_' + te + '.nc'


accum_path = '/Users/kat/DATA/Firn_data/accumulation_mask_' + ts + '_' + te + '.nc'
abl_path = '/Users/kat/DATA/Firn_data/ablation_mask_' + ts + '_' + te + '.nc'
dynamic_path = '/Users/kat/DATA/Firn_data/Slater_dynamics_mask_onCS2grid.nc'

basin_path = '/Users/kat/DATA/Firn_data/mouginot_basins_mask.nc'
basingrace_path = '/Users/kat/DATA/Firn_data/gracebasin_mask.nc'

facies_path = '/Users/kat/DATA/Firn_data/facies_mask_201101_201712.nc'

topo_path = '/Users/kat/DATA/Firn_data/topography_mask.nc'

gpath = '/Users/kat/DATA/Firn_data/IMBIE_GRIS_timeseries_BertWouters.txt'
grace_path = '/Users/kat/DATA/GRACE/GIS_GMB_basin.nc'

outpath = '/Users/kat/DATA/Firn_data/' #GRACE outpath


"""Load data"""
csgridx, csgridy, csfit, csfiterror, csunit, csprojection= load.fit(cs2_path, 'cs_dhdt')
csgsx, csgsy, csgstime, csgs, csgsunit, csgsprojection= load.grid_series(cs2series_path, 'cs_dh')
csegridx, csegridy, cse_grids, cseunit, cse_series, cssunit, cseprojection = load.error(cs2e_path, 'cserror_grids', 'cserror_series')

fdmgridx, fdmgridy, fdmfit, fdmfiterror, fdmunit, fdmprojection= load.fit(fdm_path, 'fdm_dhdt')
fdmgsx, fdmgsy, fdmgstime, fdmgs, fdmgsunit, fdmgsprojection= load.grid_series(fdmseries_path, 'fdm_dh')
fdmegridx, fdmegridy, fdme_grids, fdmeunit, fdme_series, fdmsunit, fdmeprojection = load.error(fdme_path, 'fdmerror_grids', 'fdmerror_series')

smbgridx, smbgridy, smbfit, smbfiterror, smbunit, smbprojection= load.fit(smb_path, 'smb_dhdt')
smbgsx, smbgsy, smbgstime, smbgs, smbgsunit, smbgsprojection= load.grid_series(smbseries_path, 'smb_dh')

rhogridx, rhogridy, rhogrid, rhogriderror, rhounit, rhoprojection= load.fit(rho_path, 'rho_model')
rho2gridx, rho2gridy, rho2grid, rho2griderror, rho2unit, rho2projection= load.fit(rho2_path, 'rho32cm_model')
rhosgridx, rhosgridy, rhosgrid, rhosgriderror, rhosunit, rhosprojection= load.fit(rhos_path, 'rhosurf_model')

compactiongridx, compactiongridy, compactiongrid, compactiongriderror, compactionunit, compactionprojection= load.fit(compaction_path, 'compaction_model')
compactiongsx, compactiongsy, compactiongstime, compactiongs, compactiongsunit, compactiongsprojection= load.grid_series(compactionseries_path, 'compaction_dh')

runoffgridx, runoffgridy, runoffgrid, runoffgriderror, runoffunit, runoffprojection= load.fit(runoff_path, 'runoff_dhdt')
runoffgsx, runoffgsy, runoffgstime, runoffgs, runoffgsunit, runoffgsprojection= load.grid_series(runoffseries_path, 'runoff_dh')

snowmeltgridx, snowmeltgridy, snowmeltgrid, snowmeltgriderror, snowmeltunit, snowmeltprojection= load.fit(snowmelt_path, 'snowmelt_dhdt')
snowmeltgsx, snowmeltgsy, snowmeltgstime, snowmeltgs, snowmeltgsunit, snowmeltgsprojection= load.grid_series(snowmeltseries_path, 'snowmelt_dh')

snowfallgridx, snowfallgridy, snowfallgrid, snowfallgriderror, snowfallunit, snowfallprojection= load.fit(snowfall_path, 'snowfall_dhdt')
snowfallgsx, snowfallgsy, snowfallgstime, snowfallgs, snowfallgsunit, snowfallgsprojection= load.grid_series(snowfallseries_path, 'snowfall_dh')


accumulationgridx, accumulationgridy, accumulation_mask = load.mask(accum_path, 'accumulation_only')
ablationgridx, ablationgridy, ablation_mask = load.mask(abl_path, 'ablation_only')
dynamicgridx, dynamicgridy, dynamic_mask = load.mask(dynamic_path, 'vdmask_final')

basingridx, basingridy, basin_mask = load.mask(basin_path, 'Mouginot_basins')
basingracegridx, basingracegridy, basingrace_mask = load.mask(basingrace_path, 'grace_mask')

fgridx, fgridy, facies_mask = load.mask(facies_path, 'facies_mask')
f_mask = facies_mask.copy()

grace_time, grace_data, grace_error, grace_trend, grace_errortrend, grace_area, grace_regions = load.GRACE('2011', '2018', grace_path)

topogridx, topogridy, topomask = load.mask(topo_path, 'Topography')

###############################################################################
"""METHODS"""
rho_i = 917 #[kg/m3]
rho_w = 997 #[kg/m3]

cell_area = 1500**2 #[m2]

"Altimetry only Method"
massgridkg_ao = csfit * rho_i #[kg/m2/yr] grid
massgrid_ao = (csfit * rho_i) * 10**-12 #[Gt/m2/yr] grid
gis_massrate_ao = np.round(np.nanmean(massgrid_ao) * np.sum(~np.isnan(massgrid_ao)) * cell_area, 2) #[Gt/yr] 

#error
masserrorkg_ao = (csfiterror * rho_i) #[kg/m2/yr] grid
masserror_ao = (csfiterror * rho_i) * 10**-12 #[Gt/m2/yr] grid
masserrorrate_ao = analise.function_Hurkmans(csfiterror, 3) * rho_i * 10**-12 #[Gt/m2/yr] rate per m2

gis_masserrrate_ao = np.round(masserrorrate_ao * np.sum(~np.isnan(masserror_ao)) * cell_area, 2) #[Gt/yr] GIS rate


#timeseries with error band
masstotalgrid_ao = csgs[-1] * rho_i #[kg/m2/yr] grid
massseries_ao = (csgs * rho_i) * 10**-12 #[Gt/m2/yr]
gis_massseries_ao = [np.nanmean(g) * np.sum(~np.isnan(g)) * cell_area for g in massseries_ao] #[Gt/yr]

masstotalerrorgrid_ao = cse_grids[-1] * rho_i #[kg/m2/yr] grid
masserrorseries_ao = (cse_grids * rho_i) * 10**-12 #[Gt/m2/yr]
masserrorband_ao = [analise.function_Hurkmans(g, 3) * rho_i * 10**-12 for g in cse_grids] #[Gt/m2/yr] rate per m2
gis_masserrband_ao = [g * np.sum(~np.isnan(cse_grids[0])) * cell_area for g in masserrorband_ao] #[Gt/yr] GIS rate



"Kappelsberger Method"

dynamic_mask[dynamic_mask==0] = np.nan

ice_mask = dynamic_mask.copy()
ice_mask[dynamic_mask==1] = rho_i #[kg/m2/yr]
ice_mask[ablation_mask==1] = rho_i #[kg/m2/yr]
ice_mask[csfit>1] = rho_i #[kg/m2/yr]

rhomodel = rhogrid.copy()
rhomodel[ice_mask==rho_i] = rho_i #[kg/m2/yr]

massgridkg_k = csfit * rhomodel #[kg/m2/yr] grid
massgrid_k = (csfit * rhomodel) * 10**-12 #[Gt/m2/yr] grid

gis_massrate_k = np.round(np.nanmean(massgrid_k) * np.sum(~np.isnan(massgrid_k)) * cell_area,2) #[Gt/yr] 

#error
masserrorkg_k = np.sqrt((csfiterror*rho_i)**2 + (csfit*rhogriderror)**2) #[kg/m2/yr] grid
masserror_k = np.sqrt((csfiterror*rho_i)**2 + (csfit*rhogriderror)**2) * 10**-12 #[Gt/m2/yr] grid
masserrorrate_k = np.sqrt((analise.function_Hurkmans(csfiterror, 3)*rho_i)**2 + (np.nanmean(csfit)*np.nanmean(rhogriderror))**2) * 10**-12 #[Gt/m2/yr] rate per m2

gis_masserrrate_k = np.round(masserrorrate_k * np.sum(~np.isnan(masserror_k)) * cell_area,2) #[Gt/yr] GIS rate


#timeseries with error band
masstotalgrid_k = csgs[-1] * rhomodel #[kg/m2/yr] grid
massseries_k = (csgs * rhomodel) * 10**-12 #[Gt/m2/yr]
gis_massseries_k = [np.nanmean(g) * np.sum(~np.isnan(g)) * cell_area for g in massseries_k] #[Gt/yr]

masstotalerrorgrid_k = np.sqrt((cse_grids[-1]*rho_i)**2 + (rhogriderror*csgs[-1])**2) #[kg/m2/yr] grid
masserrorband_k = [np.sqrt((analise.function_Hurkmans(g, 3)*rho_i)**2 + (np.nanmean(csgs[i])*np.nanmean(rhogriderror))**2) * 10**-12 for i,g in enumerate(cse_grids)] #[Gt/m2/yr] rate per m2
gis_masserrband_k = [g * np.sum(~np.isnan(cse_grids[0])) * cell_area for g in masserrorband_k] #[Gt/yr] GIS rate



"IMAU Method"

altimetry_corrected = (csfit - fdmfit) * rho_i #[kg/m2/yr]

smb_mass = smbfit * rho_w #[kg/m2/yr]

massgridkg_imau = altimetry_corrected + smb_mass #[kg/m2/yr] grid
massgrid_imau = (altimetry_corrected + smb_mass) * 10**-12 #[Gt/m2/yr] grid

gis_massrate_imau = np.round(np.nanmean(massgrid_imau) * np.sum(~np.isnan(massgrid_imau)) * cell_area, 2) #[Gt/yr] 

#error
masserrorkg_imau = np.sqrt(csfiterror**2 + fdmfiterror**2) * rho_i #[kg/m2/yr] grid
masserror_imau = np.sqrt(csfiterror**2 + fdmfiterror**2) * rho_i * 10**-12 #[Gt/m2/yr] grid
masserrorrate_imau = np.sqrt(analise.function_Hurkmans(csfiterror, 3)**2 + np.nanmean(fdmfiterror)**2) * rho_i * 10**-12 #[Gt/m2/yr] rate per m2

gis_masserrrate_imau = np.round(masserrorrate_imau * np.sum(~np.isnan(masserror_imau)) * cell_area, 2) #[Gt/yr] GIS rate


#timeseries with error band
masstotalgrid_imau = ((csgs[-1] - fdmgs[-1]) * rho_i) + (smbgs[-1] * rho_w) #[kg/m2/yr] grid
smb_massseries = smbgs * rho_w #[kg/m2/yr]
massseries_imau = (((csgs - fdmgs) * rho_i) + smb_massseries)  * 10**-12 #[Gt/m2/yr]

gis_massseries_imau = [np.nanmean(g) * np.sum(~np.isnan(g)) * cell_area for g in massseries_imau] #[Gt/yr]

masstotalerrorgrid_imau = np.sqrt(cse_grids[-1]**2 + fdme_grids[-1]**2) * rho_i #[kg/m2/yr] grid
masserrorband_imau = [np.sqrt(analise.function_Hurkmans(g, 3)**2 + np.nanmean(fdme_grids[i])**2) * rho_i * 10**-12 for i,g in enumerate(cse_grids)] #[Gt/m2/yr] rate per m2
gis_masserrband_imau = [g * np.sum(~np.isnan(cse_grids[0])) * cell_area for g in masserrorband_imau] #[Gt/yr] GIS rate



"McMillan Method"

rhomodel2 = rho2grid.copy()
rhomodel2[ice_mask==rho_i] = rho_i #[kg/m2/yr]

#altimetry minus compaction
compactiongrid = -compactiongrid
altimetry_compact = (csfit + compactiongrid) # we remove positive compaction rates form elevation loss, 
#i.e. we need to add the compaction rate to show elevation loss due to surface and dynamic processes only


#density model
#rho model is the same as per Kappelsberger method

#residual altimetry times density model
massgridkg_mcm = altimetry_compact * rhomodel2 #[kg/m2/yr] #rho model is the same as per Kappelsberger method
massgrid_mcm = ((csfit + compactiongrid) * rhomodel2) * 10**-12 #[Gt/m2/yr] #rho model is the same as per Kappelsberger method

gis_massrate_mcm = np.round(np.nanmean(massgrid_mcm) * np.sum(~np.isnan(massgrid_mcm)) * cell_area, 2) #[Gt/yr] 

#error
masserrorkg_mcm = np.sqrt((np.sqrt(csfiterror**2 + compactiongriderror**2) * rho2grid)**2 + (altimetry_compact*rho2griderror)**2) #[kg/m2/yr] grid
masserror_mcm = np.sqrt((np.sqrt(csfiterror**2 + compactiongriderror**2) * rho2grid)**2 + (altimetry_compact*rho2griderror)**2) * 10**-12 #[Gt/m2/yr] grid
masserrorrate_mcm = np.sqrt((np.sqrt(analise.function_Hurkmans(csfiterror, 3)**2 + np.nanmean(compactiongriderror)**2) * np.nanmean(rho2grid))**2 + np.nanmean(altimetry_compact*rho2griderror)**2) * 10**-12 #[Gt/m2/yr] rate per m2

gis_masserrrate_mcm = np.round(masserrorrate_mcm * np.sum(~np.isnan(masserror_mcm)) * cell_area, 2) #[Gt/yr] GIS rate


#timeseries with error band
compaction_grids = np.cumsum([-g/12 for g in compactiongs], axis= 0) #create cumulative no season referenced grid series to match CS2 grid series
masstotalgrid_mcm = (csgs[-1] + compaction_grids[-1]) * rhomodel2 #[kg/m2/yr] #rho model is the same as per Kappelsberger method
massseries_mcm = [((g + compaction_grids[i]) * rhomodel2) * 10**-12 for i,g in enumerate(csgs)]#[Gt/m2/yr]
gis_massseries_mcm = [np.nanmean(g) * np.sum(~np.isnan(g)) * cell_area for g in massseries_mcm] #[Gt/yr]

masstotalerrorgrid_mcm = np.sqrt((np.sqrt(cse_grids[-1]**2 + compactiongriderror**2) * rhomodel2)**2 + (rho2griderror*csgs[-1])**2) #[kg/m2/yr] #rho model is the same as per Kappelsberger method
masserrorband_mcm = [np.sqrt((np.sqrt(analise.function_Hurkmans(g, 3)**2 + np.nanmean(compactiongriderror)**2) * np.nanmean(rho2grid))**2 + np.nanmean(altimetry_compact*rho2griderror)**2) * 10**-12 for g in cse_grids] #[Gt/m2/yr] rate per m2
gis_masserrband_mcm = [g * np.sum(~np.isnan(cse_grids[0])) * cell_area for g in masserrorband_mcm] #[Gt/yr] GIS rate



"CS2 fit rate vs CS2 grids rate comparison"



"Other rates"
csrate = np.round(np.nanmean(csfit), 4) #[m/yr] 
cserrorrate = np.round(analise.function_Hurkmans(csfiterror, 3), 4) #[m/yr]

fdmrate = np.round(np.nanmean(fdmfit), 4) #[m/yr] 
fdmerrorrate = np.round(np.nanmean(fdmfiterror), 4) #[m/yr]

smbrate = np.round(np.nanmean(smbfit), 4) #[m/yr] 
smberrorrate = np.round(np.nanmean(smbfiterror), 4) #[m/yr]

average_rho = np.round(np.nanmean(rhomodel),2) #[kg/m3]
average_rhoerror = np.round(np.nanmean(rhogriderror), 2) #[kg/m3]

average_rho2 = np.round(np.nanmean(rhomodel2),2) #[kg/m3]
average_rho2error = np.round(np.nanmean(rho2griderror), 2) #[kg/m3]


average_compaction = np.round(np.nanmean(compactiongrid),4) #[m/year]
average_compactionerror = np.round(np.nanmean(compactiongriderror), 4) #[m/year] ???

smbmassrate = np.round(np.nanmean(smbfit*rho_w) * 10**-12 * np.sum(~np.isnan(smbfit)) * cell_area, 2) #[Gt/yr] 
smbmasserrorrate = np.round(np.nanmean(smbfiterror*rho_w) * 10**-12 * np.sum(~np.isnan(smbfiterror)) * cell_area, 2) #[Gt/yr]

fdmmassrate = np.round(np.nanmean(fdmfit*rho_i) * 10**-12 * np.sum(~np.isnan(fdmfit)) * cell_area, 2) #[Gt/yr] 
fdmmasserrorrate = np.round(np.nanmean(fdmfiterror*rho_i) * 10**-12 * np.sum(~np.isnan(fdmfiterror)) * cell_area, 2) #[Gt/yr]

compmassrate = np.round(np.nanmean(compactiongrid*rhomodel) * 10**-12 * np.sum(~np.isnan(compactiongrid)) * cell_area, 2) #[Gt/yr] 
compmasserrorrate = np.round(np.nanmean(compactiongriderror*rhomodel) * 10**-12 * np.sum(~np.isnan(compactiongriderror)) * cell_area, 2) #[Gt/yr]


"Mask area %"
ablation_mask[dynamic_mask==1] = np.nan
accumulation_mask[dynamic_mask==1] = np.nan

GIS_area = np.sum(~np.isnan(csfit)) * cell_area * 10**-6 #km2
Ablation_area = np.nansum(ablation_mask) * cell_area * 10**-6 #km2
Accumulation_area = np.nansum(accumulation_mask) * cell_area * 10**-6 #km2
Dynamic_area = np.nansum(dynamic_mask) * cell_area * 10**-6 #km2

ablation_p = np.round((np.nansum(ablation_mask)*100)/np.sum(~np.isnan(csfit)), 2)
accumulation_p = np.round((np.nansum(accumulation_mask)*100)/np.sum(~np.isnan(csfit)), 2)
dynamic_p = np.round((np.nansum(dynamic_mask)*100)/np.sum(~np.isnan(csfit)), 2)

#percolation
percolation_mask = f_mask.copy()
percolation_mask[percolation_mask != 4] = np.nan
percolation_mask[percolation_mask == 4] = 1

percolation_p = np.round((np.nansum(percolation_mask)*100)/np.sum(~np.isnan(csfit)), 2)

#wet snow
wetsnow_mask = f_mask.copy()
wetsnow_mask[wetsnow_mask != 3] = np.nan
wetsnow_mask[wetsnow_mask == 3] = 1

wetsnow_p = np.round((np.nansum(wetsnow_mask)*100)/np.sum(~np.isnan(csfit)), 2)

#dry snow
drysnow_mask = f_mask.copy()
drysnow_mask[drysnow_mask != 5] = np.nan
drysnow_mask[drysnow_mask == 5] = 1

drysnow_p = np.round((np.nansum(drysnow_mask)*100)/np.sum(~np.isnan(csfit)), 2)


percolation_ap = np.round((np.nansum(percolation_mask)*100)/np.sum(~np.isnan(accumulation_mask)), 2)
wetsnow_ap = np.round((np.nansum(wetsnow_mask)*100)/np.sum(~np.isnan(accumulation_mask)), 2)
drysnow_ap = np.round((np.nansum(drysnow_mask)*100)/np.sum(~np.isnan(accumulation_mask)), 2)


"Output file"
rate_data = [rho_i, rho_w, np.sqrt(cell_area), gis_massrate_ao, gis_masserrrate_ao, gis_massrate_k, 
              gis_masserrrate_k, gis_massrate_imau, gis_masserrrate_imau, gis_massrate_mcm, 
              gis_masserrrate_mcm, csrate, cserrorrate, fdmrate, fdmerrorrate, smbrate, 
              smberrorrate, average_rho, average_rhoerror, average_rho2, average_rho2error, average_compaction, average_compactionerror,
              smbmassrate, smbmasserrorrate, compmassrate, compmasserrorrate, fdmmassrate,
              fdmmasserrorrate, GIS_area, Ablation_area, Accumulation_area, Dynamic_area,
              ablation_p, accumulation_p, dynamic_p, percolation_p, wetsnow_p, drysnow_p, 
              percolation_ap, wetsnow_ap, drysnow_ap]

save.rates_txt(rate_data, ts, te, 'GIS_rates.txt', figpath)


"Plotting"
from datetime import datetime
series_time = [datetime. strptime(str(int(t)), "%Y%m%d") for t in csgstime]


plot.timeseries_mass4(series_time, gis_massseries_ao, gis_masserrband_ao, 'Altimetry only', gis_massseries_k, gis_masserrband_k, 'Kappelsberger', gis_massseries_mcm, gis_masserrband_mcm, 'McMillan', gis_massseries_imau, gis_masserrband_imau, 'IMAU', 'GIS Mass change using 4 conversion methods', '[Gt]', 'Total_mass_GIS', figpath)

#
plot.fitmap(csgridx, csgridy, compactiongrid, 'Compaction \n (m/yr)', period, 'bwr_r', -1, 0, 1, 'compaction_map_myr', figpath)


#
plot.fitmap(csgridx, csgridy, csfit, 'CryoSat-2 \n Elevation \n Change \n (m/yr)', period, 'bwr_r', -1, 0, 1, 'cs_map_myr', figpath)
plot.fitmap(csgridx, csgridy, csfiterror, 'CryoSat-2 \n Elevation \n Change error \n (m/yr)', period, 'Reds', 0, 0.05, 0.1, 'cserror_map_myr', figpath)

plot.fitmap(csgridx, csgridy, fdmfit, 'IMAU-FDM \n Elevation \n Change \n (m/yr)', period, 'bwr_r', -1, 0, 1, 'fdm_map_myr', figpath)
plot.fitmap(csgridx, csgridy, fdmfiterror, 'IMAU-FDM \n Elevation \n Change error \n (m/yr)', period, 'Reds', 0, 0.05, 0.1, 'fdmerror_map_myr', figpath)

plot.fitmap(csgridx, csgridy, smbfit, 'SMB (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', period, 'bwr_r', -1, 0, 1, 'smb_map_myr', figpath)
plot.fitmap(csgridx, csgridy, smbfiterror, 'SMB (RACMO2.3p2) \n Elevation \n Change error \n (m w.e./yr)', period, 'Reds', 0, 0.05, 0.1, 'smberror_map_myr', figpath)

residuals = csfit - fdmfit
residualserror = np.sqrt(csfiterror**2 + fdmfiterror**2)
plot.fitmap(csgridx, csgridy, residuals, 'Residuals \n (m/yr)', period, 'bwr_r', -1, 0, 1, 'residuals_map_myr', figpath)
plot.fitmap(csgridx, csgridy, residualserror, 'Residuals error \n (m/yr)', period, 'Reds', 0, 0.05, 0.1, 'residualserror_map_myr', figpath)

#
plot.fitmap(csgridx, csgridy, rhomodel, 'Density \n Model \n top 1m \n (kg/m3)', period, 'PiYG_r', 310, 600, 917, 'rhomodel_1m_map_myr', figpath)
plot.fitmap(csgridx, csgridy, rhomodel2, 'Density \n Model \n top 32cm \n (kg/m3)', period, 'PiYG_r', 310, 600, 917, 'rhomodel_32cm_map_myr', figpath)

rhomodels = rhosgrid.copy()
rhomodels[ice_mask==rho_i] = rho_i #[kg/m2/yr]
plot.fitmap(csgridx, csgridy, rhomodels, 'Density \n Model \n Surface \n (kg/m3)', period, 'PiYG_r', 200, 310, 917, 'rhomodel_surface_map_myr', figpath)

plot.fitmap(csgridx, csgridy, runoffgrid, 'Runoff \n (m. w.e./yr)', period, 'Blues', 0, 0.25, 0.5, 'runoff_map_myr', figpath)

plot.fitmap(csgridx, csgridy, snowmeltgrid, 'Melt \n (m. w.e./yr)', period, 'Purples', 0, 0.25, 0.5, 'snowmelt_map_myr', figpath)
plot.fitmap(csgridx, csgridy, snowfallgrid, 'Snowfall \n (m. w.e./yr)', period, 'YlOrRd', 0, 0.25, 0.5, 'snowfall_map_myr', figpath)

#
plot.fitmap(csgridx, csgridy, masstotalgrid_ao, 'Altimetry Only \n Method \n Total Mass \n Change \n (kg)', period, 'PiYG', -4500, 0, 4500, 'masstotal_ao_map_kg', figpath)
plot.fitmap(csgridx, csgridy, masstotalgrid_k, 'Kappelsberger \n Method \n Total Mass \n Change \n (kg)', period, 'PiYG', -4500, 0, 4500, 'masstotal_k_map_kg', figpath)
plot.fitmap(csgridx, csgridy, masstotalgrid_imau, 'IMAU \n Method \n Total Mass \n Change \n (kg)', period, 'PiYG', -4500, 0, 4500, 'masstotal_imau_map_kg', figpath)
plot.fitmap(csgridx, csgridy, masstotalgrid_mcm, 'McMillan \n Method \n Total Mass \n Change \n (kg)', period, 'PiYG', -4500, 0, 4500, 'masstotal_mcm_map_kg', figpath)

plot.fitmap(csgridx, csgridy, masstotalerrorgrid_ao, 'Altimetry Only \n Method \n Total Mass \n Change error \n (kg)', period, 'RdPu', 0, 250, 500, 'masstotalerror_ao_map_kg', figpath)
plot.fitmap(csgridx, csgridy, masstotalerrorgrid_k, 'Kappelsberger \n Method \n Total Mass \n Change error \n (kg)', period, 'RdPu', 0, 250, 500, 'masstotalerror_k_map_kg', figpath)
plot.fitmap(csgridx, csgridy, masstotalerrorgrid_imau, 'IMAU \n Method \n Total Mass \n Change error \n (kg)', period,  'RdPu', 0, 250, 500, 'masstotalerror_imau_map_kg', figpath)
plot.fitmap(csgridx, csgridy, masstotalerrorgrid_mcm, 'McMillan \n Method \n Total Mass \n Change error \n (kg)', period, 'RdPu', 0, 250, 500, 'masstotalerror_mcm_map_kg', figpath)


#
plot.fitmap(csgridx, csgridy, massgridkg_ao, 'Altimetry Only \n Method \n Mass Change \n Rate \n (kg/yr)', period, 'PuOr', -500, 0, 500, 'mass_ao_map_kgyr', figpath)
plot.fitmap(csgridx, csgridy, massgridkg_k, 'Kappelsberger \n Method \n Mass Change \n Rate \n (kg/yr)', period, 'PuOr', -500, 0, 500, 'mass_k_map_kgyr', figpath)
plot.fitmap(csgridx, csgridy, massgridkg_imau, 'IMAU \n Method \n Mass Change \n Rate \n (kg/yr)', period, 'PuOr', -500, 0, 500, 'mass_imau_map_kgyr', figpath)
plot.fitmap(csgridx, csgridy, massgridkg_mcm, 'McMillan \n Method \n Mass Change \n Rate \n (kg/yr)', period, 'PuOr', -500, 0, 500, 'mass_mcm_map_kgyr', figpath)

plot.fitmap(csgridx, csgridy, masserrorkg_ao, 'Altimetry Only \n Method \n Mass Change \n Rate error \n (kg/yr)', period, 'Oranges', 0, 25, 50, 'masserror_ao_map_kgyr', figpath)
plot.fitmap(csgridx, csgridy, masserrorkg_k, 'Kappelsberger \n Method \n Mass Change \n Rate error \n (kg/yr)', period, 'Oranges', 0, 25, 50, 'masserror_k_map_kgyr', figpath)
plot.fitmap(csgridx, csgridy, masserrorkg_imau, 'IMAU \n Method \n Mass Change \n Rate error \n (kg/yr)', period, 'Oranges', 0, 25, 50, 'masserror_imau_map_kgyr', figpath)
plot.fitmap(csgridx, csgridy, masserrorkg_mcm, 'McMillan \n Method \n Mass Change \n Rate error \n (kg/yr)', period, 'Oranges', 0, 25,  50, 'masserror_mcm_map_kgyr', figpath)

###############################################################################
"""MASK MAPS"""

"Topography"
topomask= analise.select_icesheet_onegrid(topomask, csfit)
topomask[topomask==0] = np.nan

"Ablation, Accumulation, Dynamic"
ablation_mask[dynamic_mask==1] = 2
plot.maskmap(csgridx, csgridy, ablation_mask, '', period, 'cool_r', 1, 2, 'mask_map', figpath)

"Basins"
basin_mask[basin_mask==0] = np.nan
plot.basinmap(csgridx, csgridy, basin_mask, 'Basins', period, 'Dark2_r', 0, 7, 'basin_map', figpath)


"""BASINS AND STATIONS PLOTTING"""
"Basin area and %"
basin_NW_area = len(basin_mask[basin_mask==1]) * cell_area * 10**-6 #km2
basin_CW_area = len(basin_mask[basin_mask==2]) * cell_area * 10**-6 #km2
basin_SW_area = len(basin_mask[basin_mask==3]) * cell_area * 10**-6 #km2
basin_SE_area = len(basin_mask[basin_mask==4]) * cell_area * 10**-6 #km2
basin_CE_area = len(basin_mask[basin_mask==5]) * cell_area * 10**-6 #km2
basin_NE_area = len(basin_mask[basin_mask==6]) * cell_area * 10**-6 #km2
basin_NO_area = len(basin_mask[basin_mask==7]) * cell_area * 10**-6 #km2


basin_NW_p = np.round((len(basin_mask[basin_mask==1])*100)/np.sum(~np.isnan(basin_mask)), 2)
basin_CW_p = np.round((len(basin_mask[basin_mask==2])*100)/np.sum(~np.isnan(basin_mask)), 2)
basin_SW_p = np.round((len(basin_mask[basin_mask==3])*100)/np.sum(~np.isnan(basin_mask)), 2)
basin_SE_p = np.round((len(basin_mask[basin_mask==5])*100)/np.sum(~np.isnan(basin_mask)), 2)
basin_NE_p = np.round((len(basin_mask[basin_mask==6])*100)/np.sum(~np.isnan(basin_mask)), 2)
basin_NO_p = np.round((len(basin_mask[basin_mask==7])*100)/np.sum(~np.isnan(basin_mask)), 2)



"plot per station and per basin"

plot.stations_massseries(series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', figpath)

plot.basins_massseries(basin_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', figpath)


"GRACE basins mass series"
cstime = [str(int(t))[:6] for t in csgstime]
gtime = [t[:6] for t in grace_time]
gracetime = [t if t in gtime else np.nan for t in cstime]

grace_fits = []
graceerror_fits = []
for j,r in enumerate(grace_regions):
    tseries = []
    teseries = []
    for i,t in enumerate(gracetime):
        if t in gtime:
            index = [k for k,p in enumerate(gtime) if str(p).startswith(str(t))]            
            tseries.extend((grace_data[j][index])/10**12) #convert from kg to Gt
            teseries.extend((grace_error[j][index])/10**12)
        else:
            tseries.append(np.nan)
            teseries.append(np.nan)

    tsint = pd.Series(tseries)
    tsint = tsint.interpolate()    
    ts_noseason = analise.remove_seasonality_timeseries(tsint[1:])
    gts, gtse = analise.fit_series(ts_noseason)
    
    grace_fits.append(gts)
    graceerror_fits.append(gtse)


plot.basinsgrace_massseries(basingrace_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, grace_fits, graceerror_fits, 'Gt', 'Mass change', figpath)

###############################################################################
"""FACIES RATES AND PLOTS"""
"""Accumulationa and ablation only - rates and plots"""
rates_ab, errors_ab = plot.mask_massseries(ablation_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', 'ablation', figpath)

rates_ac, errors_ac = plot.mask_massseries(accumulation_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', 'accumulation', figpath)

rates_dy, errors_dy = plot.mask_massseries(dynamic_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', 'dynamic', figpath)

#percolation
percolation_mask = f_mask.copy()
percolation_mask[percolation_mask != 4] = np.nan
percolation_mask[percolation_mask == 4] = 1

rates_p, errors_p = plot.mask_massseries(percolation_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', 'percolation', figpath)

#wet snow
wetsnow_mask = f_mask.copy()
wetsnow_mask[wetsnow_mask != 3] = np.nan
wetsnow_mask[wetsnow_mask == 3] = 1

rates_ws, errors_ws = plot.mask_massseries(wetsnow_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', 'wet snow', figpath)

#dry snow
drysnow_mask = f_mask.copy()
drysnow_mask[drysnow_mask != 5] = np.nan
drysnow_mask[drysnow_mask == 5] = 1

rates_ds, errors_ds = plot.mask_massseries(drysnow_mask, series_time, csgridx, csgridy, csgs, cse_grids, rhomodel, rhogriderror, rhomodel2, rho2griderror, fdmgs, fdme_grids, smbgs, compaction_grids, compactiongriderror, 'Gt', 'Mass change', 'dry snow', figpath)


zones = ['Ablation', 'Dynamic', 'Wet snow', 'Percolation', 'Dry snow', 'All Accumulation']
rates = [rates_ab, rates_dy, rates_ws, rates_p, rates_ds, rates_ac]
errors = [errors_ab, errors_dy, errors_ws, errors_p, errors_ds, errors_ac]

methods = ['AO', 'K', 'MCM', 'IMAU']
colors = ['purple', 'orange', 'red', 'mediumblue']

all_zones =[]
all_rates = []
all_errors =[]
all_methods =[]
all_colors = []
for i,z in enumerate(zones):
    
    zr = rates[i]
    ze = errors[i]
    
    zz = [z for r in zr]
    
    all_zones.extend(zz)
    all_rates.extend(zr)
    all_errors.extend(ze)
    all_methods.extend(methods)
    all_colors.extend(colors)


plt.rcParams['font.sans-serif'] = "Helvetica"

fig, ax = plt.subplots()

for k,r in enumerate(all_rates):
    
    ax.scatter(all_zones[k], all_rates[k], label= all_methods[k],s = 100, c= all_colors[k], marker = 's', alpha=0.6)
    ax.errorbar(all_zones[k], all_rates[k], yerr= all_errors[k], fmt= '', ecolor='black')

plt.gcf().set_size_inches(8, 6)
ax.legend(all_methods[:4])

plt.xlabel('Basins')
plt.ylabel('Mass change rate [Gt/yr]')

plt.savefig(figpath+ 'Zones_mass_permethod' + '.jpg', dpi = 300, bbox_inches='tight')
plt.clf()


"""Elevation time series"""

#whole GIS
cs_series = [np.nanmean(g) for g in csgs]

fdm_series = [np.nanmean(g) for g in fdmgs]

residual_series = [np.nanmean(csgs[i] - fdmgs[i]) for i,g in enumerate(csgs)]
residualerror_series = [np.sqrt(analise.function_Hurkmans(g, 3)**2 + np.nanmean(fdme_grids[i])**2) for i,g in enumerate(cse_grids)]

compaction_series = [np.nanmean(g) for g in compaction_grids]
compactionerror_series = [np.nanmean(compactiongriderror) for n in compaction_grids]

plot.timeseries_elev4(series_time, cs_series, cse_series, 'CryoSat-2', fdm_series, fdme_series, 'IMAU-FDM', residual_series, residualerror_series, 'Residuals', compaction_series, compactionerror_series, 'Compaction', 'Elevation Change', '[m]', 'Average_elevation_change_GIS', figpath)


#Acc, abl, dynamic
compactionegs = np.array([compactiongriderror for g in compaction_grids])

plot.mask_elevseries(ablation_mask, series_time, csgridx, csgridy, csgs, cse_grids, fdmgs, fdme_grids, compaction_grids, compactionegs, smbgs, 'm', 'Average elevation change', 'ablation', figpath)
plot.mask_elevseries(dynamic_mask, series_time, csgridx, csgridy, csgs, cse_grids, fdmgs, fdme_grids, compaction_grids, compactionegs, smbgs, 'm', 'Average elevation change', 'dynamic', figpath)
plot.mask_elevseries(accumulation_mask, series_time, csgridx, csgridy, csgs, cse_grids, fdmgs, fdme_grids, compaction_grids, compactionegs, smbgs, 'm', 'Average elevation change', 'accumulation', figpath)


#basins
plot.basins_elevseries(basin_mask, series_time, csgridx, csgridy, csgs, cse_grids, fdmgs, fdme_grids, compaction_grids, compactionegs, smbgs, 'm', 'Average elevation change', figpath)

plot.stations_elevseries(series_time, csgridx, csgridy, csgs, cse_grids, fdmgs, fdme_grids, compaction_grids, compactionegs, smbgs, 'm', 'Average elevation change', figpath)

###############################################################################
"""GRACE DATA"""

gdata = pd.read_csv(gpath, sep=",", header=None)
gdata.columns = ["author", "source", "method", "area", "id1", "id2", "time", "cumulative_mass", "2std"]


dtime = [datetime(int(x), 1, 1) + timedelta(days = (x % 1) * (366 if calendar.isleap(int(x)) else 365)) for x in gdata['time']]

gtime = [t.date() for t in dtime]

df = pd.DataFrame({'time': gtime})

gdata['time'] = df['time']

mdata_z = gdata.loc[gdata['method'] == 'Zwally']
mdata_r = gdata.loc[gdata['method'] == 'Rignot']

mdata_z.set_index(mdata_z['time'], drop=True, inplace=True)

grace_series = pd.Series(mdata_z['cumulative_mass'], index = pd.DatetimeIndex(mdata_z['time']))
grace_serieserror = pd.Series(mdata_z['2std'], index = pd.DatetimeIndex(mdata_z['time']))


grace_resampled = grace_series.resample('M').ffill()
graceerror_resampled = grace_serieserror.resample('M').ffill()

"""Extract time period, ploting and yearly rate"""

mdata_zr = grace_resampled.truncate(before = datetime(int(ts[:4]),int(ts[4:]),1).date()).truncate(after = datetime(int(te[:4]),int(te[4:]),em).date())
mdata_zre = graceerror_resampled.truncate(before = datetime(int(ts[:4]),int(ts[4:]),1).date()).truncate(after = datetime(int(te[:4]),int(te[4:]),em).date())


refmass_z = [m - mdata_zr.iloc[0] for m in mdata_zr]

mdata_zr['ref_mass'] = refmass_z


"""Remove seasonality from GRACE data and compute RMSE, plot GRACE vs other mass timeseries"""

grace_noseason = analise.remove_seasonality_timeseries(mdata_zr['ref_mass'])
plot.grace_noseason_series(series_time, grace_noseason, mdata_zre,'GRACE', ts, te, 'Mass change', 'Gt', 'GRACE', outpath)

plot.timeseries_mass5(series_time, gis_massseries_ao, gis_masserrband_ao, 'Altimetry only', gis_massseries_k, gis_masserrband_k, 
                      'Kappelsberger', gis_massseries_mcm, gis_masserrband_mcm, 'McMillan', gis_massseries_imau, gis_masserrband_imau, 'IMAU', 
                      grace_noseason, mdata_zre,'GRACE', 'GIS Mass change using 4 conversion methods', '[Gt]', 'Total_mass_GIS_5', figpath)

smb_timeseries = [np.nanmean(g)*rho_w for g in smbgs]

plot.timeseries_mass6(series_time, gis_massseries_ao, gis_masserrband_ao, 'Altimetry only', gis_massseries_k, gis_masserrband_k, 
                      'Kappelsberger', gis_massseries_mcm, gis_masserrband_mcm, 'McMillan', gis_massseries_imau, gis_masserrband_imau, 'IMAU', 
                      grace_noseason, mdata_zre,'GRACE', smb_timeseries, 'SMB', 'GIS Mass change using 4 conversion methods', '[Gt]', 'Total_mass_GIS_6', figpath)


grace_detrend = analise.detrend(grace_noseason)

#AO
mass_ao_detrend = analise.detrend(gis_massseries_ao)

rval_ao = round(np.corrcoef(mass_ao_detrend, grace_detrend)[0, 1], 2)
rmseval_ao = int(round(np.sqrt(mean_squared_error(mass_ao_detrend, grace_detrend)),0))

plot.grace_rmse(mass_ao_detrend, grace_detrend,'Altimetry Only vs GRACE', ts, te, 'Mass change', 'Gt', 'Altimetry_Only', outpath, str(rmseval_ao), str(rval_ao))


#K
mass_k_detrend = analise.detrend(gis_massseries_k)

rval_k = round(np.corrcoef(mass_k_detrend, grace_detrend)[0, 1], 2)
rmseval_k = int(round(np.sqrt(mean_squared_error(mass_k_detrend, grace_detrend)),0))

plot.grace_rmse(mass_k_detrend, grace_detrend,'Kappelsberger vs GRACE', ts, te, 'Mass change', 'Gt', 'Kappelsberger', outpath, str(rmseval_k), str(rval_k))


#MCM
mass_mcm_detrend = analise.detrend(gis_massseries_mcm)

rval_mcm = round(np.corrcoef(mass_mcm_detrend, grace_detrend)[0, 1], 2)
rmseval_mcm = int(round(np.sqrt(mean_squared_error(mass_mcm_detrend, grace_detrend)),0))

plot.grace_rmse(mass_mcm_detrend, grace_detrend,'McMillan vs GRACE', ts, te, 'Mass change', 'Gt', 'McMillan', outpath, str(rmseval_mcm), str(rval_mcm))


#IMAU
mass_imau_detrend = analise.detrend(gis_massseries_imau)

rval_imau = round(np.corrcoef(mass_imau_detrend, grace_detrend)[0, 1], 2)
rmseval_imau = int(round(np.sqrt(mean_squared_error(mass_imau_detrend, grace_detrend)),0))

plot.grace_rmse(mass_imau_detrend, grace_detrend,'IMAU vs GRACE', ts, te, 'Mass change', 'Gt', 'IMAU', outpath, str(rmseval_imau), str(rval_imau))

###############################################################################
"SMB mass plot"
plot.smb_series(series_time, smb_timeseries,'SMB', ts, te, 'Mass change', 'Gt', 'SMB_mass_change', figpath)

###############################################################################
"FDM vs CS2 RMSE"

cs_series = [np.nanmean(g) for g in csgs] #noseason
fdm_series = [np.nanmean(g) for g in fdmgs] #noseason

rno = round(np.corrcoef(fdm_series, cs_series)[0, 1], 2)
rmseno = round(np.sqrt(mean_squared_error(fdm_series, cs_series)),2)
plot.elev_rmse(fdm_series, cs_series, ts, te, 'm', 'noseason', figpath, rmseno, rno)

#load 'raw' CS2 & FDM data
cs2refseries_path = '/Users/kat/DATA/Firn_data/cs_ref_' + ts + '_' + te + '.nc'
fdm2refseries_path = '/Users/kat/DATA/Firn_data/fdm_ref_' + ts + '_' + te + '.nc'

csrx, csry, csrtime, csr, csrunit, csrprojection= load.grid_series(cs2refseries_path, 'cs_ref')
fdmrx, fdmry, fdmrtime, fdmr, fdmrunit, fdmrprojection= load.grid_series(fdm2refseries_path, 'fdm_ref')

csr_series = [np.nanmean(g) for g in csr]
fdmr_series = [np.nanmean(g) for g in fdmr]

rno = round(np.corrcoef(fdmr_series, csr_series)[0, 1], 2)
rmseno = round(np.sqrt(mean_squared_error(fdmr_series, csr_series)),2)
plot.elev_rmse(fdmr_series, csr_series, ts, te, 'm', 'referenced', figpath, rmseno, rno)

###############################################################################
"Nilsson plot"

#plot around NEEM with referenced CS2 data

plot.nilsson_elevseries(series_time, csgridx, csgridy, csr, cse_grids, 'm', 'NEEM elevation change', figpath)

###############################################################################
"N1 basin of Kappelberger"

n1_path = '/Users/kat/DATA/Firn_data/n1basin_mask.nc'
n1gridx, n1gridy, n1_mask = load.load_mask(n1_path, 'n1_mask')

n1gridx = csgridx[900:1600, 400:800]
n1gridy = csgridy[900:1600, 400:800]

f_mask = facies_mask.copy()
f_mask[f_mask==0] = np.nan
f_mask = f_mask[700:, 250:]
fgridx = csgridx[700:, 250:]
fgridy = csgridy[700:, 250:]

t_mask = topomask.copy()
t_mask = t_mask[900:1600, 400:800]


#AO
n1_ao = massgridkg_ao * n1_mask
n1_ao[n1_ao==0] = np.nan
n1_ao = n1_ao[900:1600, 400:800]

n1r_ao = massgrid_ao * n1_mask
n1r_ao[n1r_ao==0] = np.nan
n1_massrate_ao = np.round(np.nanmean(n1r_ao) * np.sum(~np.isnan(n1r_ao)) * cell_area, 2) #[Gt/yr] 

n1rerror_ao = masserror_ao * n1_mask
n1_masserrorrate_ao = analise.function_Hurkmans(csfiterror * n1_mask, 3) * rho_i * 10**-12 #[Gt/m2/yr] rate per m2
n1_masserrrate_ao = np.round(n1_masserrorrate_ao * np.sum(~np.isnan(n1rerror_ao)) * cell_area, 2) #[Gt/yr] GIS rate


#Kappelsberger
n1_k = massgridkg_k * n1_mask
n1_k[n1_k==0] = np.nan
n1_k = n1_k[900:1600, 400:800]

n1r_k = massgrid_k * n1_mask
n1r_k[n1r_k==0] = np.nan
n1_massrate_k = np.round(np.nanmean(n1r_k) * np.sum(~np.isnan(n1r_k)) * cell_area, 2) #[Gt/yr] 

n1rerror_k = masserror_k * n1_mask
n1_masserrorrate_k = np.sqrt((analise.function_Hurkmans(csfiterror * n1_mask, 3)*rho_i)**2 + (np.nanmean(csfit*n1_mask)*np.nanmean(rhogriderror*n1_mask))**2) * 10**-12 #[Gt/m2/yr] rate per m2
n1_masserrrate_k = np.round(n1_masserrorrate_k * np.sum(~np.isnan(n1rerror_k)) * cell_area, 2) #[Gt/yr] GIS rate


#McMillan
n1_mcm = massgridkg_mcm * n1_mask
n1_mcm[n1_mcm==0] = np.nan
n1_mcm = n1_mcm[900:1600, 400:800]

n1r_mcm = massgrid_mcm * n1_mask
n1r_mcm[n1r_mcm==0] = np.nan
n1_massrate_mcm = np.round(np.nanmean(n1r_mcm) * np.sum(~np.isnan(n1r_mcm)) * cell_area, 2) #[Gt/yr] 

n1rerror_mcm = masserror_mcm * n1_mask
n1_masserrorrate_mcm = np.sqrt((np.sqrt(analise.function_Hurkmans(csfiterror*n1_mask, 3)**2 + np.nanmean(compactiongriderror*n1_mask)**2) * np.nanmean(rho2grid*n1_mask))**2 + np.nanmean(altimetry_compact*rho2griderror*n1_mask)**2) * 10**-12
n1_masserrrate_mcm = np.round(n1_masserrorrate_mcm * np.sum(~np.isnan(n1rerror_mcm)) * cell_area, 2) #[Gt/yr] GIS rate


#IMAU
n1_imau = massgridkg_imau * n1_mask
n1_imau[n1_imau==0] = np.nan
n1_imau = n1_imau[900:1600, 400:800]

n1r_imau = massgrid_imau * n1_mask
n1r_imau[n1r_imau==0] = np.nan
n1_massrate_imau = np.round(np.nanmean(n1r_imau) * np.sum(~np.isnan(n1r_imau)) * cell_area, 2) #[Gt/yr] 

n1rerror_imau = masserror_imau * n1_mask
n1_masserrorrate_imau = np.sqrt(analise.function_Hurkmans(csfiterror*n1_mask, 3)**2 + np.nanmean(fdmfiterror*n1_mask)**2) * rho_i * 10**-12
n1_masserrrate_imau = np.round(n1_masserrorrate_imau * np.sum(~np.isnan(n1rerror_imau)) * cell_area, 2) #[Gt/yr] GIS rate


#plot
n1_grids = [n1_ao, n1_k, n1_mcm, n1_imau]
n1_titles = ['Altimetry Only \n Method \n Mass Change \n Rate \n (kg/yr)', 'Kappelsberger \n Method \n Mass Change \n Rate \n (kg/yr)', 
             'McMillan \n Method \n Mass Change \n Rate \n (kg/yr)', 'IMAU \n Method \n Mass Change \n Rate \n (kg/yr)']

#We use only PROMICE and GCNet stations
stations_names = ['KPC_L', 'KPC_U', 'EGP']    
stations_coords = [[391168, -1023448], [374565, -1038474], [245519, -1545750]]
stations_labels = [[301168, -1003448], [334565, -1108474], [245519, -1515750]]

color1 = 'black'
color2 = 'white'

plot.massmap4(n1gridx, n1gridy, n1_grids, n1_titles, fgridx, fgridy, f_mask, t_mask, [-20, -50, 70, 83.5], period, stations_names, stations_coords, stations_labels, 'PuOr', -200, 0, 200, color1, color2, 950000, -920000, 'dynamic', 'n1_basin_mass', figpath)


#Rates for the N1 for different methods:
n1rate_data = [n1_massrate_ao, n1_masserrrate_ao, n1_massrate_k, n1_masserrrate_k, n1_massrate_mcm, n1_masserrrate_mcm, n1_massrate_imau, n1_masserrrate_imau]   
save.basin_rates_txt(n1rate_data,'Mass change rates for N1 basin', ts, te, 'n1_rates.txt', figpath)


n1gridx = csgridx[900:1600, 400:800]
n1gridy = csgridy[900:1600, 400:800]

n1_cs = csfit*n1_mask
n1_cs = n1_cs[900:1600, 400:800]

n1_fdm = fdmfit*n1_mask
n1_fdm = n1_fdm[900:1600, 400:800]

n1_smb = smbfit*n1_mask
n1_smb = n1_smb[900:1600, 400:800]

n1_comp = compactiongrid*n1_mask
n1_comp = n1_comp[900:1600, 400:800]


n1_grids = [n1_cs, n1_fdm, n1_smb, n1_comp]
n1_titles = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)']


plot.massmap4(n1gridx, n1gridy, n1_grids, n1_titles, fgridx, fgridy, f_mask, t_mask, [-20, -50, 70, 83.5], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 950000, -920000, 'dynamic', 'n1_basin_elev', figpath)


n1_m = snowmeltgrid*n1_mask
n1_m = n1_m[900:1600, 400:800]

n1_r = runoffgrid*n1_mask
n1_r = n1_r[900:1600, 400:800]

n1_sf = snowfallgrid*n1_mask
n1_sf = n1_sf[900:1600, 400:800]


n1_grids = [n1_cs, n1_fdm, n1_smb, n1_comp, n1_m, n1_sf]
n1_titles = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)', 
             'Melt \n (m w.e./yr)', 'Snowfall \n (m w.e./yr)']

plot.massmap6(n1gridx, n1gridy, n1_grids, n1_titles, fgridx, fgridy, f_mask, t_mask, [-20, -50, 70, 83.5], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 950000, -930000, 'dynamic', 'n1_basin_elev6', figpath)

###############################################################################
"""BASIN PLOTTING CASE STUDY PLOTTING"""
mass_grids = [massgridkg_ao, massgridkg_k, massgridkg_mcm, massgridkg_imau]
elev_grids = [csfit, fdmfit, smbfit, compactiongrid]
snow_grids = [snowmeltgrid, snowfallgrid]

"NO basin"
nogridx, nogridy, no_massgrids, no_elevgrids, no_snowgrids, fgridx, fgridy, f_mask, t_mask = analise.basin_subgrid(basin_mask, facies_mask, topomask, csgridx, csgridy, mass_grids, elev_grids, snow_grids, 1, 1200, 1700, 0, 650, 800, -1, 0, -1)

#Stations
stations_names = ['Peterman_ELA', 'GITS', 'NEEM']    
stations_coords = [[-244306, -1047222], [-386548, -1344511],[-138963, -1351904]]
stations_labels = [[-244306, -1017222], [-386548, -1314511],[-138963, -1321904]]

#plot title colors
color1 = 'white'
color2 = 'black'

#plot
no_titles1 = ['Altimetry Only \n Method \n Mass Change \n Rate \n (kg/yr)', 'Kappelsberger \n Method \n Mass Change \n Rate \n (kg/yr)', 
             'McMillan \n Method \n Mass Change \n Rate \n (kg/yr)', 'IMAU \n Method \n Mass Change \n Rate \n (kg/yr)']

plot.massmap4(nogridx, nogridy, no_massgrids, no_titles1, fgridx, fgridy, f_mask, t_mask, [-30, -60, 70, 83.5], period, stations_names, stations_coords, stations_labels, 'PuOr', -200, 0, 200, color1, color2, 550000, -920000, 'dynamic', 'no_basin_mass', figpath)


no_titles2 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)']

plot.massmap4(nogridx, nogridy, no_elevgrids, no_titles1, fgridx, fgridy, f_mask, t_mask, [-30, -60, 70, 83.5], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 550000, -920000, 'dynamic', 'no_basin_elev', figpath)


no_titles3 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)', 
             'Melt \n (m w.e./yr)', 'Snowfall \n (m w.e./yr)']

no_grids = no_elevgrids.extend(no_snowgrids)
plot.massmap6(nogridx, nogridy, no_grids, no_titles3, fgridx, fgridy, f_mask, t_mask, [-30, -60, 70, 83.5], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 550000, -930000, 'dynamic', 'no_basin_elev6', figpath)


###############################################################################
"SW basin"

swgridx, swgridy, sw_massgrids, sw_elevgrids, sw_snowgrids, fgridx, fgridy, f_mask, t_mask = analise.basin_subgrid(basin_mask, facies_mask, topomask, csgridx, csgridy, mass_grids, elev_grids, snow_grids, 5, 0, 800, 0, -1, 0, 1000, 0, -1)

#Stations
stations_names = ['South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2']    
stations_coords = [[9445, -2960973], [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094]]
stations_labels = [[-129445, -2930973], [-356931, -2484048], [-178343, -2490999], [-70205, -2552571], [-107831, -2652094]]

#plot title colors
color1 = 'white'
color2 = 'black'

#plot
sw_titles1 = ['Altimetry Only \n Method \n Mass Change \n Rate \n (kg/yr)', 'Kappelsberger \n Method \n Mass Change \n Rate \n (kg/yr)', 
             'McMillan \n Method \n Mass Change \n Rate \n (kg/yr)', 'IMAU \n Method \n Mass Change \n Rate \n (kg/yr)']

plot.massmap4(swgridx, swgridy, sw_massgrids, sw_titles1, fgridx, fgridy, f_mask, t_mask, [-40, -55, 58.8, 69], period, stations_names, stations_coords, stations_labels, 'PuOr', -200, 0, 200, color1, color2, 285000, -2520000, 'wet', 'sw_basin_mass', figpath)


sw_titles2 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)']


plot.massmap4(swgridx, swgridy, sw_elevgrids, sw_titles2, fgridx, fgridy, f_mask, t_mask, [-40, -55, 58.8, 69], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 285000, -2520000, 'wet', 'sw_basin_elev', figpath)


sw_grids = sw_elevgrids.extend(sw_snowgrids)
sw_titles = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)', 
             'Melt \n (m w.e./yr)', 'Snowfall \n (m w.e./yr)']

plot.massmap6(swgridx, swgridy, sw_grids, sw_titles, fgridx, fgridy, f_mask, t_mask, [-40, -55, 58.8, 69], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 285000, -2570000, 'wet', 'sw_basin_elev6', figpath)

###############################################################################
"NW basin"

nwgridx, nwgridy, nw_massgrids, nw_elevgrids, nw_snowgrids, fgridx, fgridy, f_mask, t_mask = analise.basin_subgrid(basin_mask, facies_mask, topomask, csgridx, csgridy, mass_grids, elev_grids, snow_grids, 7, 600, 1300, 0, 600, 500, -1, 0, -1)

#Stations
stations_names = ['UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']    
stations_coords = [[-301483, -1841957], [-278493, -1846178], [-138429, -1756197], [-386548, -1344511],[-138963, -1351904]]
stations_labels = [[-321483, -1811957], [-248493, -1856178], [-138429, -1726197], [-386548, -1314511],[-138963, -1321904]]

#plot title colors
color1 = 'white'
color2 = 'black'


#plot
nw_titles1 = ['Altimetry Only \n Method \n Mass Change \n Rate \n (kg/yr)', 'Kappelsberger \n Method \n Mass Change \n Rate \n (kg/yr)', 
             'McMillan \n Method \n Mass Change \n Rate \n (kg/yr)', 'IMAU \n Method \n Mass Change \n Rate \n (kg/yr)']

plot.massmap4(nwgridx, nwgridy, nw_massgrids, nw_titles1, fgridx, fgridy, f_mask, t_mask, [-37, -65, 67, 80.5], period, stations_names, stations_coords, stations_labels, 'PuOr', -200, 0, 200, color1, color2, 350000, -1250000, 'dynamic', 'nw_basin_mass', figpath)


nw_titles2 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)']

plot.massmap4(nwgridx, nwgridy, nw_elevgrids, nw_titles2, fgridx, fgridy, f_mask, t_mask, [-37, -65, 67, 80.5], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.5, 0, 0.5, color1, color2, 350000, -1250000, 'dynamic', 'nw_basin_elev', figpath)


nw_grids = nw_elevgrids.extend(nw_snowgrids)
nw_titles3 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)', 
             'Melt \n (m w.e./yr)', 'Snowfall \n (m w.e./yr)']

plot.massmap6(nwgridx, nwgridy, nw_grids, nw_titles3, fgridx, fgridy, f_mask, t_mask, [-37, -65, 67, 80.5], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 350000, -1300000, 'dynamic', 'nw_basin_elev6', figpath)

###############################################################################
"CW basin"

cwgridx, cwgridy, cw_massgrids, cw_elevgrids, cw_snowgrids, fgridx, fgridy, f_mask, t_mask = analise.basin_subgrid(basin_mask, facies_mask, topomask, csgridx, csgridy, mass_grids, elev_grids, snow_grids, 6, 200, 900, 0, -1, 0, 1100, 0, -1)

#Stations
stations_names = ['JAR_1', 'Swiss Camp', 'CP1']    
stations_coords = [[-184047, -2236749], [-168891, -2230133], [-76703, -2200166]]
stations_labels = [[-229047, -2306749], [-148891, -2260133], [-76703, -2170166]]

#Plot titles colors
color1 = 'white'
color2 = 'black'


#plot
cw_titles1 = ['Altimetry Only \n Method \n Mass Change \n Rate \n (kg/yr)', 'Kappelsberger \n Method \n Mass Change \n Rate \n (kg/yr)', 
             'McMillan \n Method \n Mass Change \n Rate \n (kg/yr)', 'IMAU \n Method \n Mass Change \n Rate \n (kg/yr)']

plot.massmap4(cwgridx, cwgridy, cw_massgrids, cw_titles1, fgridx, fgridy, f_mask, t_mask, [-36, -54, 63, 74], period, stations_names, stations_coords, stations_labels, 'PuOr', -200, 0, 200, color1, color2, 475000, -2000000, 'dynamic', 'cw_basin_mass', figpath)


cw_titles2 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)']


plot.massmap4(cwgridx, cwgridy, cw_elevgrids, cw_titles2, fgridx, fgridy, f_mask, t_mask, [-36, -54, 63, 74], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 475000, -2000000, 'dynamic', 'cw_basin_elev', figpath)


cw_grids = cw_elevgrids.extend(cw_snowgrids)
cw_titles3 = ['CyroSat-2 \n Elevation \n Change \n (m/yr)', 'IMAU-FDM \n Elevation \n Change \n (m/yr)', 
             'SMB \n (RACMO2.3p2) \n Elevation \n Change \n (m w.e./yr)', 'Compaction \n (m/yr)', 
             'Melt \n (m w.e./yr)', 'Snowfall \n (m w.e./yr)']

plot.massmap6(cwgridx, cwgridy, cw_grids, cw_titles3, fgridx, fgridy, f_mask, t_mask, [-36, -54, 63, 74], period, stations_names, stations_coords, stations_labels, 'bwr_r', -0.2, 0, 0.2, color1, color2, 475000, -2050000, 'dynamic', 'cw_basin_elev6', figpath)


"""'Dry percolation zone' in CW basin"""
cw_mask = basin_mask.copy()
cw_mask[cw_mask!=6] = np.nan
cw_mask[cw_mask==6] = 1

dp_mask = cw_mask.copy()
cw_imau = massgridkg_imau * cw_mask
for i,v in np.ndenumerate(cw_imau):
    
    if v >= 0:
        dp_mask[i] = 1
    elif np.isnan(v):
        dp_mask[i] = np.nan
    else:
        dp_mask[i] = np.nan

dp_cwcs = csgs*dp_mask
dp_cwcsseries = [np.nanmean(c) for c in dp_cwcs]

dp_cwfdm = fdmgs*dp_mask
dp_cwfdmseries = [np.nanmean(c) for c in dp_cwfdm]

dp_cwsmb = smbgs*dp_mask
dp_cwsmbseries = [np.nanmean(c) for c in dp_cwsmb]

dp_cwcomp = compactiongs*dp_mask
dp_cwcompseries = [np.nanmean(c) for c in dp_cwcomp]

dp_cwrun = runoffgs*dp_mask
dp_cwrunseries = [np.nanmean(c) for c in dp_cwrun]

dp_cwmelt = snowmeltgs*dp_mask
dp_cwmeltseries = [np.nanmean(c) for c in dp_cwmelt]

dp_cwsf = snowfallgs*dp_mask
dp_cwsfseries = [np.nanmean(c) for c in dp_cwsf]

#make a plot where cs, fdm and compaction are on meters left y axis, 
#and the rest on m w.e. on right y axis
name = 'dry_percolation_trends_cwbasin'
color1 = 'blue' #cs
color2 = 'magenta' #fdm    
color3 = 'teal' #smb    
color4 = 'red' #compaction
color5 = 'purple' #snowfall
color6 = 'grey' #melt


label1 = 'CryoSat-2'
label2 = 'IMAU-FDM'
label3 = 'SMB'
label4 = 'Compaction'
label5 = 'Snowfall'
label6 = 'Melt'

ylabel1 = 'Elevation Change [m/yr]'
ylabel2 = 'Elevation Change [m w.e./yr]'

plt.rcParams['font.sans-serif'] = "Helvetica"

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(series_time, dp_cwcsseries, color = color1, linewidth=0.8, alpha=0.8 )
ax1.plot(series_time, dp_cwfdmseries, color = color2, linewidth=0.8, alpha=0.8 )
ax1.plot(series_time, dp_cwcompseries, color = color4, linewidth=0.8, alpha=0.8 )

ax2.plot(series_time, dp_cwsmbseries,'--', color = color3, linewidth=0.8, alpha=0.8 )
ax2.plot(series_time, dp_cwsfseries[:-1], '--', color = color5, linewidth=0.8, alpha=0.8 )
ax2.plot(series_time, dp_cwmeltseries, '--', color = color6, linewidth=0.8, alpha=0.8 )


plt.gcf().set_size_inches(8, 4)

ax1.legend([label1, label2, label4], fontsize = 8)
ax2.legend([label3, label5, label6], fontsize = 8)

plt.xlim(series_time[0], series_time[-1])

ax1.set_xlabel('time')
ax1.set_ylabel(ylabel1)
ax2.set_ylabel(ylabel2)

plt.savefig(figpath+name + '.jpg', dpi = 300, bbox_inches='tight')
plt.clf()

