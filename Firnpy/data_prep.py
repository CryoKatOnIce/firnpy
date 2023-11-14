""" This code is for data preperation of CS2, FDM and SMB data for altimetry to mass conversion.
Re-coded after a previous file lost.

Created on: 09/12/2022

@author: katse
"""

"""Libraries"""
from firnpy import load, save, analise, plot
import numpy as np
import matplotlib.pyplot as plt


period_start = '201101'
period_end = '201712'



"""Paths"""
cs2_path = '/Users/kat/DATA/Firn_data/dh_ts_GRL_2011_2018_wide.nc'
fdm_path = '/Users/kat/DATA/Firn_data/FDM_zs_monthlyK_onCS2grid_1957-2020_GrIS_GIC.nc'

rho_path = '/Users/kat/DATA/Firn_data/FDM_dens_1m_monthlyK_onCS2grid_1957-2020_GrIS_GIC.nc'
rho2_path = '/Users/kat/DATA/FIRNPY_DATA/FDM_dens_32cm_monthlyK_onCS2grid5_1957-2020_GrIS_GIC.nc'
compaction_path = '/Users/kat/DATA/Firn_data/FDM_vfc_monthlyK_onCS2grid_1957-2020_GrIS_GIC.nc'

smb_path = '/Users/kat/DATA/Firn_data/smb_monthly_onCS2grid_FGRN055_1957-2020_BN_RACMO23p2.nc'

runoff_path = '/Users/kat/DATA/FIRNPY_DATA/runoff_monthly_onCS2grid_FGRN055_1957-2020_BN_RACMO23p2.nc'
runoffheight_path = '/Users/kat/DATA/FIRNPY_DATA/runoffheight_monthly_onCS2grid_FGRN055_1957-2020_BN_RACMO23p2.nc'

snowmelt_path = '/Users/kat/DATA/FIRNPY_DATA/snowmelt_monthly_onCS2grid_FGRN055_1957-2020_BN_RACMO23p2.nc'
snowfall_path = '/Users/kat/DATA/FIRNPY_DATA/snowfall_monthly_onCS2grid_FGRN055_1957-2020_BN_RACMO23p2.nc'
snowrho_path = '/Users/kat/DATA/FIRNPY_DATA/FDM_dens_surface_monthlyK_onCS2grid5_1957-2020_GrIS_GIC.nc'

"""Load data for study period only to save memory"""

csgridx, csgridy, cstime, cs, cserr, csprojection = load.CS2(period_start, period_end, cs2_path)
fdmgridx, fdmgridy, fdmtime, fdm, fdmprojection = load.FDM_zs(period_start, period_end, fdm_path)

rhogridx, rhogridy, rhotime, rho, rhoprojection = load.FDM_rho(period_start, period_end, rho_path, 'dens1m_monthly')

rho2gridx, rho2gridy, rho2time, rho2, rho2projection = load.FDM_rho(period_start, period_end, rho2_path, 'dens32cm_monthly')


compactiongridx, compactiongridy, compactiontime, compaction, compactionprojection = load.FDM_vfc(period_start, period_end, compaction_path)
comprefgridx, comprefgridy, compreftime, compref, comprefprojection = load.FDM_vfc('196001', '197912', compaction_path) #smb for reference period of 1960-1980

smbgridx, smbgridy, smbtime, smb, smbprojection = load.SMB(period_start, period_end, smb_path)
smbrefgridx, smbrefgridy, smbreftime, smbref, smbrefprojection = load.SMB('196001', '197912', smb_path) #smb for reference period of 1960-1980

runoffgridx, runoffgridy, runofftime, runoff, runoffprojection = load.RUNOFF(period_start, period_end, runoff_path)
runoffrefgridx, runoffrefgridy, runoffreftime, runoffref, runoffrefprojection = load.RUNOFF('196001', '197912', runoff_path) #smb for reference period of 1960-1980

runoffheightgridx, runoffheightgridy, runoffheighttime, runoffheight, runoffheightprojection = load.RUNOFFheight(period_start, period_end, runoffheight_path)

snowmeltgridx, snowmeltgridy, snowmelttime, snowmelt, snowmeltprojection = load.SNOW('2011', '2018','snowmelt', snowmelt_path)
snowmelttime = snowmelttime[:-1]
snowmelt = snowmelt[:-1]

snowmeltrefgridx, snowmeltrefgridy, snowmeltreftime, snowmeltref, snowmeltrefprojection = load.SNOW('1960', '1980', 'snowmelt', snowmelt_path)

snowfallgridx, snowfallgridy, snowfalltime, snowfall, snowfallprojection = load.SNOW('2011', '2018', 'snowfall', snowfall_path)
snowfallrefgridx, snowfallrefgridy, snowfallreftime, snowfallref, snowfallrefprojection = load.SNOW('1960', '1980', 'snowfall', snowfall_path)

rhosgridx, rhosgridy, rhostime, rhos, rhosprojection = load.FDM_rho(period_start, period_end, snowrho_path, 'denssurface_monthly')

#########################################################
"""Icesheet cells only, rest nan"""

fdm = analise.select_icesheet(fdm, cs[0])
rho = analise.select_icesheet(rho, cs[0])
rho2 = analise.select_icesheet(rho2, cs[0])
compaction = analise.select_icesheet(compaction, cs[0])
smb = analise.select_icesheet(smb, cs[0])
smbref = analise.select_icesheet(smbref, cs[0])

runoff = analise.select_icesheet(runoff, cs[0])
runoffref = analise.select_icesheet(runoffref, cs[0])

runoffheight = analise.select_icesheet(runoffheight, cs[0])

snowmelt = analise.select_icesheet(snowmelt, cs[0])
snowmeltref = analise.select_icesheet(snowmeltref, cs[0])

snowfall = analise.select_icesheet(snowfall, cs[0])
snowfallref = analise.select_icesheet(snowfallref, cs[0])
rhos = analise.select_icesheet(rhos, cs[0])

###########################################################
"""Refrencing to start of study period"""

cs_referenced = analise.reference(cs)
fdm_referenced = analise.reference(fdm)


"""remove reference time period from SMB, and convert to cumulative"""

smb_cumulative = analise.referenced_cumulative(smb, smbref)

compaction_ref = compaction - np.nanmean(compref, axis = 0)

runoff_cumulative = analise.referenced_cumulative(runoff, runoffref)

snowmelt_cumulative = analise.referenced_cumulative(snowmelt, snowmeltref)

snowfall_cumulative = analise.referenced_cumulative(snowfall, snowfallref)

"""seasonality removal"""

cs_noseason = analise.remove_seasonality(cs_referenced)
fdm_noseason = analise.remove_seasonality(fdm_referenced)

rho_noseason = analise.remove_seasonality(rho)
rho2_noseason = analise.remove_seasonality(rho2)
compaction_noseason = analise.remove_seasonality(compaction_ref) #compaction 

smb_noseason = analise.remove_seasonality(smb_cumulative)
smb_noseason = smb_noseason/1000 #converting from mm w.e. to m w.e.

runoff_noseason = analise.remove_seasonality(runoff_cumulative)
runoff_noseason = runoff_noseason/1000 #converting from mm w.e. to m w.e.

snowmelt_noseason = analise.remove_seasonality(snowmelt_cumulative)
snowmelt_noseason = snowmelt_noseason/1000 #converting from mm w.e. to m w.e.

snowfall_noseason = analise.remove_seasonality(snowfall_cumulative)
snowfall_noseason = snowfall_noseason/1000 #converting from mm w.e. to m w.e.

rhos_noseason = analise.remove_seasonality(rhos)

####################################################3
"""save"""
out_path = '/Users/kat/DATA/Firn_data'

save.data(cs_referenced, csgridx, csgridy, cstime, 'cs_ref', 'm', 'cumulative', 'cs_ref_'+period_start + '_' + period_end + '.nc', out_path)
save.data(fdm_referenced, fdmgridx, fdmgridy, fdmtime, 'fdm_ref', 'm', 'cumulative', 'fdm_ref_'+period_start + '_' + period_end + '.nc', out_path)


save.data(cs_noseason, csgridx, csgridy, cstime, 'cs_dh', 'm', 'cumulative and with seasonality removed', 'cs_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)
save.data(fdm_noseason, fdmgridx, fdmgridy, fdmtime, 'fdm_dh', 'm', 'cumulative and with seasonality removed', 'fdm_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)

save.data(rho_noseason, rhogridx, rhogridy, rhotime, 'rho_dh', 'kgm3', 'with seasonality removed', 'rho_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)
save.data(rho2_noseason, rho2gridx, rho2gridy, rho2time, 'rho32cm_dh', 'kgm3', 'with seasonality removed', 'rho32cm_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)

save.data(compaction_noseason, compactiongridx, compactiongridy, compactiontime, 'compaction_dh', 'm/year', 'with seasonality removed', 'compaction_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)

save.data(smb_noseason, smbgridx, smbgridy, smbtime, 'smb_dh', 'm w.e.', 'cumulative and with seasonality removed', 'smb_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)

save.data(runoff_noseason, runoffgridx, runoffgridy, runofftime, 'runoff_dh', 'm w.e.', 'cumulative and with seasonality removed', 'runoff_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)

save.data(snowmelt_noseason, snowmeltgridx, snowmeltgridy, snowmelttime, 'snowmelt_dh', 'm w.e.', 'cumulative and with seasonality removed', 'snowmelt_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)

save.data(snowfall_noseason, snowfallgridx, snowfallgridy, snowfalltime, 'snowfall_dh', 'm w.e.', 'cumulative and with seasonality removed', 'snowfall_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)
save.data(rhos_noseason, rhosgridx, rhosgridy, rhostime, 'rhosurf_dh', 'kgm3', 'with seasonality removed', 'rhosurf_dh_noseason_'+period_start + '_' + period_end + '.nc', out_path)


"""linear fit, save"""

cs_fit, cs_fiterror = analise.fit_grid(cs_noseason)
save.fit(cs_fit, cs_fiterror, csgridx, csgridy, 'cs_dhdt', 'm/year', 'cumulative and with seasonality removed', 'cs_dhdt_gridfit_'+period_start + '_' + period_end + '.nc', out_path)

fdm_fit, fdm_fiterror = analise.fit_grid(fdm_noseason)
save.fit(fdm_fit, fdm_fiterror, fdmgridx, fdmgridy, 'fdm_dhdt', 'm/year', 'cumulative and with seasonality removed', 'fdm_dhdt_gridfit_'+period_start + '_' + period_end + '.nc', out_path)

smb_fit, smb_fiterror = analise.fit_grid(smb_noseason)
save.fit(smb_fit, smb_fiterror, smbgridx, smbgridy, 'smb_dhdt', 'm w.e./year', 'cumulative and with seasonality removed', 'smb_dhdt_gridfit_'+period_start + '_' + period_end + '.nc', out_path)

runoff_fit, runoff_fiterror = analise.fit_grid(runoff_noseason)
save.fit(runoff_fit, runoff_fiterror, runoffgridx, runoffgridy, 'runoff_dhdt', 'm w.e./year', 'cumulative and with seasonality removed', 'runoff_dhdt_gridfit_'+period_start + '_' + period_end + '.nc', out_path)

snowmelt_fit, snowmelt_fiterror = analise.fit_grid(snowmelt_noseason)
save.fit(snowmelt_fit, snowmelt_fiterror, snowmeltgridx, snowmeltgridy, 'snowmelt_dhdt', 'm w.e./year', 'cumulative and with seasonality removed', 'snowmelt_dhdt_gridfit_'+period_start + '_' + period_end + '.nc', out_path)

snowfall_fit, snowfall_fiterror = analise.fit_grid(snowfall_noseason)
save.fit(snowfall_fit, snowfall_fiterror, snowfallgridx, snowfallgridy, 'snowfall_dhdt', 'm w.e./year', 'cumulative and with seasonality removed', 'snowfall_dhdt_gridfit_'+period_start + '_' + period_end + '.nc', out_path)


#####################################################
"""density model, save"""

density_model = np.nanmean(rho_noseason, axis = 0)
densmodel_error = np.nanstd(rho_noseason, axis = 0)

save.fit(density_model, densmodel_error, rhogridx, rhogridy, 'rho_model', 'kgm3', 'cumulative and with seasonality removed', 'rho_model_'+period_start + '_' + period_end + '.nc', out_path)


density2_model = np.nanmean(rho2_noseason, axis = 0)
densmodel2_error = np.nanstd(rho2_noseason, axis = 0)

save.fit(density2_model, densmodel2_error, rho2gridx, rho2gridy, 'rho32cm_model', 'kgm3', 'cumulative and with seasonality removed', 'rho32cm_model_'+period_start + '_' + period_end + '.nc', out_path)


densitysurface_model = np.nanmean(rhos_noseason, axis = 0)
densmodelsurface_error = np.nanstd(rhos_noseason, axis = 0)

save.fit(densitysurface_model, densmodelsurface_error, rhosgridx, rhosgridy, 'rhosurf_model', 'kgm3', 'cumulative and with seasonality removed', 'rhosurf_model_'+period_start + '_' + period_end + '.nc', out_path)


"""compaction model, save"""

compaction_model = np.nanmean(compaction_noseason, axis = 0)
compmodel_error = np.nanstd(compaction_noseason, axis = 0)

save.fit(compaction_model, compmodel_error, compactiongridx, compactiongridy, 'compaction_model', 'm/year', 'cumulative and with seasonality removed', 'compaction_model_'+period_start + '_' + period_end + '.nc', out_path)

###############################################
"""ablation and accumulation masks, save"""
smb_mean = np.nanmean(smb, axis = 0)/1000

ablation_only, ablation_accumulation = analise.ablation_mask(smb_mean)
save.mask(ablation_only, ablation_accumulation, smbgridx, smbgridy, 'ablation_only', 'ablation_accumulation', 'ablation_mask_'+period_start + '_' + period_end + '.nc', out_path)

accumulation_only, accumulation_ablation = analise.accumulation_mask(smb_mean)
save.mask(accumulation_only, accumulation_ablation, smbgridx, smbgridy, 'accumulation_only', 'accumulation_ablation', 'accumulation_mask_'+period_start + '_' + period_end + '.nc', out_path)

runoff_mean = np.nanmean(runoff, axis = 0)/1000
snowmelt_mean = np.nanmean(snowmelt, axis = 0)/1000

"""facies mask"""
smb_m = smb/1000 #converting from mm w.e. to m w.e.
runoff_m = runoff/1000 #converting from mm w.e. to m w.e.
snowmelt_m = snowmelt/1000 #converting from mm w.e. to m w.e.


smb_p = analise.annual_positives(smbtime, smb_m)
snowmelt_p = analise.annual_positives(snowmelttime, snowmelt_m)
runoff_p = analise.annual_positives(smbtime, runoff_m) 


facies_mask = analise.facies_mask(smb_p, snowmelt_p, runoff_p)

dynamic_path = '/Users/kat/DATA/Firn_data/Slater_dynamics_mask_onCS2grid.nc'
dynamicgridx, dynamicgridy, dynamic_mask = load.load_mask(dynamic_path, 'vdmask_final')
facies_mask[dynamic_mask==1] = 2

period = 'Jan 2011 \n - Dec 2017' #this is for plotting
figpath = '/Users/kat/DATA/Firn_plotsandresults_2011_2017/'
plot.faciesmap(csgridx, csgridy, facies_mask, '', period, 1, 5, 'facies_map', figpath)

facies_mask[np.isnan(facies_mask)]=0

save.facies_mask(facies_mask, smbgridx, smbgridy, 'facies_mask', 'facies_mask_'+period_start + '_' + period_end + '.nc', out_path)

####
accumulation_smb = smb_mean.copy()
accumulation_smb[accumulation_ablation==0] = np.nan

asmb_max = np.nanmax(accumulation_smb) #max = 63.2 cm
asmb_mean = np.nanmean(accumulation_smb) #mean = 3.09 cm
asmb_std = np.nanstd(accumulation_smb) #std = 3.21 cm
asmb_median = np.nanmedian(accumulation_smb) #median = 2.42 cm

average_top = asmb_mean + 3*asmb_std * 3.21 #1.087241 # 3std = 12.71 cm, 2std = 9.5 cm and multiplied by 1.087241 to convert from water to snow

#################################################
"""Data errors"""
#CS2 error band and grid
from firnpy import load, analise #, plot
import numpy as np

cs2_path = '/Users/kat/DATA/Firn_data/dh_ts_GRL_2011_2018_wide.nc'

csgridx, csgridy, cstime, cs, cserr, csprojection = load.CS2(period_start, period_end, cs2_path)

mask_path = '/Users/kat/DATA/Firn_data/accumulation_mask_' + period_start + '_' + period_end + '.nc'
maskgridx, maskgridy, mask = load.mask(mask_path, 'accumulation_only')

#CS2 error
cserror_grids, cserror_timeseries = analise.error_CS2(cserr, 3)


#FDM error band and grid
fdmrefgridx, fdmrefgridy, fdmreftime, fdmref, fdmrefprojection = load.FDM_zs('195801', '198401', fdm_path)

# calculate the stds of the ref periods and the one maximum error band from that
fdmerror_grids, fdmerror_timeseries = analise.error_FDM(fdm_noseason, fdmref)

"""save errors"""
save.error(cserror_grids, cserror_timeseries, csgridx, csgridy, cstime, 'cserror_grids', 'cserror_series', 'm', 'measurement/interpolation error correlated at 3km.', 'cs_error_correlated_at3km_'+period_start + '_' + period_end + '.nc', out_path)

save.error(fdmerror_grids, fdmerror_timeseries, fdmgridx, fdmgridy, fdmtime, 'fdmerror_grids', 'fdmerror_series', 'm', 'computed by shifting the 20 year reference period between 1958 and 1984, one standard deviation of the resulting six slope values per time step is used.', 'fdm_error_'+period_start + '_' + period_end + '.nc', out_path)

##############################################
"N1 basin from Kappelsberer"

bpath = '/Users/kat/DATA/Firn_data/basins_mask.nc'

n1_mask, bx, by = analise.basin_n1(bpath)

save.basin_mask(n1_mask, bx, by, 'n1_mask', 'n1basin_mask.nc', out_path)


##############################################
"Basin mask following GRACE basins from Dresden product"

grace_mask, bx, by = analise.basins_gracedresden(bpath)        

save.basin_mask(grace_mask, bx, by, 'grace_mask', 'gracebasin_mask.nc', out_path)
