import sys
sys.path.insert(0, '/gws/nopw/j04/ceh_generic/matbro/ecocrop')
from ecocrop_utils import *
import pandas   as pd
import xarray   as xr
import numpy    as np
import dask     as da
import cftime   as cf
import datetime as dt
import subprocess as subp
import os

ecocroploc = '/gws/nopw/j04/ceh_generic/matbro/ecocrop/EcoCrop_DB_secondtrim.csv'
taspath = '/work/scratch-nopw/mattjbr/ecocrop/metdata/ukcp18_rcp85_land-rcm_uk_1km_01_v20190731_tas_bias_corrected_????.nc'
prepath = '/work/scratch-nopw/mattjbr/ecocrop/metdata/ukcp18_rcp85_land-rcm_uk_1km_01_v20190731_pr_bias_corrected_????.nc'
tmnpath = '/work/scratch-nopw/mattjbr/ecocrop/metdata/ukcp18_rcp85_land-rcm_uk_1km_01_v20190731_tasmin_bias_corrected_????.nc'
tmxpath = '/work/scratch-nopw/mattjbr/ecocrop/metdata/ukcp18_rcp85_land-rcm_uk_1km_01_v20190731_tasmax_bias_corrected_????.nc'
tasvname = 'tas'
prevname = 'pr'
tmnvname = 'tasmin'
tmxvname = 'tasmax'
yearaggmethod = 'median'
predir = '/gws/nopw/j04/ceh_generic/matbro/ecocrop/precalcs/precalcs'
savedir = '/gws/nopw/j04/ceh_generic/matbro/ecocrop/scores_agg_tests' + '/' + yearaggmethod
lcmloc = '/gws/nopw/j04/ceh_generic/matbro/ecocrop/LCM15_Arable_Mask.tif'
bgsloc = '/gws/nopw/j04/ceh_generic/matbro/ecocrop/BGS_soildata/masks'
plotdir = '/gws/nopw/j04/ceh_generic/matbro/ecocrop/plots_agg_tests' + '/' + yearaggmethod
precmethod = int(sys.argv[2]) # which precip score method to use (see utils.py for funcs)

ecocropall = pd.read_csv(ecocroploc, engine='python')
ecocrop = ecocropall.drop(['level_0'], axis=1)
cropind = int(sys.argv[1])
print('Cropind: ' + str(cropind))
testcrop = ecocrop.iloc[cropind,:] # 19 onions, #117 wheat, #147 chickpea, #66 sweet potato
TOPMIN = testcrop['TOPMN'] + 273.15 # C-->K
TOPMAX = testcrop['TOPMX'] + 273.15 # C-->K
PMIN = testcrop['RMIN']/86400. # mm-->kg/m^2/s
PMAX = testcrop['RMAX']/86400. # mm-->kg/m^2/s
POPMIN = testcrop['ROPMN']/86400. # mm-->kg/m^2/s
POPMAX = testcrop['ROPMX']/86400. # mm-->kg/m^2/s
KTMP = testcrop['KTMPR'] + 273.15 # C-->K
KMAX = testcrop['TMAX'] + 273.15  # C-->K
GMIN = testcrop['GMIN']
GMAX = testcrop['GMAX']
SOIL = testcrop['TEXT']
COMNAME = testcrop['COMNAME']
COMNAME = '_'.join(COMNAME.split(',')[0].split(' '))
if '(' in COMNAME:
    COMNAME = ''.join(COMNAME.split('('))
    COMNAME = ''.join(COMNAME.split(')'))
if "'" in COMNAME:
    COMNAME = ''.join(COMNAME.split("'"))
cropname = COMNAME

# Check for missing data
if np.isnan(testcrop['TOPMN']):
    raise ValueError('Missing TOPMN')
if np.isnan(testcrop['TOPMX']):
    raise ValueError('Missing TOPMX')
if np.isnan(testcrop['TMAX']):
    raise ValueError('Missing TMAX (KMAX)')
if np.isnan(testcrop['RMIN']):
    raise ValueError('Missing RMIN')
if np.isnan(testcrop['RMAX']):
    raise ValueError('Missing RMAX')
if np.isnan(testcrop['ROPMN']):
    raise ValueError('Missing ROPMN')
if np.isnan(testcrop['ROPMX']):
    raise ValueError('Missing ROPMX')
if np.isnan(testcrop['GMIN']):
    raise ValueError('Missing GMIN')
if np.isnan(testcrop['GMAX']):
    raise ValueError('Missing GMAX')

# exit if GMIN=GMAX, assume missing data
if GMAX-GMIN<=10:
    raise ValueError('GMIN and GMAX too close, not enough info to calculate suitability')

# assume killing temp of -1 if not specified
if np.isnan(KTMP):
    KTMP=-1

GMIN = int(GMIN)
GMAX = int(GMAX)
print('TOPMN: ' + str(testcrop['TOPMN']))
print('TOPMX: ' + str(testcrop['TOPMX']))
print('KTMP: ' + str(testcrop['KTMPR']))
print('KMAX: ' + str(testcrop['TMAX']))
print('GMIN: ' + str(testcrop['GMIN']))
print('GMAX: ' + str(testcrop['GMAX']))
print('PMIN: ' + str(testcrop['RMIN']))
print('PMAX: ' + str(testcrop['RMAX']))
print('POPMN: ' + str(testcrop['ROPMN']))
print('POPMX: ' + str(testcrop['ROPMX']))
print('SOIL: ' + str(SOIL))
sys.stdout.flush()

########## STAGE 1 & 2 ############
#
# For the KTMP, KMAX and PREC calcs, we can use the pre-calculated
# cumulative sums that are saved on disk
#
# For the TEMP calculation we have to read in the metdata from scratch
# as there are too many permutations to pre-calc it beforehand

if not os.path.exists(savedir):
    os.makedirs(savedir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# open datafiles
print('Reading in met data')
print('Start: ' + str(dt.datetime.now()))
sys.stdout.flush()
tas = xr.open_mfdataset(taspath).astype('float16')[tasvname]
tmn = xr.open_mfdataset(tmnpath).astype('float16')[tmnvname]
tmx = xr.open_mfdataset(tmxpath).astype('float16')[tmxvname]
pre = xr.open_mfdataset(prepath)[prevname]
tas = tas.sel(time=slice('2020-01-01', tas['time'][-1])).load()
tmn = tmn.sel(time=slice('2020-01-01', tmn['time'][-1])).load()
tmx = tmx.sel(time=slice('2020-01-01', tmx['time'][-1])).load()
pre = pre.sel(time=slice('2020-01-01', pre['time'][-1])).load()
print('End: ' + str(dt.datetime.now()))

print('Calculating topt_, ktmp_ and kmax_crop')
print('Start: ' + str(dt.datetime.now()))
sys.stdout.flush()
topt_crop = xr.where(xr.ufuncs.logical_and(tas < TOPMAX, tas > TOPMIN), 1, 0).astype('uint16').values
ktmp_crop = xr.where(tmn < KTMP, 1, 0).astype('uint16').values
kmax_crop = xr.where(tmx > KMAX, 1, 0).astype('uint16').values
print('End: ' + str(dt.datetime.now()))
sys.stdout.flush()

if GMAX-GMIN<=15:
    gstart = np.int16(np.floor(GMIN/10)*10)
else:
    gstart = np.int16(np.ceil(GMIN/10)*10)
gend   = np.int16(np.ceil(GMAX/10)*10)
allgtimes = list(np.arange(gstart, gend, 10, dtype='int16'))


# First check if any files are missing or corrupted and skip those gtimes
#allfilescounter=0
#missingfilescounter=0
#missingfiles = []
#missinggtimes = []
#print('Checking availability of files')
#for gtime in allgtimes:
#
#    kmaxfname = 'KMAX_' + str(int(testcrop['TMAX'])) + \
#                '_gtime_' + str(int(gtime)) + 'v2.nc'
#    print('Checking ' + kmaxfname)
#    kmaxpath = os.path.join(predir, kmaxfname)
#    if not os.path.exists(kmaxpath):
#        print(kmaxfname + ' does not exist')
#        missingfilescounter+=1    
#        missingfiles.append(kmaxfname)
#        missinggtimes.append(gtime)
#    elif np.all(xr.open_dataarray(kmaxpath)[-1,:,:].values==65535):
#        print(kmaxfname + ' file is corrupted')
#        missingfilescounter+=1
#        missingfiles.append(kmaxfname)
#        missinggtimes.append(gtime)
#    elif np.all(xr.open_dataarray(kmaxpath)[-1,:,:].values==0):
#        print(kmaxfname + ' file is corrupted')
#        missingfilescounter+=1
#        missingfiles.append(kmaxfname)
#        missinggtimes.append(gtime)
#    allfilescounter+=1
#
#    ktmpfname = 'KTMP_' + str(int(testcrop['KTMPR'])) + \
#                '_gtime_' + str(int(gtime)) + 'v2.nc'
#    print('Checking ' + ktmpfname)
#    ktmppath = os.path.join(predir, ktmpfname)
#    if not os.path.exists(ktmppath):
#        print(ktmpfname + ' does not exist')
#        missingfilescounter+=1    
#        missingfiles.append(ktmpfname)
#        missinggtimes.append(gtime)
#    elif np.all(xr.open_dataarray(ktmppath)[-1,:,:].values==65535):
#        print(ktmpfname + ' file is corrupted')
#        missingfilescounter+=1
#        missingfiles.append(ktmpfname)  
#        missinggtimes.append(gtime)
#    elif np.all(xr.open_dataarray(ktmppath)[-1,:,:].values==0):
#        print(ktmpfname + ' file is corrupted')
#        missingfiles.append(ktmpfname)   
#        missinggtimes.append(gtime)
#         missingfilescounter+=1
#    allfilescounter+=1
#
#    precfname = 'PREC_' + \
#                'gtime_' + str(int(gtime)) + 'v2.nc'
#    precpath = os.path.join(predir, precfname)
#    if not os.path.exists(precpath):
#        print(precfname + ' does not exist')
#        missingfilescounter+=1        
#    allfilescounter+=1
#
#
#missinggtimes = list(np.unique(np.asarray(missinggtimes)))
#print(str(missingfilescounter) + ' of ' + str(allfilescounter) + ' missing or corrupted: ')
#print(missingfiles)
#print('\n')
#print('The following gtimes have missing or corrupted files: ')
#print(missinggtimes)
#
#for mgt in missinggtimes:
#    allgtimes.remove(mgt)

# create arrays to store the total proportion of ktmp/kmax days amassed over all the gtimes
# for later calculating the average
print('Creating ktmp_days_prop and kmax_days_prop arrays')
print('Start: ' + str(dt.datetime.now()))
ktmp_days_prop_total = ktmp_crop.copy()[:-allgtimes[-1]+1].astype('float32')
kmax_days_prop_total = kmax_crop.copy()[:-allgtimes[-1]+1].astype('float32')
ktmp_days_prop_total[:] = 0
kmax_days_prop_total[:] = 0
kdptlen = ktmp_days_prop_total.shape[0]
print('End: ' + str(dt.datetime.now()))
sys.stdout.flush()

counter=1
GMIN = np.uint16(GMIN)
GMAX = np.uint16(GMAX)
for gtime in allgtimes:
    print('Calculating suitability for ' + cropname + \
          ' for a growing season of length ' + str(gtime) + \
          ' out of a maximum of ' + str(int(GMAX)))
    print('Start: ' + str(dt.datetime.now()))
    
    print('Calculating ndays of T in optimal range within gtime')
    sys.stdout.flush()
    # calculate ndays of T in optimal range within gtime
    tcoords_tas = tas['time'][:-gtime+1]
    ycoords_tas = tas['y']
    xcoords_tas = tas['x']
    toptdays = frs3D(topt_crop, gtime, 'uint16')
    toptdays = xr.DataArray(toptdays, coords=[tcoords_tas, ycoords_tas, xcoords_tas])
    toptdays.name = 'TOPT_days'
    print('End: ' + str(dt.datetime.now())) 

    # calculate whether any of the suitable days/locations identified above will have
    # frost/killing temp within gtime
    print('Calculating frost days and their proportions')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    tcoords_tmn = tmn['time'][:-gtime+1]
    ycoords_tmn = tmn['y']
    xcoords_tmn = tmn['x']
    ktmp_days = frs3D(ktmp_crop, gtime, 'uint16')
    ktmp_days_prop_total += (ktmp_days/gtime)[:kdptlen] 
    ktmp_days = xr.DataArray(ktmp_days, coords=[tcoords_tmn, ycoords_tmn, xcoords_tmn])
    ktmp_days.name = 'KTMP_days'
    print(ktmp_days_prop_total.dtype)
    print('End: ' + str(dt.datetime.now())) 

    # calculate whether any of the suitable days/locations identified above will have
    # heat killing temp within gtime
    print('Calculating heat-stress days and their proportions')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    tcoords_tmx = tmx['time'][:-gtime+1]
    ycoords_tmx = tmx['y']
    xcoords_tmx = tmx['x']
    kmax_days = frs3D(kmax_crop, gtime, 'uint16')
    kmax_days_prop_total += (kmax_days/gtime)[:kdptlen]
    kmax_days = xr.DataArray(kmax_days, coords=[tcoords_tmx, ycoords_tmx, xcoords_tmx])
    kmax_days.name = 'KMAX_days'
    print('End: ' + str(dt.datetime.now())) 

    print('Calculating total precipitation')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    # calculate total precipitation in gtime
    tcoords_pre = pre['time'][:-gtime+1]
    ycoords_pre = pre['y']
    xcoords_pre = pre['x']
    pre2 = pre.values
    precip_crop = frs3D(pre2, gtime, 'float32')
    precip_crop = xr.DataArray(precip_crop, coords=[tcoords_pre, ycoords_pre, xcoords_pre])
    precip_crop.name = 'precip_total'
    print('End: ' + str(dt.datetime.now())) 

    ########### STAGE 3 #############

    print('Calculating where T suitable within GTIME')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    suit_crop = xr.where(toptdays >= GMIN, 1, 0).astype('uint8')
    print('End: ' + str(dt.datetime.now())) 

    # if they are different lengths it will likely be because the end of the data is missing
    # (due to a minor bug in the frs3D functions), so need to account for this
    #print('Calculating KMAX')
    #kmaxfname = 'KMAX_' + str(int(testcrop['TMAX'])) + \
    #            '_gtime_' + str(int(gtime)) + 'v2.nc'
    #kmaxpath = os.path.join(predir, kmaxfname)
    #kmax_days = xr.open_dataset(kmaxpath)['KMAXdays'].astype('uint16')
    #if len(kmax_days['time']) > len(suit_crop['time']):
    #    kmax_days = kmax_days.sel(time=slice(suit_crop['time'][0], suit_crop['time'][-1]))
    #elif len(kmax_days['time']) < len(suit_crop['time']):
    #    suit_crop = suit_crop.sel(time=slice(kmax_days['time'][0], kmax_days['time'][-1]))
    #suit_crop2 = xr.where(kmax_days > np.uint8(0), suit_crop - np.uint8(kmax_days), suit_crop)

    print('Masking based on KTMP')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    #ktmpfname = 'KTMP_' + str(int(testcrop['KTMPR'])) + \
    #            '_gtime_' + str(int(gtime)) + 'v2.nc'
    #ktmppath = os.path.join(predir, ktmpfname)
    #ktmp_days = xr.open_dataset(ktmppath)['KTMPdays'].astype('uint16')
    #if len(ktmp_days['time']) > len(suit_crop2['time']):
    #    ktmp_days = ktmp_days.sel(time=slice(suit_crop2['time'][0], suit_crop2['time'][-1]))
    #elif len(ktmp_days['time']) < len(suit_crop2['time']):
    #    suit_crop2 = suit_crop2.sel(time=slice(ktmp_days['time'][0], ktmp_days['time'][-1]))
    suit_crop2 = xr.where(ktmp_days > np.uint8(0), np.uint8(0), suit_crop)
    print('End: ' + str(dt.datetime.now())) 

    print('Calculating T suitability score and KMAX days penalty')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    tscore = score_temp(gtime, GMIN, GMAX)
    tempscore1 = xr.where(suit_crop2 > np.uint8(0), tscore, np.uint8(0)).astype('int8')
    kmax_days = xr.where(suit_crop2 > np.uint8(0), kmax_days, np.uint(0))
    tempscore = tempscore1 - np.int8(kmax_days)
    tempscore = xr.where(tempscore < 0, 0, tempscore).astype('uint8')
    print('End: ' + str(dt.datetime.now())) 

    #precfname = 'PREC_' + \
    #            'gtime_' + str(int(gtime)) + 'v2.nc'
    #precpath = os.path.join(predir, precfname)
    #print('Loading prec file')
    #precip_crop = xr.open_dataset(precpath)['PRECtotal'].astype('float16')
    print('Calculating precip suitability score using method ' + str(precmethod))
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    if precmethod==1:
        precscore = score_prec1(precip_crop, PMIN, PMAX, POPMIN, POPMAX)
    elif precmethod==2:
        precscore = score_prec2(precip_crop, PMIN, PMAX, POPMIN, POPMAX)
    elif precmethod==3:
        precscore = score_prec3(precip_crop, PMIN, PMAX, POPMIN, POPMAX)
    else:
        raise ValueError('precmethod must be 1, 2 or 3. Currently set as ' + str(precmethod))
    print('End: ' + str(dt.datetime.now())) 

    print('Updating T & P suitability scores for this gtime')
    print('Start: ' + str(dt.datetime.now()))
    sys.stdout.flush()
    if counter==1:
        tempscore_old = tempscore
        precscore_old = precscore
    else:
        if len(tempscore_old['time']) > len(tempscore['time']):
            tempscore_old = tempscore_old.sel(time=slice(tempscore['time'][0], tempscore['time'][-1]))
        if len(precscore_old['time']) > len(precscore['time']):
            precscore_old = precscore_old.sel(time=slice(precscore['time'][0], precscore['time'][-1]))
        tempscore = xr.where(tempscore > tempscore_old, tempscore, tempscore_old)#.astype('uint8')
        precscore = xr.where(precscore > precscore_old, precscore, precscore_old)#.astype('uint8')
        tempscore_old = tempscore
        precscore_old = precscore

        del suit_crop
        del suit_crop2
        print('End: ' + str(dt.datetime.now())) 
        sys.stdout.flush()
    counter+=1

print('Calculating final combined crop suitability score')
print('Start: ' + str(dt.datetime.now()))
sys.stdout.flush()
final_score_crop = xr.where(precscore < tempscore, precscore, tempscore)
print(final_score_crop.dtype)
print('End: ' + str(dt.datetime.now())) 

print('Calculating average ktmp_ and kmax_days proportions')
print('Start: ' + str(dt.datetime.now()))
sys.stdout.flush()
ktmp_days_avg_prop = ktmp_days_prop_total/len(allgtimes)
kmax_days_avg_prop = kmax_days_prop_total/len(allgtimes)
tcoords_k = tmn['time'][:-allgtimes[-1]+1]
ktmp_days_avg_prop = xr.DataArray(ktmp_days_avg_prop, coords=[tcoords_k, ycoords_tmx, xcoords_tmx])
kmax_days_avg_prop = xr.DataArray(kmax_days_avg_prop, coords=[tcoords_k, ycoords_tmx, xcoords_tmx])
print(ktmp_days_avg_prop)
print(ktmp_days_avg_prop.dtype)
print('End: ' + str(dt.datetime.now())) 

print('Saving to netcdf')
print('Start: ' + str(dt.datetime.now()))
sys.stdout.flush()
# save to netcdf
final_score_crop.name = 'crop_suitability_score'
final_score_crop.encoding['zlib'] = True
final_score_crop.encoding['complevel'] = 1
final_score_crop.encoding['shuffle'] = False
final_score_crop.encoding['contiguous'] = False
final_score_crop.encoding['dtype'] = np.dtype('uint8')
encoding = {}
encoding['crop_suitability_score'] = final_score_crop.encoding
final_score_crop.to_netcdf(os.path.join(savedir, cropname + '.nc'), encoding=encoding)

tempscore.name = 'temperature_suitability_score'
tempscore.encoding['zlib'] = True
tempscore.encoding['complevel'] = 1
tempscore.encoding['shuffle'] = False
tempscore.encoding['contiguous'] = False
tempscore.encoding['dtype'] = np.dtype('uint8')
encoding = {}
encoding['temperature_suitability_score'] = tempscore.encoding
tempscore.to_netcdf(os.path.join(savedir, cropname + '_temp.nc'), encoding=encoding)

precscore.name = 'precip_suitability_score'
precscore.encoding['zlib'] = True
precscore.encoding['complevel'] = 1
precscore.encoding['shuffle'] = False
precscore.encoding['contiguous'] = False
precscore.encoding['dtype'] = np.dtype('uint8')
encoding = {}
encoding['precip_suitability_score'] = precscore.encoding
precscore.to_netcdf(os.path.join(savedir, cropname + '_prec.nc'), encoding=encoding)

ktmp_days_avg_prop.name = 'average_proportion_of_ktmp_days_in_gtime'
ktmp_days_avg_prop.encoding['zlib'] = True
ktmp_days_avg_prop.encoding['complevel'] = 1
ktmp_days_avg_prop.encoding['shuffle'] = False
ktmp_days_avg_prop.encoding['contiguous'] = False
ktmp_days_avg_prop.encoding['dtype'] = np.dtype('float32')
encoding = {}
encoding['average_proportion_of_ktmp_days_in_gtime'] = ktmp_days_avg_prop.encoding
ktmp_days_avg_prop.to_netcdf(os.path.join(savedir, cropname + '_ktmp_days_avg_prop.nc'), encoding=encoding)

kmax_days_avg_prop.name = 'average_proportion_of_kmax_days_in_gtime'
kmax_days_avg_prop.encoding['zlib'] = True
kmax_days_avg_prop.encoding['complevel'] = 1
kmax_days_avg_prop.encoding['shuffle'] = False
kmax_days_avg_prop.encoding['contiguous'] = False
kmax_days_avg_prop.encoding['dtype'] = np.dtype('float32')
encoding = {}
encoding['average_proportion_of_kmax_days_in_gtime'] = kmax_days_avg_prop.encoding
kmax_days_avg_prop.to_netcdf(os.path.join(savedir, cropname + '_kmax_days_avg_prop.nc'), encoding=encoding)
print('End: ' + str(dt.datetime.now())) 


# calculate and plot monthly climos of ktmp & kmax days avg prop for each decade and their differences
print('Calculating monthly climo of ktmp/kmax proportions and decadal changes')
sys.stdout.flush()
ktmpap_monavg_climo_diffs, kmaxap_monavg_climo_diffs = \
calc_decadal_kprop_changes(ktmp_days_avg_prop, kmax_days_avg_prop, str(SOIL), lcmloc, bgsloc, cropname, savedir)
for month in range(1, 13):
    plot_decadal_changes(kmaxam_monavg_climo_diffs.sel(month=month), 
                         save=os.path.join(plotdir, cropname + '_kmaxdaysprop_decadal_change_month' + str(month) + '.png'),
                         revcolbar = 1)
for month in range(1, 13):
    plot_decadal_changes(ktmpam_monavg_climo_diffs.sel(month=month), 
                         save=os.path.join(plotdir, cropname + '_ktmpdaysprop_decadal_change_month' + str(month) + '.png'),
                         revcolbar = 1)

# calculate day of year of maximum score
print('Finding days of years of the maximum score')
sys.stdout.flush()
maxdoys, maxdoys_temp, maxdoys_prec = calculate_max_doy(final_score_crop, tempscore, precscore)
print('Calculating yearly average of this and decadal changes using modulo arithmetic/circular averaging')
sys.stdout.flush()
maxdoys_decadal_changes, maxdoys_temp_decadal_changes, maxdoys_prec_decadal_changes = \
calc_decadal_doy_changes(maxdoys, maxdoys_temp, maxdoys_prec, str(SOIL), lcmloc, bgsloc, cropname, savedir)
plot_decadal_changes(maxdoys_decadal_changes, save=os.path.join(plotdir, cropname + '_maxdoys_decadal_changes.png'))
plot_decadal_changes(maxdoys_temp_decadal_changes, save=os.path.join(plotdir, cropname + '_maxdoys_temp_decadal_changes.png'))
plot_decadal_changes(maxdoys_prec_decadal_changes, save=os.path.join(plotdir, cropname + '_maxdoys_prec_decadal_changes.png'))

# calculate yearly scores and decadal changes
print('Calculating yearly scores and decadal changes')
sys.stdout.flush()
allscore_decades, tempscore_decades, precscore_decades, allscore_decadal_changes, tempscore_decadal_changes, precscore_decadal_changes = \
calc_decadal_changes(tempscore, precscore, str(SOIL), lcmloc, bgsloc, cropname, savedir, yearaggmethod)
plot_decadal_changes(allscore_decadal_changes, save=os.path.join(plotdir, cropname + '_decadal_changes.png'))
plot_decadal_changes(tempscore_decadal_changes, save=os.path.join(plotdir, cropname + '_tempscore_decadal_changes.png'))
plot_decadal_changes(precscore_decadal_changes, save=os.path.join(plotdir, cropname + '_precscore_decadal_changes.png'))
plot_decade(allscore_decades[0,:,:], tempscore_decades[0,:,:], precscore_decades[0,:,:], save=os.path.join(plotdir, cropname + '_current_decade.png'))
