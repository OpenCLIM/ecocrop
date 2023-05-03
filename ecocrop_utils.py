import os
import xarray as xr
import numpy  as np
import cartopy as cp
import netCDF4 as nc4
import rioxarray as rioxr
import matplotlib.pyplot as plt

def circular_avg(maxdoys, dim):
    '''
    Function to calculate the average on a circular domain 0-360, 
    e.g. for calculating the average day of year. Process is:
    - Convert to radians
    - Calculate the sine of a given year's 'angles' and store it
    - Calculate the cosine of the same year's 'angles' and store it
    - Repeat for all the years in a given average
    - Sum up the sines and cosines separately
    - Calculate the arctan of these element-wise
    - Convert back to 'degrees' (i.e. days of year)
    as defined at https://en.wikipedia.org/wiki/Circular_mean
    '''
    maxdoys_rad = np.deg2rad(maxdoys)

    maxdoys_sin = np.sin(maxdoys_rad)
    maxdoys_cos = np.cos(maxdoys_rad)

    maxdoys_sinsum = maxdoys_sin.sum(dim)/len(maxdoys[dim])
    maxdoys_cossum = maxdoys_cos.sum(dim)/len(maxdoys[dim])

    maxdoys_radavg = xr.where(xr.ufuncs.logical_and(maxdoys_sinsum>0, maxdoys_cossum>0), 
                              xr.ufuncs.arctan(maxdoys_sinsum/maxdoys_cossum),
                              xr.where(xr.ufuncs.logical_and(maxdoys_sinsum<0, maxdoys_cossum>0),
                              xr.ufuncs.arctan(maxdoys_sinsum/maxdoys_cossum)+(2*np.pi),
                              xr.where(maxdoys_cossum<0,
                              xr.ufuncs.arctan(maxdoys_sinsum/maxdoys_cossum)+np.pi,
                              0)))

    maxdoys_avg = np.rad2deg(maxdoys_radavg)
    
    return maxdoys_avg

# Mask out non-growing regions using a version of the land-
# cover map supplied by John Redhead
def lcm_mask(lcm, data):
    if type(lcm)==str:
        if lcm[-3:]=='tif':
            lcm = xr.open_dataset(lcm, engine='rasterio')
            lcm = lcm['band_data']
            lcm = lcm.drop('band').squeeze()
        else:
            lcm = xr.open_dataarray(lcm)
    
    dataxlims = [data['x'].values[0], data['x'].values[-1]]
    dataylims = [data['y'].values[0], data['y'].values[-1]]
    lcmylims = [lcm['y'].values[0], lcm['y'].values[-1]]
    if dataylims[0] < dataylims[1]:
        if lcmylims[0] > lcmylims[1]:
            lcm=lcm[::-1,:]
    elif dataylims[0] > dataylims[1]:
        if lcmylims[0] < lcmylims[1]:
            lcm=lcm[::-1,:]
    
    lcm_cropped = lcm.sel(x=slice(dataxlims[0], dataxlims[1]), 
                          y=slice(dataylims[0], dataylims[1]))
    
    data_masked_npy = np.where(lcm_cropped.values>0, data.values, 0)
    data_masked = data.copy()
    data_masked.values = data_masked_npy
    return data_masked

def lcm_mask_xr(lcm, data):
    if type(lcm)==str:
        if lcm[-3:]=='tif':
            lcm = xr.open_dataset(lcm, engine="rasterio")
            lcm = lcm['band_data']
            lcm = lcm.drop('band').squeeze()
        else:
            lcm = xr.open_dataarray(lcm)
    
    dataxlims = [data['x'].values[0], data['x'].values[-1]]
    dataylims = [data['y'].values[0], data['y'].values[-1]]
    if dataylims[0] < dataylims[1]:
        lcm=lcm[::-1,:]
    
    lcm_cropped = lcm.sel(x=slice(dataxlims[0], dataxlims[1]), 
                          y=slice(dataylims[0], dataylims[1]))
    
    data_masked = xr.where(lcm_cropped>0, data, 0)
    return data_masked

# Mask based on soil type, using a soil type mask in netcdf format
def soil_type_mask(mask, data):
    dataxlims = [data['x'].values[0], data['x'].values[-1]]
    dataylims = [data['y'].values[0], data['y'].values[-1]]
    
    if type(mask)==str:
        mask = xr.open_dataarray(mask)
        # this dtype conversion is to get around the case where the coordinates might
        # be in different dtypes to the datalims, sometimes causing an error
        xdtype = mask.x.dtype
        ydtype = mask.y.dtype
    mask_cropped = mask.sel(x=slice(dataxlims[0].astype(xdtype), dataxlims[1].astype(xdtype)), 
                            y=slice(dataylims[0].astype(ydtype), dataylims[1].astype(ydtype)))
    
    data_masked_npy = np.where(mask_cropped.values>0, data.values, 0)
    data_masked = data.copy()
    data_masked.values = data_masked_npy
    return data_masked


def soil_type_mask_all(data, SOIL, maskloc):
    # apply the masking function for all the soil types
    # dependent on which the crop grows in (SOIL)
    if 'heavy' in SOIL and 'medium' in SOIL and 'light' in SOIL:
        print('Doing masking for all soil groups')
        maskfile = os.path.join(maskloc, 'all_soil_mask.nc')
        data = soil_type_mask(maskfile, data)
    elif 'heavy' in SOIL and 'medium' in SOIL:
        print('Doing masking for heavy and medium soil groups')
        maskfile = os.path.join(maskloc, 'heavy_med_soil_mask.nc')
        data = soil_type_mask(maskfile, data)
    elif 'heavy' in SOIL and 'light' in SOIL:
        print('Doing masking for light and heavy soil groups')
        maskfile = os.path.join(maskloc, 'heavy_light_soil_mask.nc')
        data = soil_type_mask(maskfile, data)
    elif 'medium' in SOIL and 'light' in SOIL:
        print('Doing masking for light and medium soil groups')
        maskfile = os.path.join(maskloc, 'med_light_soil_mask.nc')
        data = soil_type_mask(maskfile, data)
    elif 'light' in SOIL:
        print('Doing masking for light soil group')
        maskfile = os.path.join(maskloc, 'light_soil_mask.nc')
        data = soil_type_mask(maskfile, data)
    elif 'medium' in SOIL:
        print('Doing masking for medium soil group')
        maskfile = os.path.join(maskloc, 'medium_soil_mask.nc')
        data = soil_type_mask(maskfile, data)
    elif 'heavy' in SOIL:
        print('Doing masking for heavy soil group')
        maskfile = os.path.join(maskloc, 'heavy_soil_mask.nc')
        data = soil_type_mask(maskfile, data)

    return data

def calculate_max_doy(allscore, tempscore, precscore):
    '''
    Return the day of year of the maximum score for allscore, tempscore, precscore
    '''
    maxdoys = []
    for yr, yrdata in allscore.groupby('time.year'):
        print('Calculating doy of max score for year ' + str(yr))
        maxdoy = yrdata.idxmax('time').dt.dayofyear.expand_dims({'year': [yr]})
        maxdoy = maxdoy.where(maxdoy>1)
        maxdoys.append(maxdoy)
    maxdoys = maxdoys[:-1]
    maxdoys = xr.concat(maxdoys, dim='year')

    maxdoys_temp = []
    for yr, yrdata in tempscore.groupby('time.year'):
        print('Calculating doy of max temp score for year ' + str(yr))
        maxdoy = yrdata.idxmax('time').dt.dayofyear.expand_dims({'year': [yr]})
        maxdoy = maxdoy.where(maxdoy>1)
        maxdoys_temp.append(maxdoy)
    maxdoys_temp = maxdoys_temp[:-1]
    maxdoys_temp = xr.concat(maxdoys_temp, dim='year')

    maxdoys_prec = []
    for yr, yrdata in precscore.groupby('time.year'):
        print('Calculating doy of max prec score for year ' + str(yr))
        maxdoy = yrdata.idxmax('time').dt.dayofyear.expand_dims({'year': [yr]})
        maxdoy = maxdoy.where(maxdoy>1)
        maxdoys_prec.append(maxdoy)
    maxdoys_prec = maxdoys_prec[:-1]
    maxdoys_prec = xr.concat(maxdoys_prec, dim='year')
    
    return maxdoys, maxdoys_temp, maxdoys_prec


def calc_decadal_changes(tempscore, precscore, SOIL, LCMloc, sgmloc, cropname, outdir, yearaggmethod):
    '''
    Calculate decadal changes of crop suitability scores from the 
    daily crop suitability scores. 
    tempscore, precscore inputs either netcdf filenames
    or xarray dataarrays. Both must have variables 
    'temperature_suitability_score' and
    'precip_suitability_score', respectively
    
    SOIL: Soil group suitability string from the ecocrop database
    LCMloc: Land cover mask. Path to tif
    sgmloc: Soil group mask netcdfs folder as string
    outdir: Where to store output netcdf files
    cropname: For output filenames
    '''
    
    print('Calculating yearly score')
    # crop suitability score for a given year is the max
    # over all days in the year
    if yearaggmethod == 'max':
        tempscore_years = tempscore.groupby('time.year').max()
        precscore_years = precscore.groupby('time.year').max()
    elif yearaggmethod == 'median':
        tempscore_years = tempscore.groupby('time.year').median()
        precscore_years = precscore.groupby('time.year').median()
    elif yearaggmethod == 'mean':
        tempscore_years = tempscore.groupby('time.year').mean()
        precscore_years = precscore.groupby('time.year').mean()
    elif yearaggmethod == 'percentile':
        tempscore_years = tempscore.groupby('time.year').quantile(0.95)
        precscore_years = precscore.groupby('time.year').quantile(0.95)
    else:
        raise SyntaxError("yearaggmethod must be one of max, median, mean or percentile")
    allscore_years = xr.where(precscore_years < tempscore_years, precscore_years, tempscore_years)

    print('Doing masking')
    # mask at this stage to avoid memory issues
    lcm = xr.open_dataset(LCMloc, engine="rasterio")
    lcm = lcm['band_data']
    lcm = lcm.drop('band').squeeze()
    lcm=lcm[::-1,:]
    allscore_years  = lcm_mask(lcm, allscore_years)
    tempscore_years = lcm_mask(lcm, tempscore_years)
    precscore_years = lcm_mask(lcm, precscore_years)
    allscore_years  = soil_type_mask_all(allscore_years,  SOIL, sgmloc)
    tempscore_years = soil_type_mask_all(tempscore_years, SOIL, sgmloc)
    precscore_years = soil_type_mask_all(precscore_years, SOIL, sgmloc)

    # compress and save to disk
    allscore_years.name = 'crop_suitability_score'
    allscore_years.encoding['zlib'] = True
    allscore_years.encoding['complevel'] = 1
    allscore_years.encoding['shuffle'] = False
    allscore_years.encoding['contiguous'] = False
    allscore_years.encoding['dtype'] = np.dtype('uint8')
    encoding = {}
    encoding['crop_suitability_score'] = allscore_years.encoding
    allscore_years.to_netcdf(os.path.join(outdir, cropname + '_years.nc'), encoding=encoding)

    tempscore_years.encoding['zlib'] = True
    tempscore_years.encoding['complevel'] = 1
    tempscore_years.encoding['shuffle'] = False
    tempscore_years.encoding['contiguous'] = False
    tempscore_years.encoding['dtype'] = np.dtype('uint8')
    encoding = {}
    encoding['temperature_suitability_score'] = tempscore_years.encoding
    tempscore_years.to_netcdf(os.path.join(outdir, cropname + '_tempscore_years.nc'), encoding=encoding)
    
    precscore_years.encoding['zlib'] = True
    precscore_years.encoding['complevel'] = 1
    precscore_years.encoding['shuffle'] = False
    precscore_years.encoding['contiguous'] = False
    precscore_years.encoding['dtype'] = np.dtype('uint8')
    encoding = {}
    encoding['precip_suitability_score'] = precscore_years.encoding
    precscore_years.to_netcdf(os.path.join(outdir, cropname + '_precscore_years.nc'), encoding=encoding)

    print('Calculating decadal score')
    # crop suitability score for a given decade is the mean
    # over all years in the decade
    allscore_decades = []
    tempscore_decades = []
    precscore_decades = []
    syear = allscore_years['year'][0].values
    nyears = allscore_years.shape[0]
    nyears_u = int(np.floor(nyears/10)*10)
    for idx in range(0, nyears_u, 10):
        allscore_decade  = allscore_years[idx:idx+10,:,:].mean(dim='year')
        tempscore_decade = tempscore_years[idx:idx+10,:,:].mean(dim='year')
        precscore_decade = precscore_years[idx:idx+10,:,:].mean(dim='year')
        
        allscore_decade  = allscore_decade.expand_dims({'decade': [syear+idx]})
        tempscore_decade = tempscore_decade.expand_dims({'decade': [syear+idx]})
        precscore_decade = precscore_decade.expand_dims({'decade': [syear+idx]})
        
        allscore_decades.append(allscore_decade)
        tempscore_decades.append(tempscore_decade)
        precscore_decades.append(precscore_decade)
    allscore_decades  = xr.merge(allscore_decades)['crop_suitability_score']
    tempscore_decades = xr.merge(tempscore_decades)['temperature_suitability_score']
    precscore_decades = xr.merge(precscore_decades)['precip_suitability_score']

    # compress and save to disk
    allscore_decades.name = 'crop_suitability_score'
    allscore_decades.encoding['zlib'] = True
    allscore_decades.encoding['complevel'] = 1
    allscore_decades.encoding['shuffle'] = False
    allscore_decades.encoding['contiguous'] = False
    allscore_decades.encoding['dtype'] = np.dtype('int8')
    encoding = {}
    encoding['crop_suitability_score'] = allscore_decades.encoding
    allscore_decades.to_netcdf(os.path.join(outdir, cropname + '_decades.nc'), encoding=encoding)

    tempscore_decades.encoding['zlib'] = True
    tempscore_decades.encoding['complevel'] = 1
    tempscore_decades.encoding['shuffle'] = False
    tempscore_decades.encoding['contiguous'] = False
    tempscore_decades.encoding['dtype'] = np.dtype('int8')
    encoding = {}
    encoding['temperature_suitability_score'] = tempscore_decades.encoding
    tempscore_decades.to_netcdf(os.path.join(outdir, cropname + '_tempscore_decades.nc'), encoding=encoding)

    precscore_decades.encoding['zlib'] = True
    precscore_decades.encoding['complevel'] = 1
    precscore_decades.encoding['shuffle'] = False
    precscore_decades.encoding['contiguous'] = False
    precscore_decades.encoding['dtype'] = np.dtype('int8')
    encoding = {}
    encoding['precip_suitability_score'] = precscore_decades.encoding
    precscore_decades.to_netcdf(os.path.join(outdir, cropname + '_precscore_decades.nc'), encoding=encoding)

    # decadal changes
    allscore_decadal_changes = allscore_decades.copy()[1:,:,:]
    tempscore_decadal_changes = tempscore_decades.copy()[1:,:,:]
    precscore_decadal_changes = precscore_decades.copy()[1:,:,:]
    decs = allscore_decades.shape[0]
    for dec in range(1,decs):
        allscore_decadal_changes[dec-1,:,:] = allscore_decades[dec,:,:] - \
                                              allscore_decades[0,:,:]
        tempscore_decadal_changes[dec-1,:,:] = tempscore_decades[dec,:,:] - \
                                               tempscore_decades[0,:,:]
        precscore_decadal_changes[dec-1,:,:] = precscore_decades[dec,:,:] - \
                                               precscore_decades[0,:,:]
    allscore_decadal_changes.to_netcdf(os.path.join(outdir, cropname + '_decadal_changes.nc'))
    tempscore_decadal_changes.to_netcdf(os.path.join(outdir, cropname + '_tempscore_decadal_changes.nc'))
    precscore_decadal_changes.to_netcdf(os.path.join(outdir, cropname + '_precscore_decadal_changes.nc'))

    return allscore_decades, tempscore_decades, precscore_decades, allscore_decadal_changes, tempscore_decadal_changes, precscore_decadal_changes

def calc_decadal_doy_changes(maxdoys, maxdoys_temp, maxdoys_prec, SOIL, LCMloc, sgmloc, cropname, outdir):
    '''
    Calculate decadal changes in the 'day of year of the maximum score' metric, 
    using circular averaging 
    '''

    # mask land-cover and soil
    lcm = xr.open_dataset(LCMloc, engine="rasterio")
    lcm = lcm['band_data']
    lcm = lcm.drop('band').squeeze()
    lcm=lcm[::-1,:]
    maxdoys      = lcm_mask(lcm, maxdoys)
    maxdoys_temp = lcm_mask(lcm, maxdoys_temp)
    maxdoys_prec = lcm_mask(lcm, maxdoys_prec)
    maxdoys       = soil_type_mask_all(maxdoys,  SOIL, sgmloc)
    maxdoys_temp  = soil_type_mask_all(maxdoys_temp,  SOIL, sgmloc)
    maxdoys_prec  = soil_type_mask_all(maxdoys_prec,  SOIL, sgmloc)
    # compress and save to disk
    maxdoys.encoding['zlib'] = True
    maxdoys.encoding['complevel'] = 1
    maxdoys.encoding['shuffle'] = False
    maxdoys.encoding['contiguous'] = False
    maxdoys.encoding['dtype'] = np.dtype('uint16')
    encoding = {}
    encoding['dayofyear'] = maxdoys.encoding
    maxdoys.to_netcdf(os.path.join(outdir, cropname + '_max_score_doys.nc'), encoding=encoding)

    maxdoys_temp.encoding['zlib'] = True
    maxdoys_temp.encoding['complevel'] = 1
    maxdoys_temp.encoding['shuffle'] = False
    maxdoys_temp.encoding['contiguous'] = False
    maxdoys_temp.encoding['dtype'] = np.dtype('int16')
    encoding = {}
    encoding['dayofyear'] = maxdoys_temp.encoding
    maxdoys_temp.to_netcdf(os.path.join(outdir, cropname + '_max_tempscore_doys.nc'), encoding=encoding)

    maxdoys_prec.to_netcdf(os.path.join(outdir, cropname + '_max_precscore_doys.nc'))
    maxdoys_prec.encoding['zlib'] = True
    maxdoys_prec.encoding['complevel'] = 1
    maxdoys_prec.encoding['shuffle'] = False
    maxdoys_prec.encoding['contiguous'] = False
    maxdoys_prec.encoding['dtype'] = np.dtype('int16')
    encoding = {}
    encoding['dayofyear'] = maxdoys_prec.encoding
    maxdoys_prec.to_netcdf(os.path.join(outdir, cropname + '_max_precscore_doys.nc'), encoding=encoding)


    # calculate the decadal averages, using circular averaging
    maxdoys_decades = []
    maxdoys_temp_decades = []
    maxdoys_prec_decades = []
    syear = maxdoys['year'][0].values
    nyears = maxdoys.shape[0]
    nyears_u = int(np.floor(nyears/10)*10)
    for idx in range(0, nyears_u, 10):
        maxdoys_decade       = maxdoys[idx:idx+10,:,:]
        maxdoys_temp_decade  = maxdoys_temp[idx:idx+10,:,:]
        maxdoys_prec_decade  = maxdoys_prec[idx:idx+10,:,:]
        maxdoys_decade       = circular_avg(maxdoys_decade, 'year')
        maxdoys_temp_decade  = circular_avg(maxdoys_temp_decade, 'year')
        maxdoys_prec_decade  = circular_avg(maxdoys_prec_decade, 'year')
        maxdoys_decade       = maxdoys_decade.expand_dims({'decade': [syear+idx]})
        maxdoys_temp_decade  = maxdoys_temp_decade.expand_dims({'decade': [syear+idx]})
        maxdoys_prec_decade  = maxdoys_prec_decade.expand_dims({'decade': [syear+idx]})
        maxdoys_decades.append(maxdoys_decade)
        maxdoys_temp_decades.append(maxdoys_temp_decade)
        maxdoys_prec_decades.append(maxdoys_prec_decade)
    maxdoys_decades       = xr.merge(maxdoys_decades)['dayofyear']
    maxdoys_temp_decades  = xr.merge(maxdoys_temp_decades)['dayofyear']
    maxdoys_prec_decades  = xr.merge(maxdoys_prec_decades)['dayofyear']
    # save to disk
    maxdoys_decades.to_netcdf(os.path.join(outdir, cropname + '_max_score_doys_decades.nc'))
    maxdoys_temp_decades.to_netcdf(os.path.join(outdir, cropname + '_max_tempscore_doys_decades.nc'))
    maxdoys_prec_decades.to_netcdf(os.path.join(outdir, cropname + '_max_precscore_doys_decades.nc'))
    
    # calculate the decadal changes from the 2020s, using modulo (circular) arithmetic
    maxdoys_decadal_changes      = maxdoys_decades.copy()[1:,:,:]
    maxdoys_temp_decadal_changes = maxdoys_temp_decades.copy()[1:,:,:]
    maxdoys_prec_decadal_changes = maxdoys_prec_decades.copy()[1:,:,:]
    decs = maxdoys_decades.shape[0]
    for dec in range(1,decs):
        maxdoys_decadal_changes[dec-1,:,:] = (maxdoys_decades[dec,:,:] - \
                                              maxdoys_decades[0,:,:])
        maxdoys_temp_decadal_changes[dec-1,:,:] = (maxdoys_temp_decades[dec,:,:] - \
                                                   maxdoys_temp_decades[0,:,:])
        maxdoys_prec_decadal_changes[dec-1,:,:] = (maxdoys_prec_decades[dec,:,:] - \
                                                   maxdoys_prec_decades[0,:,:])
    maxdoys_decadal_changes = xr.where(maxdoys_decadal_changes > 180, maxdoys_decadal_changes % -180,
                              xr.where(maxdoys_decadal_changes < -180, maxdoys_decadal_changes % 180,
                                       maxdoys_decadal_changes))
    maxdoys_temp_decadal_changes = xr.where(maxdoys_temp_decadal_changes > 180, maxdoys_temp_decadal_changes % -180,
                                   xr.where(maxdoys_temp_decadal_changes < -180, maxdoys_temp_decadal_changes % 180,
                                            maxdoys_temp_decadal_changes))
    maxdoys_prec_decadal_changes = xr.where(maxdoys_prec_decadal_changes > 180, maxdoys_prec_decadal_changes % -180,
                                   xr.where(maxdoys_prec_decadal_changes < -180, maxdoys_prec_decadal_changes % 180,
                                            maxdoys_prec_decadal_changes))
    maxdoys_decadal_changes.to_netcdf(os.path.join(outdir, cropname + '_max_score_doys_decadal_changes.nc'))
    maxdoys_temp_decadal_changes.to_netcdf(os.path.join(outdir, cropname + '_max_tempscore_doys_decadal_changes.nc'))
    maxdoys_prec_decadal_changes.to_netcdf(os.path.join(outdir, cropname + '_max_precscore_doys_decadal_changes.nc'))
    return maxdoys_decadal_changes, maxdoys_temp_decadal_changes, maxdoys_prec_decadal_changes
    
def calc_decadal_kprop_changes(ktmpap, kmaxap, SOIL, LCMloc, sgmloc, cropname, outdir):
    '''
    Calculate decadal changes in the average proportion of ktmp & kmax days
    '''

    # Calculate monthly average
    ktmpap_monavg = ktmpap.resample(time="1MS").mean(dim="time")
    kmaxap_monavg = kmaxap.resample(time="1MS").mean(dim="time")

    # mask
    lcm = xr.open_dataset(LCMloc, engine='rasterio')
    lcm = lcm['band_data']
    lcm = lcm.drop('band').squeeze()
    lcm=lcm[::-1,:]
    ktmpap_monavg = lcm_mask(lcm, ktmpap_monavg)
    kmaxap_monavg = lcm_mask(lcm, kmaxap_monavg)
    ktmpap_monavg = soil_type_mask_all(ktmpap_monavg,  SOIL, sgmloc)
    kmaxap_monavg = soil_type_mask_all(kmaxap_monavg,  SOIL, sgmloc)
    
    # calculate monthly climatologies for each decade
    ktmpap_monavg_climos = []
    kmaxap_monavg_climos = []
    nyears = ktmpap_monavg.shape[0] // 12
    ndecs = int(np.round(nyears, -1)/10)
    for d in range(0, ndecs):
        sind = d*120
        eind = (d+1)*120
        year = ktmpap_monavg['time'][sind].dt.year
        if eind >= ktmpap_monavg.shape[0]:
            ktmpap_monavg_climo = ktmpap_monavg[sind:,:,:].groupby('time.month').mean().expand_dims({'decade': [year]})
            kmaxap_monavg_climo = kmaxap_monavg[sind:,:,:].groupby('time.month').mean().expand_dims({'decade': [year]})
            ktmpap_monavg_climos.append(ktmpap_monavg_climo)
            kmaxap_monavg_climos.append(kmaxap_monavg_climo)
        else:
            ktmpap_monavg_climo = ktmpap_monavg[sind:eind,:,:].groupby('time.month').mean().expand_dims({'decade': [year]})
            kmaxap_monavg_climo = kmaxap_monavg[sind:eind,:,:].groupby('time.month').mean().expand_dims({'decade': [year]})
            ktmpap_monavg_climos.append(ktmpap_monavg_climo)
            kmaxap_monavg_climos.append(kmaxap_monavg_climo)
    
    ktmpap_monavg_climos2 = xr.concat(ktmpap_monavg_climos,
                                      dim='decade')
    kmaxap_monavg_climos2 = xr.concat(kmaxap_monavg_climos,
                                      dim='decade')
    # compress and save to disk
    ktmpap_monavg_climos2.encoding['zlib'] = True
    ktmpap_monavg_climos2.encoding['complevel'] = 1
    ktmpap_monavg_climos2.encoding['shuffle'] = False
    ktmpap_monavg_climos2.encoding['contiguous'] = False
    ktmpap_monavg_climos2.encoding['dtype'] = np.dtype('float32')
    encoding = {}
    encoding['average_proportion_of_ktmp_days_in_gtime'] = ktmpap_monavg_climos2.encoding
    ktmpap_monavg_climos2.to_netcdf(os.path.join(outdir, cropname + '_ktmpdaysavgprop_decades.nc'), encoding=encoding)

    kmaxap_monavg_climos2.encoding['zlib'] = True
    kmaxap_monavg_climos2.encoding['complevel'] = 1
    kmaxap_monavg_climos2.encoding['shuffle'] = False
    kmaxap_monavg_climos2.encoding['contiguous'] = False
    kmaxap_monavg_climos2.encoding['dtype'] = np.dtype('float32')
    encoding = {}
    encoding['average_proportion_of_kmax_days_in_gtime'] = kmaxap_monavg_climos2.encoding
    kmaxap_monavg_climos2.to_netcdf(os.path.join(outdir, cropname + '_kmaxdaysavgprop_decades.nc'), encoding=encoding)
    
    # difference the climatologies
    ktmpap_monavg_climo_diffs = ktmpap_monavg_climos2.copy()[1:]
    kmaxap_monavg_climo_diffs = kmaxap_monavg_climos2.copy()[1:]
    decs = kmaxap_monavg_climos2.shape[0]
    for dec in range(1,decs):
        ktmpap_monavg_climo_diffs[dec-1] = ktmpap_monavg_climos2[dec] - \
                                           ktmpap_monavg_climos2[0]
        kmaxap_monavg_climo_diffs[dec-1] = kmaxap_monavg_climos2[dec] - \
                                           kmaxap_monavg_climos2[0]
    
    # compress and save to disk
    ktmpap_monavg_climo_diffs.encoding['zlib'] = True
    ktmpap_monavg_climo_diffs.encoding['complevel'] = 1
    ktmpap_monavg_climo_diffs.encoding['shuffle'] = False
    ktmpap_monavg_climo_diffs.encoding['contiguous'] = False
    ktmpap_monavg_climo_diffs.encoding['dtype'] = np.dtype('float32')
    encoding = {}
    encoding['average_proportion_of_ktmp_days_in_gtime'] = ktmpap_monavg_climo_diffs.encoding
    ktmpap_monavg_climo_diffs.to_netcdf(os.path.join(outdir, cropname + '_ktmpdaysavgprop_decadal_changes.nc'), encoding=encoding)

    kmaxap_monavg_climo_diffs.encoding['zlib'] = True
    kmaxap_monavg_climo_diffs.encoding['complevel'] = 1
    kmaxap_monavg_climo_diffs.encoding['shuffle'] = False
    kmaxap_monavg_climo_diffs.encoding['contiguous'] = False
    kmaxap_monavg_climo_diffs.encoding['dtype'] = np.dtype('float32')
    encoding = {}
    encoding['average_proportion_of_kmax_days_in_gtime'] = kmaxap_monavg_climo_diffs.encoding
    kmaxap_monavg_climo_diffs.to_netcdf(os.path.join(outdir, cropname + '_kmaxdaysavgprop_decadal_changes.nc'), encoding=encoding)

    return ktmpap_monavg_climo_diffs, kmaxap_monavg_climo_diffs

def plot_decade(allscore, tempscore, precscore, save=None):
    fig, axs = plt.subplots(1,3, subplot_kw={'projection': cp.crs.OSGB()})
    fig.set_figwidth(10)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.coastlines(resolution='10m')
    ax2.coastlines(resolution='10m')
    ax3.coastlines(resolution='10m')
    
    allscore.where(allscore>0).plot(ax=ax1, vmin=0, vmax=100)
    tempscore.where(tempscore>0).plot(ax=ax2, vmin=0, vmax=100)
    precscore.where(precscore>0).plot(ax=ax3, vmin=0, vmax=100)
    
    cbarax1 = ax1.collections[0].colorbar.ax
    cbarax2 = ax2.collections[0].colorbar.ax
    cbarax3 = ax3.collections[0].colorbar.ax
    cbarax1.set_ylabel('')
    cbarax2.set_ylabel('')
    cbarax3.set_ylabel('')
    
    ax1.set_title('crop_suitability')
    ax2.set_title('temperature_suitability')
    ax3.set_title('precip_suitability')

    if not save==None:
        savedir = os.path.dirname(save)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(save, dpi=300)
        plt.close()


def plot_decadal_changes(dcdata, save=None, cmin=None, cmax=None, revcolbar=None):
    
    if not cmax:
        cmax2 = np.ceil(dcdata.max().values)
        if cmax2 not in (0., 1.):
            cmax = cmax2
        else:
            cmax = dcdata.max().values
        print(cmax)
    if not cmin:
        cmin2 = np.floor(dcdata.min().values)
        if cmin2 not in (0., -1.):
            cmin = cmin2
        else:
            cmin = dcdata.min().values
        print(cmin)
    
    if abs(cmax) > abs(cmin):
        cmin = -1*cmax
    else:
        cmax = -1*cmin

    fig, axs = plt.subplots(1,3, subplot_kw={'projection': cp.crs.OSGB()})
    fig.set_figwidth(10)    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.coastlines(resolution='10m')
    ax2.coastlines(resolution='10m')
    ax3.coastlines(resolution='10m')

    # get colourbar limits
    #dcdata[0,:,:].plot(ax=ax1, robust=True)
    #vmax = ax1.collections[0].colorbar.vmax
    #vmin = ax1.collections[0].colorbar.vmin
    #ax1.collections[0].colorbar.remove()
    
    #ax1.pcolormesh(dcdata['x'].values, dcdata['y'].values, dcdata[0,:,:].values, cmap='bwr_r', vmin=cmin, vmax=cmax)
    #ax2.pcolormesh(dcdata['x'].values, dcdata['y'].values, dcdata[2,:,:].values, cmap='bwr_r', vmin=cmin, vmax=cmax)
    #c = ax3.pcolormesh(dcdata['x'].values, dcdata['y'].values, dcdata[4,:,:].values, cmap='bwr_r', vmin=cmin, vmax=cmax)
    if not revcolbar:
        cmap = 'bwr_r'
    else:
        cmap = 'bwr'
    dcdata[0,:,:].plot(ax=ax1, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata[2,:,:].plot(ax=ax2, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata[4,:,:].plot(ax=ax3, vmin=cmin, vmax=cmax, cmap=cmap)

    cbartext = dcdata.name + '_change'
    cbarax1 = ax1.collections[0].colorbar.ax
    cbarax2 = ax2.collections[0].colorbar.ax
    cbarax3 = ax3.collections[0].colorbar.ax
    cbarax1.set_ylabel('')
    cbarax2.set_ylabel('')
    cbarax3.set_ylabel(cbartext)
    #cbar = plt.colorbar(c)
    #cbar.ax.yaxis.set_label_text(cbartext)
    
    ax1.set_title('2030s')
    ax2.set_title('2050s')
    ax3.set_title('2070s')
    
    #fig.subplots_adjust(wspace=-0.1)

    if not save==None:
        savedir = os.path.dirname(save)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(save, dpi=300)
        plt.close()

def plot_degC_changes(dcdata, savedir=None, cmin=None, cmax=None, revcolbar=None):
    
    if not cmax:
        cmax2 = np.ceil(dcdata.max().values)
        if cmax2 not in (0., 1.):
            cmax = cmax2
        else:
            cmax = dcdata.max().values

    if not cmin:
        cmin2 = np.floor(dcdata.min().values)
        if cmin2 not in (0., -1.):
            cmin = cmin2
        else:
            cmin = dcdata.min().values
    
    if abs(cmax) > abs(cmin):
        cmin = -1*cmax
    else:
        cmax = -1*cmin

    fig, axs = plt.subplots(1,3, subplot_kw={'projection': cp.crs.OSGB()})
    fig.set_figwidth(10)    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.coastlines(resolution='10m')
    ax2.coastlines(resolution='10m')
    ax3.coastlines(resolution='10m')

    if not revcolbar:
        cmap = 'bwr_r'
    else:
        cmap = 'bwr'
    dcdata.sel(deg='2C').plot(ax=ax1, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata.sel(deg='3C').plot(ax=ax2, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata.sel(deg='4C').plot(ax=ax3, vmin=cmin, vmax=cmax, cmap=cmap)

    cbartext = dcdata.name + '_change'
    cbarax1 = ax1.collections[0].colorbar.ax
    cbarax2 = ax2.collections[0].colorbar.ax
    cbarax3 = ax3.collections[0].colorbar.ax
    cbarax1.set_ylabel('')
    cbarax2.set_ylabel('')
    cbarax3.set_ylabel(cbartext, fontsize='x-small')
    cbarax1.tick_params(axis='y', labelsize=7)
    cbarax2.tick_params(axis='y', labelsize=7)
    cbarax3.tick_params(axis='y', labelsize=7)
    
    cropname = str(dcdata.crop.values)
    ax1.set_title('Suitability change for ' + cropname + ' at 2C', size='x-small')
    ax2.set_title('Suitability change for ' + cropname + ' at 3C', size='x-small')
    ax3.set_title('Suitability change for ' + cropname + ' at 4C', size='x-small')

    if not savedir==None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plotname = cropname + '_degC_changes.png'
        savepath = os.path.join(savedir, plotname)
        plt.savefig(savepath, dpi=600)
        plt.close()

# Clever function that does a forward rolling sum without loops
# see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
# can't be used with numba as it doesn't support cumsum on n-d arrays
def frs3D(ind, window, dtype):
    inds = np.cumsum(ind, axis=0, dtype=dtype)
    tmp = inds[window:,...] - inds[:-window,...]
    tmp2 = inds[window-1,...]
    ret = np.concatenate([tmp2[None,...], tmp], axis=0)
    return ret 

# same function, without the initial step
def frs3Dwcs(ind, window):
    tmp = ind[window:,...] - ind[:-window,...]
    tmp2 = ind[window-1,...]
    ret = np.concatenate([tmp2[None,...], tmp], axis=0)
    return ret



def add_glen_dim(ds):
    glen = ds.encoding['source'].split('.')[0].split('_')[-1]
    ds = ds.expand_dims({'glen': [int(glen)]})
    return ds

#@njit(parallel=True)
def score_temp(gtime, gmin, gmax):
    score = 100*(1 - ((gtime-gmin)/(gmax-gmin)))
    return np.round(score).astype('uint8')

#@njit(parallel=True)
#def score_prec(total, pmin, pmax, popmin, popmax):
#    score = xr.where(total > popmax, (100/(pmax-popmax))*(pmax-total), \
#            xr.where(total > popmin, 100, \
#            (100/(popmin-pmin))*(total-pmin)))
#    return score.round()

def score_temp2(temp, tmin, tmax, topmin, topmax):
    tmin = tmin.astype('float32')
    tmax = tmax.astype('float32')
    topmin = topmin.astype('float32')
    topmax = topmax.astype('float32')
    score = xr.where(temp > tmax, 0, \
            xr.where(temp > topmax, (tmax - temp)/(tmax - topmax), \
            xr.where(temp > topmin, 1, \
            xr.where(temp > tmin, (temp - tmin)/(topmin - tmin), 0))))
    return score.astype('float16')

def score_temp3(avgt, tmin, tmax, topmin, topmax):
    tmin = tmin.astype('float32')
    tmax = tmax.astype('float32')
    topmin = topmin.astype('float32')
    topmax = topmax.astype('float32')
    score = xr.where(avgt > tmax, 0, \
            xr.where(avgt > topmax, (100/(tmax-topmax)) * (tmax-avgt), \
            xr.where(avgt > topmin, 1, \
            xr.where(avgt > tmin, (100/(topmin-tmin)) * (avgt-tmin), 0))))
    return score.round().astype('uint8')

def score_temp4(avgt, tmin, tmax, topmin, topmax):
    tmin = tmin.astype('float32')
    tmax = tmax.astype('float32')
    topmin = topmin.astype('float32')
    topmax = topmax.astype('float32')
    score = xr.where(avgt > tmax, 0, \
            xr.where(avgt > 0.5*(topmax+topmin), (100/(tmax-0.5*(topmax+topmin)))*(tmax-avgt), \
            xr.where(avgt > tmin, (100/(0.5*(topmax+topmin)-tmin))*(avgt-tmin), 0)))
    return score.round().astype('uint8')


def score_prec1(total, pmin, pmax, popmin, popmax):
    pmin=pmin.astype('float32')
    pmax=pmax.astype('float32')
    popmin=popmin.astype('float32')
    popmax=popmax.astype('float32')
    score = xr.where(total > pmax, 0, \
            xr.where(total > popmax, (100/(pmax-popmax))*(pmax-total), \
            xr.where(total > popmin, 100, \
            xr.where(total > pmin, (100/(popmin-pmin))*(total-pmin), 0))))
    return score.round().astype('uint8')

def score_prec2(total, pmin, pmax, popmin, popmax):
    pmin=pmin.astype('float32')
    pmax=pmax.astype('float32')
    popmin=popmin.astype('float32')
    popmax=popmax.astype('float32')
    score = xr.where(total > pmax, 0, \
            xr.where(total > 0.5*(popmax+popmin), (200/(2*pmax-popmin-popmax))*(pmax-total), \
            xr.where(total > pmin, (200/(popmin+popmax-2*pmin))*(total-pmin), 0)))
    return score.round().astype('uint8')

def score_prec3(total, pmin, pmax, popmin, popmax):
    pmin=pmin.astype('float32')
    pmax=pmax.astype('float32')
    popmin=popmin.astype('float32')
    popmax=popmax.astype('float32')
    score = xr.where(total > pmax, 0, \
            xr.where(total > popmax, 50*((popmax-total)/(pmax-popmax) + 1), \
            xr.where(total > 0.5*(popmax+popmin), 50*((2*(popmax-total))/(popmax-popmin) + 1), \
            xr.where(total > popmin, 50*((2*(total-popmin))/(popmax-popmin) + 1), \
            xr.where(total > pmin, 50*((total-popmin)/(popmin-pmin) + 1), 0)))))
    return score.round().astype('uint8')


def factors(n):
    '''
    Function to calculate factors of a number from stackoverflow:
    https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    '''

    return set(
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    )

def time_chunk_calc(window, tlen):
    '''
    Calculate the optimal time dimension chunksize for dask for a given rolling window size.
    This has to be greater than or equal to half the rolling window size and also be 
    a factor of the total time dimension length (i.e. divide into it with no remainder).
    The 'optimal' time dimension chunksize is the factor nearest to half the rolling window size.

    Inputs:
    window - Rolling window size
    tlen   - Total length of time dimension

    Outputs:
    Optimal chunk size. 
    '''
    
    minlen = int(np.floor(window/2.))
    lenfactors = list(factors(tlen))
    lenfactors.sort()

    for factor in lenfactors:
        if factor >= minlen:
            optimal = factor
            break
        else:
            continue

    if optimal==tlen:
        print('There is no factor of dataset time dimension (' + str(tlen) + ') that is ' + \
              'greater than half the rolling window size (' + str(minlen) + '), ' + \
              'checking for chunksizes with a remainder greater than ' + str(minlen))

        for cs in range(minlen, tlen-minlen):
            remainder = tlen % cs
            if remainder >= minlen:
                optimal = cs
                break
            else:
                continue

        if remainder == tlen-minlen-1:
            print('Still no possible chunksize, no chunking will be carried out, may lead to memory issues.')
        
    print('Rolling window size is ' + str(window) + ' --> minimum dask chunksize is ' + str(minlen))
    print('Time dimension length is ' + str(tlen) + ', nearest possible chunksize equal to or above ' + str(minlen) + ' is ' + str(optimal))
    print('Time chunksize will therefore be set to ' + str(optimal))
        
    return optimal


def read_grd_file(filen, mask=0):
    vals = np.loadtxt(filen,skiprows=6)
    if mask==1:
        vals = np.ma.masked_values(vals, 0)
        vals = np.ma.masked_values(vals, -9999.)
        # preserve the mask whilst setting masked values to nans
        vals.data[np.where(vals.data == 0)]      = np.nan 
        vals.data[np.where(vals.data == -9999.)] = np.nan
    return vals
    
def read_nc(filen,var):
    ncf = nc4.Dataset(filen,'r')    # open and read from netcdf file
    data = ncf.variables[var][:]    # variable
    x = ncf.variables['x'][:]       # easting
    y = ncf.variables['y'][:]       # northing
    if len(data.shape) >= 3:     # time
        t = ncf.variables['t'][:]
        tunits = ncf.variables['t'].units
    else:
        t = None
        tunits = None
    ncf.close()                     # close netcdf file  
    return data,x,y,t,tunits
    
def get_coords(filein):
    filen = open(filein,'r')
    ncols = int(filen.readline().split()[1])
    nrows = int(filen.readline().split()[1])
    xllc  = float(filen.readline().split()[1])
    yllc  = float(filen.readline().split()[1])
    res   = float(filen.readline().split()[1])
    nodata= float(filen.readline().split()[1])
    filen.close()
    return nrows, ncols, xllc, yllc, res, nodata

def make_xarray(filein, xdimname='x', ydimname='y'):
    nrows, ncols, xllc, yllc, res, nodata = get_coords(filein)
    xcoords = np.linspace(xllc + (res/2), xllc + (res/2) + (res*(ncols-1)), ncols)
    ycoords = np.linspace(yllc + (res/2), yllc + (res/2) +  (res*(nrows-1)), nrows)[::-1]
    vals = np.loadtxt(filein, skiprows=6)
    vals = xr.DataArray(vals, coords=[(ydimname, ycoords), (xdimname, xcoords)])
    vals = vals.where(vals != nodata)
    return vals, xcoords, ycoords, nodata

def nctoxr(filein, var):
    data,x,y = read_nc(filein, var)
    dataxr = xr.DataArray(data, coords=[('y',y), ('x',x)])
    return dataxr

def subset_to_template(filein, template):
    tvals, xcoords_template, ycoords_template, nodata = make_xarray(template)
    tvalsnp = tvals.values.copy()
    vals, xcoords, ycoords, nodata = make_xarray(filein)
    subvals = vals.sel(y = slice(ycoords_template[0], ycoords_template[-1]),
                       x  = slice(xcoords_template[0], xcoords_template[-1]))
    newxcoords = subvals.coords['x'].values
    newycoords = subvals.coords['y'].values
    return subvals, newxcoords, newycoords, nodata

def grd_to_netcdf_single(filein, varname, units, filesave=1, filenameout='grd.ncf', subset=0, template=None):
    if subset == 1:
        vals, junk1, junk2, nodata = subset_to_template(filein, template)
    else:
        vals, junk, junk2, nodata = make_xarray(filein)
    vals.name = varname
    vals.attrs = {'Units': units, 'missing_value': nodata}
    vals.coords['x'].attrs = {'long_name':'easting', 'standard_name':'projection_x_coordinate', 'units':'m', 'point_spacing':'even', 'axis':'x'}
    vals.coords['y'].attrs = {'long_name':'northing', 'standard_name':'projection_y_coordinate', 'units':'m', 'point_spacing':'even', 'axis':'y'}

    if filesave == 1:
        vals.to_netcdf(filenameout)
    else:
        return vals, nodata

    


# NOT USED
def inddaycalc(metdata, ecocrop, index, comparison, gyear, yeardata=0, saveloc='.', x='projection_x_coordinate', y='projection_y_coordinate', t='time'):
    '''
    Calculate the number of days an index in the ecocrop spreadsheet exceeds,is less than,equals
    the relevant metdata. 
    Inputs:
    metdata: Xarray dataset containing the metdata for the comparison. Dimensions: [time,y,x]
    ecocrop: The ecocrop spreadsheet as a pandas DataFrame, containing the index, and 'ScientificName' cols
    index: The name of the index for the comparison, must be one of the col names in ecocrop
    comparison: The type of comparison to do. Choose from:
                'greater', 'greater_or_equal', 'equal', 'less_or_equal', 'less'. 
    gyear: 1 or 0. Choose whether or not to use the growing year or calendar year for the comparison
           Using the growing year requires the 'GMAX', 'GMIN' and 'GLEN' columns to be in ecocrop
           Ignored if yeardata==1
    yeardata: 1 or 0. Set whether of not the metdata is yearly (1) or daily (0)
    saveloc: Directory to save the resulting netcdf file
    x: X coordinate name of the metdata. Defaults to 'projection_x_coordinate'
    y: Y coordinate name of the metdata. Defaults to 'projection_y_coordinate'
    t: t coordinate name of the metdata. Defaults to 'time'

    Outputs:
    inds: Xarray dataset with one variable per plant of dimensions [year,y,x] containing the sum total
          of days per year in which the condition was met. This will also be saved to disk in saveloc
          with name INDEXdays.nc where INDEX is replaced with the index var used in the comparison.
    '''

    if yeardata==0:
        # determine number of whole calendar years
        for tt in range(0, len(metdata[t])):
            times = metdata[t].values
            mon = times[tt].month
            day = times[tt].day
            if mon==1 and day==1:
                sind = tt
                break

        for tt in range(0, len(metdata[t])):
            times = metdata[t].values[::-1]
            mon = times[tt].month
            day = times[tt].day
            if mon==12 and day==30:
                eind = tt
                break
        
        startyear = metdata[t].values[sind].year
        endyear = metdata[t].values[::-1][eind].year
    else:
        startyear = metdata[t][0].values
        endyear = metdata[t][-1].values
    nyears = endyear - startyear + 1

    # create xarray to store results of comparison
    plantarrs = {}
    sns = []
    years = np.arange(startyear,endyear+1)
    xs = metdata[x].values
    ys = metdata[y].values
    for row in range(0, ecocrop.shape[0]):
        sn = ecocrop.loc[row,'ScientificName']
        sn = '_'.join(sn.split(' '))
        sns.append(sn)
        if not np.isnan(ecocrop.loc[row,index]):
            basearr = np.zeros((nyears-1, len(ys), len(xs)))
            plantarrs[sn] = xr.DataArray(basearr, coords=[years[:-1], ys, xs], dims=['year', 'y', 'x'])
    inds = xr.Dataset(plantarrs)


    metdata = metdata.rename({t: 'time'})
    for year in range(startyear, endyear):
        # Step 1: Trim down driving data to year in question, 
        # with an extra year on the end to account for cases where 
        # the growing season spans the year boundary.
        if yeardata==0:
            metdata_year = metdata.sel(time=slice(str(year),str(year+1)))
        else:
            metdata_year = metdata.sel(time=year)

        for row in range(0, ecocrop.shape[0]):
            plantdata = ecocrop.loc[row,:]
            if not np.isnan(plantdata[index]):
                sn = ecocrop.loc[row,'ScientificName']
                sn = '_'.join(sn.split(' '))
                print('Calculating ' + index + ' inds for ' + sn + ' for ' + str(year))
                
                # Step 2: Create an array of zeros matching the shape of tempn_year. 
                # Where tempn is equal or below the killing temp, set to 1, 
                # otherwise leave as 0.
                if comparison=='less_or_equal':
                    tmparr = np.where(np.isnan(metdata_year.values), np.nan, 
                                      np.where(metdata_year.values <= plantdata[index], 1, 0))
                elif comparison=='less':
                    tmparr = np.where(np.isnan(metdata_year.values), np.nan, 
                                      np.where(metdata_year.values < plantdata[index], 1, 0))
                elif comparison=='equal':
                    tmparr = np.where(np.isnan(metdata_year.values), np.nan, 
                                      np.where(metdata_year.values == plantdata[index], 1, 0))
                elif comparison=='greater_or_equal':
                    tmparr = np.where(np.isnan(metdata_year.values), np.nan, 
                                      np.where(metdata_year.values >= plantdata[index], 1, 0))
                elif comparison=='greater':
                    tmparr = np.where(np.isnan(metdata_year.values), np.nan, 
                                      np.where(metdata_year.values > plantdata[index], 1, 0))
                    
                    
                # Step 3: Sum up over the time dimension, from GMIN to GMAX
                # If GMIN/GMAX NaN, assume growing season spans whole calendar year,
                # Also if gyear!=1, use calendar year
                if yeardata==0:
                    if gyear==1:
                        if np.isnan(plantdata['GMIN']):
                            gmin=1
                        else:
                            gmin = int(plantdata['GMIN'])
                        if np.isnan(plantdata['GMAX']):
                            gmax=360
                        else:
                            gmax = int(plantdata['GMAX'])
                        if gmin==0 and gmax==0:
                            gmin=1
                            gmax=360
                            
                        if gmax < gmin:
                            gmax+=360
                    else:
                        gmin=1
                        gmax=360

                    tmpsum = tmparr[gmin-1:gmax].sum(axis=0)
                else:
                    tmpsum = tmparr

                #plt.pcolormesh(tmpsum)
                #plt.title(sn)
                #display.display(plt.gcf())
                #plt.close()
                
                # Step 4: Store in existing xarray dataset
                inds[sn].loc[dict(year=year)] = tmpsum
                
    encoding = {var: {'dtype': 'int16', '_FillValue': -9999} for var in inds.data_vars}
    fullpath = os.path.join(saveloc, index + 'days.nc')
    if not os.path.exists(fullpath):
        inds.to_netcdf(os.path.join(saveloc, index + 'days.nc'), encoding=encoding)
    else:
        print('File already exists, not saving netcdf file')

    return inds
