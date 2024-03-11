import os
import xarray as xr
import numpy as np
import cartopy as cp
import netCDF4 as nc4
import rioxarray as rioxr
import matplotlib.pyplot as plt


def circular_avg(maxdoys, dim):
    """
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
    """
    maxdoys_rad = np.deg2rad(maxdoys)

    maxdoys_sin = np.sin(maxdoys_rad)
    maxdoys_cos = np.cos(maxdoys_rad)

    maxdoys_sinsum = maxdoys_sin.sum(dim) / len(maxdoys[dim])
    maxdoys_cossum = maxdoys_cos.sum(dim) / len(maxdoys[dim])

    maxdoys_radavg = xr.where(
        xr.ufuncs.logical_and(maxdoys_sinsum > 0, maxdoys_cossum > 0),
        xr.ufuncs.arctan(maxdoys_sinsum / maxdoys_cossum),
        xr.where(
            xr.ufuncs.logical_and(maxdoys_sinsum < 0, maxdoys_cossum > 0),
            xr.ufuncs.arctan(maxdoys_sinsum / maxdoys_cossum) + (2 * np.pi),
            xr.where(
                maxdoys_cossum < 0,
                xr.ufuncs.arctan(maxdoys_sinsum / maxdoys_cossum) + np.pi,
                0,
            ),
        ),
    )

    maxdoys_avg = np.rad2deg(maxdoys_radavg)

    return maxdoys_avg


def lcm_mask(lcm, data):
    """
    Mask out non-growing regions using a version of the land-
    cover map

    Parameters
    ----------
    lcm : string or xarray dataarray
        path to land cover map file or xarray dataarray of it.
    data : xarray dataarray or dataset
        Data to mask.

    Returns
    -------
    data_masked : xarray dataarray or dataset
        Masked version of data

    """
    if type(lcm) == str:
        if lcm[-3:] == "tif":
            lcm = xr.open_dataset(lcm, engine="rasterio")
            lcm = lcm["band_data"]
            lcm = lcm.drop("band").squeeze()
        else:
            lcm = xr.open_dataarray(lcm)

    dataxlims = [data["x"].values[0], data["x"].values[-1]]
    dataylims = [data["y"].values[0], data["y"].values[-1]]
    lcmylims = [lcm["y"].values[0], lcm["y"].values[-1]]
    if dataylims[0] < dataylims[1]:
        if lcmylims[0] > lcmylims[1]:
            lcm = lcm[::-1, :]
    elif dataylims[0] > dataylims[1]:
        if lcmylims[0] < lcmylims[1]:
            lcm = lcm[::-1, :]

    lcm_cropped = lcm.sel(
        x=slice(dataxlims[0], dataxlims[1]),
        y=slice(dataylims[0], dataylims[1]),
    )

    data_masked_npy = np.where(lcm_cropped.values > 0, data.values, 0)
    data_masked = data.copy()
    data_masked.values = data_masked_npy
    return data_masked


def soil_type_mask(mask, data):
    """
    Mask based on soil type, using a soil type mask in netcdf format

    Parameters
    ----------
    mask : string
        path to netcdf mask file with values <=0 indicating locations to be
        masked out in data.
    data : xarray dataarray or dataset
        data to be masked.

    Returns
    -------
    data_masked : xarray dataarray or dataset
        masked version of data.

    """
    dataxlims = [data["x"].values[0], data["x"].values[-1]]
    dataylims = [data["y"].values[0], data["y"].values[-1]]

    if type(mask) == str:
        mask = xr.open_dataarray(mask)
        # this dtype conversion is to get around the case where the coordinates might
        # be in different dtypes to the datalims, sometimes causing an error
        xdtype = mask.x.dtype
        ydtype = mask.y.dtype
    mask_cropped = mask.sel(
        x=slice(dataxlims[0].astype(xdtype), dataxlims[1].astype(xdtype)),
        y=slice(dataylims[0].astype(ydtype), dataylims[1].astype(ydtype)),
    )

    data_masked_npy = np.where(mask_cropped.values > 0, data.values, 0)
    data_masked = data.copy()
    data_masked.values = data_masked_npy
    return data_masked


def soil_type_mask_all(data, SOIL, maskloc):
    """
    apply the masking function for all the soil types
    dependent on which the crop grows in (SOIL)

    Parameters
    ----------
    data : xarray dataarray or dataset
        data to be masked.
    SOIL : string
        'heavy', 'medium' or 'light', describing the soil type suitable for
        the crop
    maskloc : string
        path to netcdf mask file with values <=0 indicating locations to be
        masked out in data.

    Returns
    -------
    data_masked : xarray dataarray or dataset
        masked version of data.

    """
    if "heavy" in SOIL and "medium" in SOIL and "light" in SOIL:
        print("Doing masking for all soil groups")
        maskfile = os.path.join(maskloc, "all_soil_mask.nc")
        data = soil_type_mask(maskfile, data)
    elif "heavy" in SOIL and "medium" in SOIL:
        print("Doing masking for heavy and medium soil groups")
        maskfile = os.path.join(maskloc, "heavy_med_soil_mask.nc")
        data = soil_type_mask(maskfile, data)
    elif "heavy" in SOIL and "light" in SOIL:
        print("Doing masking for light and heavy soil groups")
        maskfile = os.path.join(maskloc, "heavy_light_soil_mask.nc")
        data = soil_type_mask(maskfile, data)
    elif "medium" in SOIL and "light" in SOIL:
        print("Doing masking for light and medium soil groups")
        maskfile = os.path.join(maskloc, "med_light_soil_mask.nc")
        data = soil_type_mask(maskfile, data)
    elif "light" in SOIL:
        print("Doing masking for light soil group")
        maskfile = os.path.join(maskloc, "light_soil_mask.nc")
        data = soil_type_mask(maskfile, data)
    elif "medium" in SOIL:
        print("Doing masking for medium soil group")
        maskfile = os.path.join(maskloc, "medium_soil_mask.nc")
        data = soil_type_mask(maskfile, data)
    elif "heavy" in SOIL:
        print("Doing masking for heavy soil group")
        maskfile = os.path.join(maskloc, "heavy_soil_mask.nc")
        data = soil_type_mask(maskfile, data)

    return data


def calculate_max_doy(allscore, tempscore, precscore):
    """
    Return the day of year of the maximum score for allscore, tempscore,
    precscore

    Inputs
    ------
    allscore: xarray dataset/dataarray
        Daily crop combined temp and prec suitability scores
    tempscore: as allscore but temperature score only
    precscore: as allscore but precipitation score only

    Returns
    -------
    maxdoys: xarray dataset/dataarray
        The day in the year that has the highest value in allscore, for each
        year, for each gridcell.
    maxdoys_temp: as maxdoys but for tempscore
    maxdoys_prec: as maxdoys but for precscore
    """

    maxdoys = []
    for yr, yrdata in allscore.groupby("time.year"):
        print("Calculating doy of max score for year " + str(yr))
        maxdoy = yrdata.idxmax("time").dt.dayofyear.expand_dims({"year": [yr]})
        maxdoy = maxdoy.where(maxdoy > 1)
        maxdoys.append(maxdoy)
    maxdoys = maxdoys[:-1]
    maxdoys = xr.concat(maxdoys, dim="year")

    maxdoys_temp = []
    for yr, yrdata in tempscore.groupby("time.year"):
        print("Calculating doy of max temp score for year " + str(yr))
        maxdoy = yrdata.idxmax("time").dt.dayofyear.expand_dims({"year": [yr]})
        maxdoy = maxdoy.where(maxdoy > 1)
        maxdoys_temp.append(maxdoy)
    maxdoys_temp = maxdoys_temp[:-1]
    maxdoys_temp = xr.concat(maxdoys_temp, dim="year")

    maxdoys_prec = []
    for yr, yrdata in precscore.groupby("time.year"):
        print("Calculating doy of max prec score for year " + str(yr))
        maxdoy = yrdata.idxmax("time").dt.dayofyear.expand_dims({"year": [yr]})
        maxdoy = maxdoy.where(maxdoy > 1)
        maxdoys_prec.append(maxdoy)
    maxdoys_prec = maxdoys_prec[:-1]
    maxdoys_prec = xr.concat(maxdoys_prec, dim="year")

    return maxdoys, maxdoys_temp, maxdoys_prec


def calc_yearly_scores_only(
    tempscore, precscore, SOIL, LCMloc, sgmloc, cropname, outdir, yearaggmethod
):
    """
    Calculate aggregated yearly crop suitability scores from the
    daily scores.
    tempscore, precscore inputs either netcdf filenames
    or xarray dataarrays. Both must have variables
    'temperature_suitability_score' and
    'precip_suitability_score', respectively

    Inputs
    ------
    SOIL: Soil group suitability string from the ecocrop database
    LCMloc: Land cover mask. Path to tif
    sgmloc: Soil group mask netcdfs folder as string
    outdir: Where to store output netcdf files
    cropname: For output filenames
    yearaggmethod: What metric to use to aggregate the scores to yearly values,
                   can be 'max', 'median', 'mean' or 'percentile'.
                   'percentile' is recommended and uses the 95th percentile.

    Outputs
    -------
    allscore_years: xarray dataset/dataarray
        The elementwise minimum of tempscore_years and precscore_years
    tempscore_years: xarray dataset/dataarray
        tempscore but aggregated to a yearly timestep according to
        yearaggmethod
    precscore_years: as tempscore_years but for precscore
    """

    print("Calculating yearly score")
    # crop suitability score for a given year is the max
    # over all days in the year
    if yearaggmethod == "max":
        tempscore_years = tempscore.groupby("time.year").max()
        precscore_years = precscore.groupby("time.year").max()
    elif yearaggmethod == "median":
        tempscore_years = tempscore.groupby("time.year").median()
        precscore_years = precscore.groupby("time.year").median()
    elif yearaggmethod == "mean":
        tempscore_years = tempscore.groupby("time.year").mean()
        precscore_years = precscore.groupby("time.year").mean()
    elif yearaggmethod == "percentile":
        tempscore_years = tempscore.groupby("time.year").quantile(0.95)
        precscore_years = precscore.groupby("time.year").quantile(0.95)
    else:
        raise SyntaxError(
            "yearaggmethod must be one of max, median, mean or percentile"
        )
    allscore_years = xr.where(
        precscore_years < tempscore_years, precscore_years, tempscore_years
    )

    print("Doing masking")
    # mask at this stage to avoid memory issues
    lcm = xr.open_dataset(LCMloc, engine="rasterio")
    lcm = lcm["band_data"]
    lcm = lcm.drop("band").squeeze()
    lcm = lcm[::-1, :]
    allscore_years = lcm_mask(lcm, allscore_years)
    tempscore_years = lcm_mask(lcm, tempscore_years)
    precscore_years = lcm_mask(lcm, precscore_years)
    allscore_years = soil_type_mask_all(allscore_years, SOIL, sgmloc)
    tempscore_years = soil_type_mask_all(tempscore_years, SOIL, sgmloc)
    precscore_years = soil_type_mask_all(precscore_years, SOIL, sgmloc)

    # compress and save to disk
    allscore_years.name = "crop_suitability_score"
    allscore_years.encoding["zlib"] = True
    allscore_years.encoding["complevel"] = 1
    allscore_years.encoding["shuffle"] = False
    allscore_years.encoding["contiguous"] = False
    allscore_years.encoding["dtype"] = np.dtype("uint8")
    encoding = {}
    encoding["crop_suitability_score"] = allscore_years.encoding
    allscore_years.to_netcdf(
        os.path.join(outdir, cropname + "_years.nc"), encoding=encoding
    )

    tempscore_years.encoding["zlib"] = True
    tempscore_years.encoding["complevel"] = 1
    tempscore_years.encoding["shuffle"] = False
    tempscore_years.encoding["contiguous"] = False
    tempscore_years.encoding["dtype"] = np.dtype("uint8")
    encoding = {}
    encoding["temperature_suitability_score"] = tempscore_years.encoding
    tempscore_years.to_netcdf(
        os.path.join(outdir, cropname + "_tempscore_years.nc"),
        encoding=encoding,
    )

    precscore_years.encoding["zlib"] = True
    precscore_years.encoding["complevel"] = 1
    precscore_years.encoding["shuffle"] = False
    precscore_years.encoding["contiguous"] = False
    precscore_years.encoding["dtype"] = np.dtype("uint8")
    encoding = {}
    encoding["precip_suitability_score"] = precscore_years.encoding
    precscore_years.to_netcdf(
        os.path.join(outdir, cropname + "_precscore_years.nc"),
        encoding=encoding,
    )

    return allscore_years, tempscore_years, precscore_years


def calc_decadal_changes(
    tempscore, precscore, SOIL, LCMloc, sgmloc, cropname, outdir, yearaggmethod
):
    """
    Calculate decadal changes of crop suitability scores from the
    daily crop suitability scores.
    tempscore, precscore inputs either netcdf filenames
    or xarray dataarrays. Both must have variables
    'temperature_suitability_score' and
    'precip_suitability_score', respectively

    Inputs
    ------
    SOIL: Soil group suitability string from the ecocrop database
    LCMloc: Land cover mask. Path to tif
    sgmloc: Soil group mask netcdfs folder as string
    outdir: Where to store output netcdf files
    cropname: For output filenames
    yearaggmethod: What metric to use to aggregate the scores to yearly values,
                   can be 'max', 'median', 'mean' or 'percentile'.
                   'percentile' is recommended and uses the 95th percentile.


    Outputs
    -------
    allscore_decades: xarray dataset/dataarray
        The elementwise minimum of tempscore_years and precscore_years
        averaged to a decadal timestep.
    tempscore_decades: xarray dataset/dataarray
        tempscore but aggregated to a decadal timestep according to
        yearaggmethod for aggregation to a yearly timestep, then averaged over
        the decades.
    precscore_decades: as tempscore_decades but for precscore
    allscore_decadal_changes: xarray dataset/dataarray
        allscore_decades but the grid elementwise differences between each
        decade and the first (which is dropped)
    tempscore_decadal_changes: as allscore_decadal_changes but for tempscore
    precscore_decadal_changes: as allscore_decadal_changes but for precscore

    """

    print("Calculating yearly score")
    # crop suitability score for a given year is the max
    # over all days in the year
    if yearaggmethod == "max":
        tempscore_years = tempscore.groupby("time.year").max()
        precscore_years = precscore.groupby("time.year").max()
    elif yearaggmethod == "median":
        tempscore_years = tempscore.groupby("time.year").median()
        precscore_years = precscore.groupby("time.year").median()
    elif yearaggmethod == "mean":
        tempscore_years = tempscore.groupby("time.year").mean()
        precscore_years = precscore.groupby("time.year").mean()
    elif yearaggmethod == "percentile":
        tempscore_years = tempscore.groupby("time.year").quantile(0.95)
        precscore_years = precscore.groupby("time.year").quantile(0.95)
    else:
        raise SyntaxError(
            "yearaggmethod must be one of max, median, mean or percentile"
        )
    allscore_years = xr.where(
        precscore_years < tempscore_years, precscore_years, tempscore_years
    )

    print("Doing masking")
    # mask at this stage to avoid memory issues
    lcm = xr.open_dataset(LCMloc, engine="rasterio")
    lcm = lcm["band_data"]
    lcm = lcm.drop("band").squeeze()
    lcm = lcm[::-1, :]
    allscore_years = lcm_mask(lcm, allscore_years)
    tempscore_years = lcm_mask(lcm, tempscore_years)
    precscore_years = lcm_mask(lcm, precscore_years)
    allscore_years = soil_type_mask_all(allscore_years, SOIL, sgmloc)
    tempscore_years = soil_type_mask_all(tempscore_years, SOIL, sgmloc)
    precscore_years = soil_type_mask_all(precscore_years, SOIL, sgmloc)

    # compress and save to disk
    allscore_years.name = "crop_suitability_score"
    allscore_years.encoding["zlib"] = True
    allscore_years.encoding["complevel"] = 1
    allscore_years.encoding["shuffle"] = False
    allscore_years.encoding["contiguous"] = False
    allscore_years.encoding["dtype"] = np.dtype("uint8")
    encoding = {}
    encoding["crop_suitability_score"] = allscore_years.encoding
    allscore_years.to_netcdf(
        os.path.join(outdir, cropname + "_years.nc"), encoding=encoding
    )

    tempscore_years.encoding["zlib"] = True
    tempscore_years.encoding["complevel"] = 1
    tempscore_years.encoding["shuffle"] = False
    tempscore_years.encoding["contiguous"] = False
    tempscore_years.encoding["dtype"] = np.dtype("uint8")
    encoding = {}
    encoding["temperature_suitability_score"] = tempscore_years.encoding
    tempscore_years.to_netcdf(
        os.path.join(outdir, cropname + "_tempscore_years.nc"),
        encoding=encoding,
    )

    precscore_years.encoding["zlib"] = True
    precscore_years.encoding["complevel"] = 1
    precscore_years.encoding["shuffle"] = False
    precscore_years.encoding["contiguous"] = False
    precscore_years.encoding["dtype"] = np.dtype("uint8")
    encoding = {}
    encoding["precip_suitability_score"] = precscore_years.encoding
    precscore_years.to_netcdf(
        os.path.join(outdir, cropname + "_precscore_years.nc"),
        encoding=encoding,
    )

    print("Calculating decadal score")
    # crop suitability score for a given decade is the mean
    # over all years in the decade
    allscore_decades = []
    tempscore_decades = []
    precscore_decades = []
    syear = allscore_years["year"][0].values
    nyears = allscore_years.shape[0]
    nyears_u = int(np.floor(nyears / 10) * 10)
    for idx in range(0, nyears_u, 10):
        allscore_decade = allscore_years[idx : idx + 10, :, :].mean(dim="year")
        tempscore_decade = tempscore_years[idx : idx + 10, :, :].mean(
            dim="year"
        )
        precscore_decade = precscore_years[idx : idx + 10, :, :].mean(
            dim="year"
        )

        allscore_decade = allscore_decade.expand_dims(
            {"decade": [syear + idx]}
        )
        tempscore_decade = tempscore_decade.expand_dims(
            {"decade": [syear + idx]}
        )
        precscore_decade = precscore_decade.expand_dims(
            {"decade": [syear + idx]}
        )

        allscore_decades.append(allscore_decade)
        tempscore_decades.append(tempscore_decade)
        precscore_decades.append(precscore_decade)
    allscore_decades = xr.merge(allscore_decades)["crop_suitability_score"]
    tempscore_decades = xr.merge(tempscore_decades)[
        "temperature_suitability_score"
    ]
    precscore_decades = xr.merge(precscore_decades)["precip_suitability_score"]

    # compress and save to disk
    allscore_decades.name = "crop_suitability_score"
    allscore_decades.encoding["zlib"] = True
    allscore_decades.encoding["complevel"] = 1
    allscore_decades.encoding["shuffle"] = False
    allscore_decades.encoding["contiguous"] = False
    allscore_decades.encoding["dtype"] = np.dtype("int8")
    encoding = {}
    encoding["crop_suitability_score"] = allscore_decades.encoding
    allscore_decades.to_netcdf(
        os.path.join(outdir, cropname + "_decades.nc"), encoding=encoding
    )

    tempscore_decades.encoding["zlib"] = True
    tempscore_decades.encoding["complevel"] = 1
    tempscore_decades.encoding["shuffle"] = False
    tempscore_decades.encoding["contiguous"] = False
    tempscore_decades.encoding["dtype"] = np.dtype("int8")
    encoding = {}
    encoding["temperature_suitability_score"] = tempscore_decades.encoding
    tempscore_decades.to_netcdf(
        os.path.join(outdir, cropname + "_tempscore_decades.nc"),
        encoding=encoding,
    )

    precscore_decades.encoding["zlib"] = True
    precscore_decades.encoding["complevel"] = 1
    precscore_decades.encoding["shuffle"] = False
    precscore_decades.encoding["contiguous"] = False
    precscore_decades.encoding["dtype"] = np.dtype("int8")
    encoding = {}
    encoding["precip_suitability_score"] = precscore_decades.encoding
    precscore_decades.to_netcdf(
        os.path.join(outdir, cropname + "_precscore_decades.nc"),
        encoding=encoding,
    )

    # decadal changes
    allscore_decadal_changes = allscore_decades.copy()[1:, :, :]
    tempscore_decadal_changes = tempscore_decades.copy()[1:, :, :]
    precscore_decadal_changes = precscore_decades.copy()[1:, :, :]
    decs = allscore_decades.shape[0]
    for dec in range(1, decs):
        allscore_decadal_changes[dec - 1, :, :] = (
            allscore_decades[dec, :, :] - allscore_decades[0, :, :]
        )
        tempscore_decadal_changes[dec - 1, :, :] = (
            tempscore_decades[dec, :, :] - tempscore_decades[0, :, :]
        )
        precscore_decadal_changes[dec - 1, :, :] = (
            precscore_decades[dec, :, :] - precscore_decades[0, :, :]
        )
    allscore_decadal_changes.to_netcdf(
        os.path.join(outdir, cropname + "_decadal_changes.nc")
    )
    tempscore_decadal_changes.to_netcdf(
        os.path.join(outdir, cropname + "_tempscore_decadal_changes.nc")
    )
    precscore_decadal_changes.to_netcdf(
        os.path.join(outdir, cropname + "_precscore_decadal_changes.nc")
    )

    return (
        allscore_decades,
        tempscore_decades,
        precscore_decades,
        allscore_decadal_changes,
        tempscore_decadal_changes,
        precscore_decadal_changes,
    )


def calc_decadal_doy_changes(
    maxdoys, maxdoys_temp, maxdoys_prec, SOIL, LCMloc, sgmloc, cropname, outdir
):
    """
    Calculate decadal changes in the 'day of year of the maximum score' metric,
    using circular averaging

    Inputs
    ------
    maxdoys: Xarray dataarray from calc_maximum_doy containing the day of year
             on which the maximum crop suitability score occured for each
             gridcell for each year.
    maxdoys_temp: As maxdoys but for the temperature crop suitability score
    maxdoys_prec: As maxdoys but for the precipitation crop suitability score
    SOIL: Soil group suitability string from the ecocrop database
    LCMloc: Land cover mask. Path to tif
    sgmloc: Soil group mask netcdfs folder as string
    outdir: Where to store output netcdf files
    cropname: For output filenames

    Outputs
    -------
    maxdoys_decadal_changes: xarray dataarray
        The grid elementwise difference between each decade and the first
        decade of the modulo average day of year denoting the day of year of
        the maximum score for the combined temperature and precipitation crop
        suitability score.
    maxdoys_temp_decadal_changes: As maxdoys_decadal_changes but for the
                                  temperature crop suitability score only
    maxdoys_prec_decadal_changes: As maxdoys_decadal_changes but for the
                                  precipitation crop suitability score only
    """

    # mask land-cover and soil
    lcm = xr.open_dataset(LCMloc, engine="rasterio")
    lcm = lcm["band_data"]
    lcm = lcm.drop("band").squeeze()
    lcm = lcm[::-1, :]
    maxdoys = lcm_mask(lcm, maxdoys)
    maxdoys_temp = lcm_mask(lcm, maxdoys_temp)
    maxdoys_prec = lcm_mask(lcm, maxdoys_prec)
    maxdoys = soil_type_mask_all(maxdoys, SOIL, sgmloc)
    maxdoys_temp = soil_type_mask_all(maxdoys_temp, SOIL, sgmloc)
    maxdoys_prec = soil_type_mask_all(maxdoys_prec, SOIL, sgmloc)
    # compress and save to disk
    maxdoys.encoding["zlib"] = True
    maxdoys.encoding["complevel"] = 1
    maxdoys.encoding["shuffle"] = False
    maxdoys.encoding["contiguous"] = False
    maxdoys.encoding["dtype"] = np.dtype("uint16")
    encoding = {}
    encoding["dayofyear"] = maxdoys.encoding
    maxdoys.to_netcdf(
        os.path.join(outdir, cropname + "_max_score_doys.nc"),
        encoding=encoding,
    )

    maxdoys_temp.encoding["zlib"] = True
    maxdoys_temp.encoding["complevel"] = 1
    maxdoys_temp.encoding["shuffle"] = False
    maxdoys_temp.encoding["contiguous"] = False
    maxdoys_temp.encoding["dtype"] = np.dtype("int16")
    encoding = {}
    encoding["dayofyear"] = maxdoys_temp.encoding
    maxdoys_temp.to_netcdf(
        os.path.join(outdir, cropname + "_max_tempscore_doys.nc"),
        encoding=encoding,
    )

    maxdoys_prec.to_netcdf(
        os.path.join(outdir, cropname + "_max_precscore_doys.nc")
    )
    maxdoys_prec.encoding["zlib"] = True
    maxdoys_prec.encoding["complevel"] = 1
    maxdoys_prec.encoding["shuffle"] = False
    maxdoys_prec.encoding["contiguous"] = False
    maxdoys_prec.encoding["dtype"] = np.dtype("int16")
    encoding = {}
    encoding["dayofyear"] = maxdoys_prec.encoding
    maxdoys_prec.to_netcdf(
        os.path.join(outdir, cropname + "_max_precscore_doys.nc"),
        encoding=encoding,
    )

    # calculate the decadal averages, using circular averaging
    maxdoys_decades = []
    maxdoys_temp_decades = []
    maxdoys_prec_decades = []
    syear = maxdoys["year"][0].values
    nyears = maxdoys.shape[0]
    nyears_u = int(np.floor(nyears / 10) * 10)
    for idx in range(0, nyears_u, 10):
        maxdoys_decade = maxdoys[idx : idx + 10, :, :]
        maxdoys_temp_decade = maxdoys_temp[idx : idx + 10, :, :]
        maxdoys_prec_decade = maxdoys_prec[idx : idx + 10, :, :]
        maxdoys_decade = circular_avg(maxdoys_decade, "year")
        maxdoys_temp_decade = circular_avg(maxdoys_temp_decade, "year")
        maxdoys_prec_decade = circular_avg(maxdoys_prec_decade, "year")
        maxdoys_decade = maxdoys_decade.expand_dims({"decade": [syear + idx]})
        maxdoys_temp_decade = maxdoys_temp_decade.expand_dims(
            {"decade": [syear + idx]}
        )
        maxdoys_prec_decade = maxdoys_prec_decade.expand_dims(
            {"decade": [syear + idx]}
        )
        maxdoys_decades.append(maxdoys_decade)
        maxdoys_temp_decades.append(maxdoys_temp_decade)
        maxdoys_prec_decades.append(maxdoys_prec_decade)
    maxdoys_decades = xr.merge(maxdoys_decades)["dayofyear"]
    maxdoys_temp_decades = xr.merge(maxdoys_temp_decades)["dayofyear"]
    maxdoys_prec_decades = xr.merge(maxdoys_prec_decades)["dayofyear"]
    # save to disk
    maxdoys_decades.to_netcdf(
        os.path.join(outdir, cropname + "_max_score_doys_decades.nc")
    )
    maxdoys_temp_decades.to_netcdf(
        os.path.join(outdir, cropname + "_max_tempscore_doys_decades.nc")
    )
    maxdoys_prec_decades.to_netcdf(
        os.path.join(outdir, cropname + "_max_precscore_doys_decades.nc")
    )

    # calculate the decadal changes from the first decade,
    # using modulo (circular) arithmetic
    maxdoys_decadal_changes = maxdoys_decades.copy()[1:, :, :]
    maxdoys_temp_decadal_changes = maxdoys_temp_decades.copy()[1:, :, :]
    maxdoys_prec_decadal_changes = maxdoys_prec_decades.copy()[1:, :, :]
    decs = maxdoys_decades.shape[0]
    for dec in range(1, decs):
        maxdoys_decadal_changes[dec - 1, :, :] = (
            maxdoys_decades[dec, :, :] - maxdoys_decades[0, :, :]
        )
        maxdoys_temp_decadal_changes[dec - 1, :, :] = (
            maxdoys_temp_decades[dec, :, :] - maxdoys_temp_decades[0, :, :]
        )
        maxdoys_prec_decadal_changes[dec - 1, :, :] = (
            maxdoys_prec_decades[dec, :, :] - maxdoys_prec_decades[0, :, :]
        )
    maxdoys_decadal_changes = xr.where(
        maxdoys_decadal_changes > 180,
        maxdoys_decadal_changes % -180,
        xr.where(
            maxdoys_decadal_changes < -180,
            maxdoys_decadal_changes % 180,
            maxdoys_decadal_changes,
        ),
    )
    maxdoys_temp_decadal_changes = xr.where(
        maxdoys_temp_decadal_changes > 180,
        maxdoys_temp_decadal_changes % -180,
        xr.where(
            maxdoys_temp_decadal_changes < -180,
            maxdoys_temp_decadal_changes % 180,
            maxdoys_temp_decadal_changes,
        ),
    )
    maxdoys_prec_decadal_changes = xr.where(
        maxdoys_prec_decadal_changes > 180,
        maxdoys_prec_decadal_changes % -180,
        xr.where(
            maxdoys_prec_decadal_changes < -180,
            maxdoys_prec_decadal_changes % 180,
            maxdoys_prec_decadal_changes,
        ),
    )
    maxdoys_decadal_changes.to_netcdf(
        os.path.join(outdir, cropname + "_max_score_doys_decadal_changes.nc")
    )
    maxdoys_temp_decadal_changes.to_netcdf(
        os.path.join(
            outdir, cropname + "_max_tempscore_doys_decadal_changes.nc"
        )
    )
    maxdoys_prec_decadal_changes.to_netcdf(
        os.path.join(
            outdir, cropname + "_max_precscore_doys_decadal_changes.nc"
        )
    )
    return (
        maxdoys_decadal_changes,
        maxdoys_temp_decadal_changes,
        maxdoys_prec_decadal_changes,
    )


def calc_decadal_kprop_changes(
    ktmpap, kmaxap, SOIL, LCMloc, sgmloc, cropname, outdir
):
    """
    Calculate decadal changes in the gtime-average proportion of
    ktmp & kmax days for each month


    Inputs
    ------
    ktmpap: Xarray dataarray from containing the gtime-average proportion of
            days within each gtime that are below the crop KTMP, for each day
            and gridcell
    kmaxap: As ktmpap but for above the crop KMAX (TMAX)
    SOIL: Soil group suitability string from the ecocrop database
    LCMloc: Land cover mask. Path to tif
    sgmloc: Soil group mask netcdfs folder as string
    outdir: Where to store output netcdf files
    cropname: For output filenames

    Outputs
    -------
    ktmpap_monavg_climo_diffs: An xarray dataarray containing the
                               difference between the decadally averaged
                               monthly averaged ktmpap and the first decade
    kmaxap_monavg_climo_diffs: As ktmpap_monavg_climo_diffs but for kmaxap
    """

    # Calculate monthly average
    ktmpap_monavg = ktmpap.resample(time="1MS").mean(dim="time")
    kmaxap_monavg = kmaxap.resample(time="1MS").mean(dim="time")

    # mask
    lcm = xr.open_dataset(LCMloc, engine="rasterio")
    lcm = lcm["band_data"]
    lcm = lcm.drop("band").squeeze()
    lcm = lcm[::-1, :]
    ktmpap_monavg = lcm_mask(lcm, ktmpap_monavg)
    kmaxap_monavg = lcm_mask(lcm, kmaxap_monavg)
    ktmpap_monavg = soil_type_mask_all(ktmpap_monavg, SOIL, sgmloc)
    kmaxap_monavg = soil_type_mask_all(kmaxap_monavg, SOIL, sgmloc)

    # calculate monthly climatologies for each decade
    ktmpap_monavg_climos = []
    kmaxap_monavg_climos = []
    nyears = ktmpap_monavg.shape[0] // 12
    ndecs = int(np.round(nyears, -1) / 10)
    for d in range(0, ndecs):
        sind = d * 120
        eind = (d + 1) * 120
        year = ktmpap_monavg["time"][sind].dt.year
        if eind >= ktmpap_monavg.shape[0]:
            ktmpap_monavg_climo = (
                ktmpap_monavg[sind:, :, :]
                .groupby("time.month")
                .mean()
                .expand_dims({"decade": [year]})
            )
            kmaxap_monavg_climo = (
                kmaxap_monavg[sind:, :, :]
                .groupby("time.month")
                .mean()
                .expand_dims({"decade": [year]})
            )
            ktmpap_monavg_climos.append(ktmpap_monavg_climo)
            kmaxap_monavg_climos.append(kmaxap_monavg_climo)
        else:
            ktmpap_monavg_climo = (
                ktmpap_monavg[sind:eind, :, :]
                .groupby("time.month")
                .mean()
                .expand_dims({"decade": [year]})
            )
            kmaxap_monavg_climo = (
                kmaxap_monavg[sind:eind, :, :]
                .groupby("time.month")
                .mean()
                .expand_dims({"decade": [year]})
            )
            ktmpap_monavg_climos.append(ktmpap_monavg_climo)
            kmaxap_monavg_climos.append(kmaxap_monavg_climo)

    ktmpap_monavg_climos2 = xr.concat(ktmpap_monavg_climos, dim="decade")
    kmaxap_monavg_climos2 = xr.concat(kmaxap_monavg_climos, dim="decade")
    # compress and save to disk
    ktmpap_monavg_climos2.encoding["zlib"] = True
    ktmpap_monavg_climos2.encoding["complevel"] = 1
    ktmpap_monavg_climos2.encoding["shuffle"] = False
    ktmpap_monavg_climos2.encoding["contiguous"] = False
    ktmpap_monavg_climos2.encoding["dtype"] = np.dtype("float32")
    encoding = {}
    encoding[
        "average_proportion_of_ktmp_days_in_gtime"
    ] = ktmpap_monavg_climos2.encoding
    ktmpap_monavg_climos2.to_netcdf(
        os.path.join(outdir, cropname + "_ktmpdaysavgprop_decades.nc"),
        encoding=encoding,
    )

    kmaxap_monavg_climos2.encoding["zlib"] = True
    kmaxap_monavg_climos2.encoding["complevel"] = 1
    kmaxap_monavg_climos2.encoding["shuffle"] = False
    kmaxap_monavg_climos2.encoding["contiguous"] = False
    kmaxap_monavg_climos2.encoding["dtype"] = np.dtype("float32")
    encoding = {}
    encoding[
        "average_proportion_of_kmax_days_in_gtime"
    ] = kmaxap_monavg_climos2.encoding
    kmaxap_monavg_climos2.to_netcdf(
        os.path.join(outdir, cropname + "_kmaxdaysavgprop_decades.nc"),
        encoding=encoding,
    )

    # difference the climatologies
    ktmpap_monavg_climo_diffs = ktmpap_monavg_climos2.copy()[1:]
    kmaxap_monavg_climo_diffs = kmaxap_monavg_climos2.copy()[1:]
    decs = kmaxap_monavg_climos2.shape[0]
    for dec in range(1, decs):
        ktmpap_monavg_climo_diffs[dec - 1] = (
            ktmpap_monavg_climos2[dec] - ktmpap_monavg_climos2[0]
        )
        kmaxap_monavg_climo_diffs[dec - 1] = (
            kmaxap_monavg_climos2[dec] - kmaxap_monavg_climos2[0]
        )

    # compress and save to disk
    ktmpap_monavg_climo_diffs.encoding["zlib"] = True
    ktmpap_monavg_climo_diffs.encoding["complevel"] = 1
    ktmpap_monavg_climo_diffs.encoding["shuffle"] = False
    ktmpap_monavg_climo_diffs.encoding["contiguous"] = False
    ktmpap_monavg_climo_diffs.encoding["dtype"] = np.dtype("float32")
    encoding = {}
    encoding[
        "average_proportion_of_ktmp_days_in_gtime"
    ] = ktmpap_monavg_climo_diffs.encoding
    ktmpap_monavg_climo_diffs.to_netcdf(
        os.path.join(outdir, cropname + "_ktmpdaysavgprop_decadal_changes.nc"),
        encoding=encoding,
    )

    kmaxap_monavg_climo_diffs.encoding["zlib"] = True
    kmaxap_monavg_climo_diffs.encoding["complevel"] = 1
    kmaxap_monavg_climo_diffs.encoding["shuffle"] = False
    kmaxap_monavg_climo_diffs.encoding["contiguous"] = False
    kmaxap_monavg_climo_diffs.encoding["dtype"] = np.dtype("float32")
    encoding = {}
    encoding[
        "average_proportion_of_kmax_days_in_gtime"
    ] = kmaxap_monavg_climo_diffs.encoding
    kmaxap_monavg_climo_diffs.to_netcdf(
        os.path.join(outdir, cropname + "_kmaxdaysavgprop_decadal_changes.nc"),
        encoding=encoding,
    )

    return ktmpap_monavg_climo_diffs, kmaxap_monavg_climo_diffs


def plot_decade(allscore, tempscore, precscore, save=None):
    """
    Plot a given decade's allscore, tempscore and precscore

    Inputs
    ------
    allscore : xarray dataarray/set
        A given decade's gridded crop suitability score.
    tempscore : xarray dataarray/set
        A given decade's gridded temperature suitability score.
    precscore : xarray dataarray/set
        A given decade's gridded precipitation suitability score.
    save : boolean, optional
        Controls whether or not the plot is saved to disk. The default is None.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(1, 3, subplot_kw={"projection": cp.crs.OSGB()})
    fig.set_figwidth(10)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.coastlines(resolution="10m")
    ax2.coastlines(resolution="10m")
    ax3.coastlines(resolution="10m")

    allscore.where(allscore > 0).plot(ax=ax1, vmin=0, vmax=100)
    tempscore.where(tempscore > 0).plot(ax=ax2, vmin=0, vmax=100)
    precscore.where(precscore > 0).plot(ax=ax3, vmin=0, vmax=100)

    cbarax1 = ax1.collections[0].colorbar.ax
    cbarax2 = ax2.collections[0].colorbar.ax
    cbarax3 = ax3.collections[0].colorbar.ax
    cbarax1.set_ylabel("")
    cbarax2.set_ylabel("")
    cbarax3.set_ylabel("")

    ax1.set_title("crop_suitability")
    ax2.set_title("temperature_suitability")
    ax3.set_title("precip_suitability")

    if not save == None:
        savedir = os.path.dirname(save)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(save, dpi=300)
        plt.close()


def plot_decadal_changes(
    dcdata, save=None, cmin=None, cmax=None, revcolbar=None
):
    """
    Produce a plot of the decadally averaged crop suitability score data
    for a given crop. Produces a 1x3 plot of the first, third and fifth
    decades respectively.

    Inputs
    ------
    dcdata : xarray dataarray/set
        A given crop's decadally averaged gridded suitability score.
    save : boolean, optional
        Controls whether or not the plot is saved to disk. The default is None.
    cmin : boolean, optional
        Minimum value of the colourbar to use when plotting. The default is
        None, which automatically sets the colourbar values
    cmax : boolean, optional
        Maximum value of the colourbar to use when plotting. The default is
        None, which automatically sets the colourbar values
    revcolbar : boolean, optional
        Controls which way around the blue,white,red colourbar runs. The
        default is None which used the 'bwr_r' matplotlib colourbar.

    Returns
    -------
    None.

    """
    if not cmax:
        cmax2 = np.ceil(dcdata.max().values)
        if cmax2 not in (0.0, 1.0):
            cmax = cmax2
        else:
            cmax = dcdata.max().values
        print(cmax)
    if not cmin:
        cmin2 = np.floor(dcdata.min().values)
        if cmin2 not in (0.0, -1.0):
            cmin = cmin2
        else:
            cmin = dcdata.min().values
        print(cmin)

    if abs(cmax) > abs(cmin):
        cmin = -1 * cmax
    else:
        cmax = -1 * cmin

    fig, axs = plt.subplots(1, 3, subplot_kw={"projection": cp.crs.OSGB()})
    fig.set_figwidth(10)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.coastlines(resolution="10m")
    ax2.coastlines(resolution="10m")
    ax3.coastlines(resolution="10m")

    # get colourbar limits
    # dcdata[0,:,:].plot(ax=ax1, robust=True)
    # vmax = ax1.collections[0].colorbar.vmax
    # vmin = ax1.collections[0].colorbar.vmin
    # ax1.collections[0].colorbar.remove()

    # ax1.pcolormesh(dcdata['x'].values, dcdata['y'].values, dcdata[0,:,:].values, cmap='bwr_r', vmin=cmin, vmax=cmax)
    # ax2.pcolormesh(dcdata['x'].values, dcdata['y'].values, dcdata[2,:,:].values, cmap='bwr_r', vmin=cmin, vmax=cmax)
    # c = ax3.pcolormesh(dcdata['x'].values, dcdata['y'].values, dcdata[4,:,:].values, cmap='bwr_r', vmin=cmin, vmax=cmax)
    if not revcolbar:
        cmap = "bwr_r"
    else:
        cmap = "bwr"
    dcdata[0, :, :].plot(ax=ax1, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata[2, :, :].plot(ax=ax2, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata[4, :, :].plot(ax=ax3, vmin=cmin, vmax=cmax, cmap=cmap)

    cbartext = dcdata.name + "_change"
    cbarax1 = ax1.collections[0].colorbar.ax
    cbarax2 = ax2.collections[0].colorbar.ax
    cbarax3 = ax3.collections[0].colorbar.ax
    cbarax1.set_ylabel("")
    cbarax2.set_ylabel("")
    cbarax3.set_ylabel(cbartext)
    # cbar = plt.colorbar(c)
    # cbar.ax.yaxis.set_label_text(cbartext)

    ax1.set_title("2030s")
    ax2.set_title("2050s")
    ax3.set_title("2070s")

    # fig.subplots_adjust(wspace=-0.1)

    if not save == None:
        savedir = os.path.dirname(save)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(save, dpi=300)
        plt.close()


def plot_degC_changes(
    dcdata, savedir=None, cmin=None, cmax=None, revcolbar=None
):
    """
    Produce a plot of the degC differences in the crop suitability score
    data for a given crop. Produces a 1x3 plot of the 2C, 3C and 4C
    differences from baseline respectively.

    Inputs
    ------
    dcdata : xarray dataarray/set
        A given crop's degC differenced gridded suitability score.
    save : boolean, optional
        Controls whether or not the plot is saved to disk. The default is None.
    cmin : boolean, optional
        Minimum value of the colourbar to use when plotting. The default is
        None, which automatically sets the colourbar values
    cmax : boolean, optional
        Maximum value of the colourbar to use when plotting. The default is
        None, which automatically sets the colourbar values
    revcolbar : boolean, optional
        Controls which way around the blue,white,red colourbar runs. The
        default is None which used the 'bwr_r' matplotlib colourbar.

    Returns
    -------
    None.

    """

    if not cmax:
        cmax2 = np.ceil(dcdata.max().values)
        if cmax2 not in (0.0, 1.0):
            cmax = cmax2
        else:
            cmax = dcdata.max().values

    if not cmin:
        cmin2 = np.floor(dcdata.min().values)
        if cmin2 not in (0.0, -1.0):
            cmin = cmin2
        else:
            cmin = dcdata.min().values

    if abs(cmax) > abs(cmin):
        cmin = -1 * cmax
    else:
        cmax = -1 * cmin

    fig, axs = plt.subplots(1, 3, subplot_kw={"projection": cp.crs.OSGB()})
    fig.set_figwidth(10)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.coastlines(resolution="10m")
    ax2.coastlines(resolution="10m")
    ax3.coastlines(resolution="10m")

    if not revcolbar:
        cmap = "bwr_r"
    else:
        cmap = "bwr"
    dcdata.sel(deg="2C").plot(ax=ax1, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata.sel(deg="3C").plot(ax=ax2, vmin=cmin, vmax=cmax, cmap=cmap)
    dcdata.sel(deg="4C").plot(ax=ax3, vmin=cmin, vmax=cmax, cmap=cmap)

    cbartext = dcdata.name + "_change"
    cbarax1 = ax1.collections[0].colorbar.ax
    cbarax2 = ax2.collections[0].colorbar.ax
    cbarax3 = ax3.collections[0].colorbar.ax
    cbarax1.set_ylabel("")
    cbarax2.set_ylabel("")
    cbarax3.set_ylabel(cbartext, fontsize="x-small")
    cbarax1.tick_params(axis="y", labelsize=7)
    cbarax2.tick_params(axis="y", labelsize=7)
    cbarax3.tick_params(axis="y", labelsize=7)

    cropname = str(dcdata.crop.values)
    ax1.set_title(
        "Suitability change for " + cropname + " at 2C", size="x-small"
    )
    ax2.set_title(
        "Suitability change for " + cropname + " at 3C", size="x-small"
    )
    ax3.set_title(
        "Suitability change for " + cropname + " at 4C", size="x-small"
    )

    if not savedir == None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plotname = cropname + "_degC_changes.png"
        savepath = os.path.join(savedir, plotname)
        plt.savefig(savepath, dpi=600)
        plt.close()


def frs3D(ind, window, dtype):
    """
    Clever function that does a forward rolling sum without loops
    see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    can't be used with numba as it doesn't support cumsum on n-d arrays


    Parameters
    ----------
    ind : array-like
        3-dimensional array over which to calculate the rolling sum.
    window : int
        The window size to use for the rolling sum.
    dtype : np.dtype
        The dtype to output the result as.

    Returns
    -------
    ret : array-like
        3-dimensional array, the result of the forward rolling sum over ind.

    """
    inds = np.cumsum(ind, axis=0, dtype=dtype)
    tmp = inds[window:, ...] - inds[:-window, ...]
    tmp2 = inds[window - 1, ...]
    ret = np.concatenate([tmp2[None, ...], tmp], axis=0)
    return ret


def frs3Dwcs(ind, window):
    """
    As frs3D, but without the initial step
    """
    tmp = ind[window:, ...] - ind[:-window, ...]
    tmp2 = ind[window - 1, ...]
    ret = np.concatenate([tmp2[None, ...], tmp], axis=0)
    return ret


# @njit(parallel=True)
def score_temp(gtime, gmin, gmax):
    """
    Function to calculate the temperature suitability score for a given crop
    for the 'annual' method.

    Parameters
    ----------
    gtime : int
        The growing season length under consideration.
    gmin : int
        The minimum growing season length of the crop.
    gmax : int
        The maximum growing season length of the crop.

    Returns
    -------
    uint8
        The temperature suitability score.

    """
    score = 100 * (1 - ((gtime - gmin) / (gmax - gmin)))
    return np.round(score).astype("uint8")


# @njit(parallel=True)
# def score_prec(total, pmin, pmax, popmin, popmax):
#    score = xr.where(total > popmax, (100/(pmax-popmax))*(pmax-total), \
#            xr.where(total > popmin, 100, \
#            (100/(popmin-pmin))*(total-pmin)))
#    return score.round()


def score_temp2(temp, tmin, tmax, topmin, topmax):
    """
    Calculate the temperature suitability of a given day in the driving dataset
    for a given crop, between 0 and 1.

    Parameters
    ----------
    temp : array-lkike
        The 3D temperature dataset.
    tmin : int
        The minimum suitable temperature for the crop.
    tmax : int
        The maximum suitable temperature for the crop.
    topmin : int
        The minimum optimum temperature for the crop.
    topmax : int
        The maximum optimum temperature for the crop.

    Returns
    -------
    array-like, float16
        The temperature suitability for each day and grid cell between 0 and 1.

    """
    tmin = tmin.astype("float32")
    tmax = tmax.astype("float32")
    topmin = topmin.astype("float32")
    topmax = topmax.astype("float32")
    score = xr.where(
        temp > tmax,
        0,
        xr.where(
            temp > topmax,
            (tmax - temp) / (tmax - topmax),
            xr.where(
                temp > topmin,
                1,
                xr.where(temp > tmin, (temp - tmin) / (topmin - tmin), 0),
            ),
        ),
    )
    return score.astype("float16")


# Not used currently
def score_temp3(avgt, tmin, tmax, topmin, topmax):
    tmin = tmin.astype("float32")
    tmax = tmax.astype("float32")
    topmin = topmin.astype("float32")
    topmax = topmax.astype("float32")
    score = xr.where(
        avgt > tmax,
        0,
        xr.where(
            avgt > topmax,
            (100 / (tmax - topmax)) * (tmax - avgt),
            xr.where(
                avgt > topmin,
                1,
                xr.where(
                    avgt > tmin, (100 / (topmin - tmin)) * (avgt - tmin), 0
                ),
            ),
        ),
    )
    return score.round().astype("uint8")


def score_temp4(avgt, tmin, tmax, topmin, topmax):
    """
    Function to calculate the temperature suitability score for a given crop
    for the 'perennial' method.

    Parameters
    ----------
    avgt : array-like
        The average temperature over the growing season length in question,
        starting on each day and for each gridcell
    tmin : int
        The minimum suitable temperature for the crop.
    tmax : int
        The maximum suitable temperature for the crop.
    topmin : int
        The minimum optimum temperature for the crop.
    topmax : int
        The maximum optimum temperature for the crop.

    Returns
    -------
    uint8
        The temperature suitability score.

    """
    tmin = tmin.astype("float32")
    tmax = tmax.astype("float32")
    topmin = topmin.astype("float32")
    topmax = topmax.astype("float32")
    score = xr.where(
        avgt > tmax,
        0,
        xr.where(
            avgt > 0.5 * (topmax + topmin),
            (100 / (tmax - 0.5 * (topmax + topmin))) * (tmax - avgt),
            xr.where(
                avgt > tmin,
                (100 / (0.5 * (topmax + topmin) - tmin)) * (avgt - tmin),
                0,
            ),
        ),
    )
    return score.round().astype("uint8")


def score_prec1(total, pmin, pmax, popmin, popmax):
    """
    Method 1 for scoring the precipitation suitability of a given crop.
    Not recommended.

    Parameters
    ----------
    total : array-like
        The precipitation totals for the growing season length in question, for
        each day and gridcell. 3D array.
    pmin : int or float
        The minimum suitable precipitation for the crop.
    pmax : int or float
        The maximum suitable precipitation for the crop.
    popmin : int or float
        The minimum optimum precipitation for the crop.
    popmax : int or float
        The maximum optimum precipitation for the crop.

    Returns
    -------
    array-like, dtype uint8
        The precipitation score for each day and grid cell.

    """
    pmin = pmin.astype("float32")
    pmax = pmax.astype("float32")
    popmin = popmin.astype("float32")
    popmax = popmax.astype("float32")
    score = xr.where(
        total > pmax,
        0,
        xr.where(
            total > popmax,
            (100 / (pmax - popmax)) * (pmax - total),
            xr.where(
                total > popmin,
                100,
                xr.where(
                    total > pmin, (100 / (popmin - pmin)) * (total - pmin), 0
                ),
            ),
        ),
    )
    return score.round().astype("uint8")


def score_prec2(total, pmin, pmax, popmin, popmax):
    """
    Method 2 for scoring the precipitation suitability of a given crop.
    The recommended method.

    Parameters
    ----------
    total : array-like
        The precipitation totals for the growing season length in question, for
        each day and gridcell. 3D array.
    pmin : int or float
        The minimum suitable precipitation for the crop.
    pmax : int or float
        The maximum suitable precipitation for the crop.
    popmin : int or float
        The minimum optimum precipitation for the crop.
    popmax : int or float
        The maximum optimum precipitation for the crop.

    Returns
    -------
    array-like, dtype uint8
        The precipitation score for each day and grid cell.

    """
    pmin = pmin.astype("float32")
    pmax = pmax.astype("float32")
    popmin = popmin.astype("float32")
    popmax = popmax.astype("float32")
    score = xr.where(
        total > pmax,
        0,
        xr.where(
            total > 0.5 * (popmax + popmin),
            (200 / (2 * pmax - popmin - popmax)) * (pmax - total),
            xr.where(
                total > pmin,
                (200 / (popmin + popmax - 2 * pmin)) * (total - pmin),
                0,
            ),
        ),
    )
    return score.round().astype("uint8")


def score_prec3(total, pmin, pmax, popmin, popmax):
    """
    Method 3 for scoring the precipitation suitability of a given crop.
    Not recommended.

    Parameters
    ----------
    total : array-like
        The precipitation totals for the growing season length in question, for
        each day and gridcell. 3D array.
    pmin : int or float
        The minimum suitable precipitation for the crop.
    pmax : int or float
        The maximum suitable precipitation for the crop.
    popmin : int or float
        The minimum optimum precipitation for the crop.
    popmax : int or float
        The maximum optimum precipitation for the crop.

    Returns
    -------
    array-like, dtype uint8
        The precipitation score for each day and grid cell.

    """
    pmin = pmin.astype("float32")
    pmax = pmax.astype("float32")
    popmin = popmin.astype("float32")
    popmax = popmax.astype("float32")
    score = xr.where(
        total > pmax,
        0,
        xr.where(
            total > popmax,
            50 * ((popmax - total) / (pmax - popmax) + 1),
            xr.where(
                total > 0.5 * (popmax + popmin),
                50 * ((2 * (popmax - total)) / (popmax - popmin) + 1),
                xr.where(
                    total > popmin,
                    50 * ((2 * (total - popmin)) / (popmax - popmin) + 1),
                    xr.where(
                        total > pmin,
                        50 * ((total - popmin) / (popmin - pmin) + 1),
                        0,
                    ),
                ),
            ),
        ),
    )
    return score.round().astype("uint8")
