import sys
from ecocrop_utils import (
    calc_yearly_scores_only,
    frs3D,
    score_temp,
    score_temp2,
    score_temp4,
    score_prec1,
    score_prec2,
    score_prec3,
    plot_year,
)
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import os

#######################################################
# Setup
#######################################################
"""
Inputs:

cropind: ------ integer
                Index of ecocroploc to use. Determines the crop
                that is run
rcp: ---------- string
                Relative Concentration Pathway version of the
                driving meteorological data to use. Options are
                "85" or "26". Determines the path to and
                filenames of the driving data used. Only "85" is
                available in the test script
ensmem: ------- string
                As rcp, but for the ensemble member. Options are
                "01", "04", "06", "15". Only "01" is available in
                the test script
pf: ----------- string
                "before", "after" or "". If the data spans
                multiple years including 2020, "before" only runs
                for years before 2020 and "after", after 2020.
                "" runs all years. Only "" is available in the
                test script
ecocroploc: --- string
                Path to EcoCrop csv database containing the crop
                indices
tasvname: ----- string
                Variable name of the daily average temperature in
                the meterological driving data
prevname: ----- As tasvname but for daily precipitation totals
tmnvname: ----- As tasvname but for daily minimum temperature
tmxvname: ----- As tasvname but for daily maximum temperature
lcmloc: ------- string
                Path to land cover map file for masking
bgsloc: ------- string
                Path to soil texture maps for masking
savedir: ------ string
                Path to save netcdf outputs in
plotdir: ------ string
                Path to save output plots in
yearaggmethod : string
                Method to use to aggregate the daily scores
                to yearly scores. Available options are
                "mean", "max", "min", "percentile".
                "percentile" is recommended and uses the
                95th percentile.
precmethod: --- integer
                The method to use to calculate the
                precipitation suitability score.
                Available options 1, 2 or 3. 2 is recommended,
                and used in the documented results
verify: ------- integer
                Switch to enable verfication of results against
                existing files. Only available for wheat crop,
                cropind 117, yearaggmethod "percentile",
                precmethod 2.
"""

cropind = 117
rcp = "85"
ensmem = "01"
pf = ""
method = "annual"
ecocroploc = "./EcoCrop_DB_secondtrim.csv"
tasvname = "tas"
prevname = "pr"
tmnvname = "tasmin"
tmxvname = "tasmax"
lcmloc = "./Mask_arable_LCM2015_UK.tif"
bgsloc = "./EU_STM_soildata"
savedir = "./testoutputs"
plotdir = "./testoutputs"
yearaggmethod = "percentile"
precmethod = 2
verify = 1

taspath = (
    "./testdata/tas/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_tas_uk_1km_daily_????????-????????.nc"
)
prepath = (
    "./testdata/pr/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_pr_uk_1km_daily_????????-????????.nc"
)
tmnpath = (
    "./testdata/tasmin/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_tasmin_uk_1km_daily_????????-????????.nc"
)
tmxpath = (
    "./testdata/tasmax/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_tasmax_uk_1km_daily_????????-????????.nc"
)

#######################################################
# Main script
#######################################################

# Read in ecocrop database and select out indices for
# the crop, and convert the units
ecocropall = pd.read_csv(ecocroploc, engine="python")
ecocrop = ecocropall.drop(["level_0"], axis=1)
print("Cropind: " + str(cropind))
testcrop = ecocrop.iloc[
    cropind, :
]  # 19 onions, #117 wheat, #147 chickpea, #66 sweet potato
TOPMIN = testcrop["TOPMN"] + 273.15  # C-->K
TOPMAX = testcrop["TOPMX"] + 273.15  # C-->K
TMIN = testcrop["TMIN"] + 273.15  # C-->K
TMAX = testcrop["TMAX"] + 273.15  # C-->K
PMIN = testcrop["RMIN"] / 86400.0  # mm-->kg/m^2/s
PMAX = testcrop["RMAX"] / 86400.0  # mm-->kg/m^2/s
POPMIN = testcrop["ROPMN"] / 86400.0  # mm-->kg/m^2/s
POPMAX = testcrop["ROPMX"] / 86400.0  # mm-->kg/m^2/s
KTMP = testcrop["KTMPR"] + 273.15  # C-->K
KMAX = testcrop["TMAX"] + 273.15  # C-->K
GMIN = int(testcrop["GMIN"])
GMAX = int(testcrop["GMAX"])
SOIL = testcrop["TEXT"]
COMNAME = testcrop["COMNAME"]
COMNAME = "_".join(COMNAME.split(",")[0].split(" "))
if "(" in COMNAME:
    COMNAME = "".join(COMNAME.split("("))
    COMNAME = "".join(COMNAME.split(")"))
if "'" in COMNAME:
    COMNAME = "".join(COMNAME.split("'"))
cropname = COMNAME

# Check for missing data
if np.isnan(testcrop["TOPMN"]):
    raise ValueError("Missing TOPMN")
if np.isnan(testcrop["TOPMX"]):
    raise ValueError("Missing TOPMX")
if np.isnan(testcrop["TMIN"]):
    raise ValueError("Missing TMIN")
if np.isnan(testcrop["TMAX"]):
    raise ValueError("Missing TMAX (KMAX)")
if np.isnan(testcrop["RMIN"]):
    raise ValueError("Missing RMIN")
if np.isnan(testcrop["RMAX"]):
    raise ValueError("Missing RMAX")
if np.isnan(testcrop["ROPMN"]):
    raise ValueError("Missing ROPMN")
if np.isnan(testcrop["ROPMX"]):
    raise ValueError("Missing ROPMX")
if np.isnan(testcrop["GMIN"]):
    raise ValueError("Missing GMIN")
if np.isnan(testcrop["GMAX"]):
    raise ValueError("Missing GMAX")

# exit if GMIN=GMAX, assume missing data
if GMAX - GMIN <= 10:
    raise ValueError(
        "GMIN and GMAX too close, not enough info to calculate suitability"
    )

# assume killing temp of -1 if not specified
if np.isnan(KTMP):
    KTMP = -1

GMIN = int(GMIN)
GMAX = int(GMAX)
print("TMN: " + str(testcrop["TMIN"]))
print("TMX: " + str(testcrop["TMAX"]))
print("TOPMN: " + str(testcrop["TOPMN"]))
print("TOPMX: " + str(testcrop["TOPMX"]))
print("KTMP: " + str(testcrop["KTMPR"]))
print("KMAX: " + str(testcrop["TMAX"]))
print("GMIN: " + str(testcrop["GMIN"]))
print("GMAX: " + str(testcrop["GMAX"]))
print("PMIN: " + str(testcrop["RMIN"]))
print("PMAX: " + str(testcrop["RMAX"]))
print("POPMN: " + str(testcrop["ROPMN"]))
print("POPMX: " + str(testcrop["ROPMX"]))
print("SOIL: " + str(SOIL))

if not os.path.exists(savedir):
    os.makedirs(savedir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# open datafiles
print("Reading in met data")
print("Start: " + str(dt.datetime.now()))
tas = xr.open_mfdataset(taspath).astype("float16")[tasvname]
tmn = xr.open_mfdataset(tmnpath).astype("float16")[tmnvname]
tmx = xr.open_mfdataset(tmxpath).astype("float16")[tmxvname]
pre = xr.open_mfdataset(prepath)[prevname]
tas = tas.load()
tmn = tmn.load()
tmx = tmx.load()
pre = pre.load()

tastime = tas["time"]
tasy = tas["y"]
tasx = tas["x"]

if method == "perennial":
    tas = tas.values
print("End: " + str(dt.datetime.now()))

# Calculate the days within the crop temperature range,
# below the killing temperature and above the
# maximum temperature
print("Calculating topt_, ktmp_ and kmax_crop")
print("Start: " + str(dt.datetime.now()))
if method == "annual":
    topt_crop = score_temp2(tas, TMIN, TMAX, TOPMIN, TOPMAX).values
ktmp_crop = xr.where(tmn < KTMP, 1, 0).astype("uint16").values
kmax_crop = xr.where(tmx > KMAX, 1, 0).astype("uint16").values
print("End: " + str(dt.datetime.now()))

# Determine growing season lengths to assess
# Intervals of 10 days are used to reduce
# computational cost
if GMAX - GMIN <= 15:
    gstart = np.int16(np.floor(GMIN / 10) * 10)
else:
    gstart = np.int16(np.ceil(GMIN / 10) * 10)
gend = np.int16(np.ceil(GMAX / 10) * 10)
allgtimes = list(np.arange(gstart, gend, 10, dtype="int16"))

counter = 1
GMIN = np.uint16(GMIN)
GMAX = np.uint16(GMAX)
# Loop over each growing season length
for gtime in allgtimes:
    print(
        "Calculating suitability for "
        + cropname
        + " for a growing season of length "
        + str(gtime)
        + " out of a maximum of "
        + str(int(GMAX))
    )
    print("Start: " + str(dt.datetime.now()))

    # Calculate temperature suitability score
    print("Calculating T suitability")
    if method == "annual":
        tscore1 = score_temp(gtime, GMIN, GMAX).astype("uint8")
    # calculate ndays of T in optimal/suitable range within gtime
    tcoords_tas = tastime[: -gtime + 1]
    ycoords_tas = tasy
    xcoords_tas = tasx
    if method == "annual":
        toptdays = (frs3D(topt_crop, gtime, "float32")).round().astype("uint16")
    elif method == "perennial":
        toptdays = (frs3D(tas, gtime, "float32") / gtime).round().astype("uint16")
    toptdays = xr.DataArray(toptdays, coords=[tcoords_tas, ycoords_tas, xcoords_tas])
    toptdays.name = "TOPT_days"
    if method == "annual":
        tscore = xr.where(toptdays >= GMIN, tscore1, np.uint8(0))
    elif method == "perennial":
        tscore = score_temp4(toptdays, TMIN, TMAX, TOPMIN, TOPMAX)
    print("End: " + str(dt.datetime.now()))

    # calculate whether any of the suitable days/locations identified above will have
    # frost/killing temp within gtime
    print("Calculating frost days and their proportions")
    print("Start: " + str(dt.datetime.now()))
    tcoords_tmn = tmn["time"][: -gtime + 1]
    ycoords_tmn = tmn["y"]
    xcoords_tmn = tmn["x"]
    ktmp_days = frs3D(ktmp_crop, gtime, "uint16")
    ktmp_days = xr.DataArray(ktmp_days, coords=[tcoords_tmn, ycoords_tmn, xcoords_tmn])
    ktmp_days.name = "KTMP_days"
    print("End: " + str(dt.datetime.now()))

    # calculate whether any of the suitable days/locations identified above will have
    # heat killing temp within gtime
    print("Calculating heat-stress days and their proportions")
    print("Start: " + str(dt.datetime.now()))
    sys.stdout.flush()
    tcoords_tmx = tmx["time"][: -gtime + 1]
    ycoords_tmx = tmx["y"]
    xcoords_tmx = tmx["x"]
    kmax_days = frs3D(kmax_crop, gtime, "uint16")
    kmax_days = xr.DataArray(kmax_days, coords=[tcoords_tmx, ycoords_tmx, xcoords_tmx])
    kmax_days.name = "KMAX_days"
    print("End: " + str(dt.datetime.now()))

    # calculate total precipitation in gtime
    print("Calculating total precipitation")
    print("Start: " + str(dt.datetime.now()))
    sys.stdout.flush()
    tcoords_pre = pre["time"][: -gtime + 1]
    ycoords_pre = pre["y"]
    xcoords_pre = pre["x"]
    pre2 = pre.values
    precip_crop = frs3D(pre2, gtime, "float32")
    precip_crop = xr.DataArray(
        precip_crop, coords=[tcoords_pre, ycoords_pre, xcoords_pre]
    )
    precip_crop.name = "precip_total"
    print("End: " + str(dt.datetime.now()))

    print("Processing KTMP")
    print("Start: " + str(dt.datetime.now()))
    tempscore = xr.where(ktmp_days > np.uint8(0), np.uint8(0), tscore)
    print("End: " + str(dt.datetime.now()))

    print("Processing KMAX days penalty")
    print("Start: " + str(dt.datetime.now()))
    tempscore = tempscore - np.int8(kmax_days)
    tempscore = xr.where(tempscore < 0, 0, tempscore).astype("uint8")
    print("End: " + str(dt.datetime.now()))

    # Calculate precipitation suitability score
    print("Calculating precip suitability score using method " + str(precmethod))
    print("Start: " + str(dt.datetime.now()))
    if precmethod == 1:
        precscore = score_prec1(precip_crop, PMIN, PMAX, POPMIN, POPMAX)
    elif precmethod == 2:
        precscore = score_prec2(precip_crop, PMIN, PMAX, POPMIN, POPMAX)
    elif precmethod == 3:
        precscore = score_prec3(precip_crop, PMIN, PMAX, POPMIN, POPMAX)
    else:
        raise ValueError(
            "precmethod must be 1, 2 or 3. Currently set as " + str(precmethod)
        )
    print("End: " + str(dt.datetime.now()))

    # Always take the highest of the growing season scores as this
    # is the growing season length the crop will likely grow in
    print("Updating T & P suitability scores for this gtime")
    print("Start: " + str(dt.datetime.now()))
    if counter == 1:
        tempscore_old = tempscore
        precscore_old = precscore
    else:
        if len(tempscore_old["time"]) > len(tempscore["time"]):
            tempscore_old = tempscore_old.sel(
                time=slice(tempscore["time"][0], tempscore["time"][-1])
            )
        if len(precscore_old["time"]) > len(precscore["time"]):
            precscore_old = precscore_old.sel(
                time=slice(precscore["time"][0], precscore["time"][-1])
            )
        tempscore = xr.where(
            tempscore > tempscore_old, tempscore, tempscore_old
        )  # .astype('uint8')
        precscore = xr.where(
            precscore > precscore_old, precscore, precscore_old
        )  # .astype('uint8')
        tempscore_old = tempscore
        precscore_old = precscore

        print("End: " + str(dt.datetime.now()))
    counter += 1

# Combine the temperature and precipitation suitability scores
# by taking the minimum, as this will likely be the
# constraining factor on any crop growth
print("Calculating final combined crop suitability score")
print("Start: " + str(dt.datetime.now()))
final_score_crop = xr.where(precscore < tempscore, precscore, tempscore)
print(final_score_crop.dtype)
print("End: " + str(dt.datetime.now()))

# Save outputs to file
print("Saving to netcdf")
print("Start: " + str(dt.datetime.now()))
final_score_crop.name = "crop_suitability_score"
final_score_crop.encoding["zlib"] = True
final_score_crop.encoding["complevel"] = 1
final_score_crop.encoding["shuffle"] = False
final_score_crop.encoding["contiguous"] = False
final_score_crop.encoding["dtype"] = np.dtype("uint8")
encoding = {}
encoding["crop_suitability_score"] = final_score_crop.encoding
final_score_crop.to_netcdf(os.path.join(savedir, cropname + ".nc"), encoding=encoding)

tempscore.name = "temperature_suitability_score"
tempscore.encoding["zlib"] = True
tempscore.encoding["complevel"] = 1
tempscore.encoding["shuffle"] = False
tempscore.encoding["contiguous"] = False
tempscore.encoding["dtype"] = np.dtype("uint8")
encoding = {}
encoding["temperature_suitability_score"] = tempscore.encoding
tempscore.to_netcdf(os.path.join(savedir, cropname + "_temp.nc"), encoding=encoding)

precscore.name = "precip_suitability_score"
precscore.encoding["zlib"] = True
precscore.encoding["complevel"] = 1
precscore.encoding["shuffle"] = False
precscore.encoding["contiguous"] = False
precscore.encoding["dtype"] = np.dtype("uint8")
encoding = {}
encoding["precip_suitability_score"] = precscore.encoding
precscore.to_netcdf(os.path.join(savedir, cropname + "_prec.nc"), encoding=encoding)

# calculate yearly scores
print("Calculating yearly scores")
(allscore_years, tempscore_years, precscore_years) = calc_yearly_scores_only(
    tempscore,
    precscore,
    str(SOIL),
    lcmloc,
    bgsloc,
    cropname,
    savedir,
    yearaggmethod,
)

# verify
if verify == 1:
    try:
        testall = xr.open_dataarray(os.path.join(verifypath, cropname + "_years.nc"))
        testtemp = xr.open_dataarray(
            os.path.join(verifypath, cropname + "_tempscore_years.nc")
        )
        testprec = xr.open_dataarray(
            os.path.join(verifypath, cropname + "_precscore_years.nc")
        )
        assert np.all(
            testall == allscore_years.astype("uint8")
        ), "Output is different to verified file"
        assert np.all(
            testtemp == tempscore_years.astype("uint8")
        ), "Output is different to verified file"
        assert np.all(
            testprec == precscore_years.astype("uint8")
        ), "Output is different to verified file"
    except FileNotFoundError:
        print("Verification files not available, not doing output verification")
        continue

# plot first year's scores
plot_year(
    allscore_years[0, :, :],
    tempscore_years[0, :, :],
    precscore_years[0, :, :],
    save=os.path.join(plotdir, cropname + "_2020.png"),
)
