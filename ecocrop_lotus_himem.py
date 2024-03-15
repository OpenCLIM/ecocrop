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

cropind = int(sys.argv[1])
rcp = sys.argv[2]  # '85' or '26'
ensmem = sys.argv[3]  # '01', '04', '06' or '15
pf = sys.argv[4]  # 'past' or 'future'
method = sys.argv[5]  # 'annual' or 'perennial'
ecocroploc = "/gws/nopw/j04/ceh_generic/matbro/ecocrop/EcoCrop_DB_secondtrim.csv"
tasvname = "tas"
prevname = "pr"
tmnvname = "tasmin"
tmxvname = "tasmax"
lcmloc = "/gws/nopw/j04/ceh_generic/matbro/ecocrop/Mask_arable_LCM2015_UK.tif"
bgsloc = "/gws/nopw/j04/ceh_generic/matbro/ecocrop/EU_STM_soildata"

if pf == "past":
    ab = "b2020"
elif pf == "future":
    ab = "a2020"
else:
    ab = ""

if rcp in ["85", "26"]:
    rcp2 = rcp + "/"
else:
    rcp = ""
    rcp2 = ""

if ensmem in ["01", "04", "06", "15"]:
    ensmem2 = ensmem + "/"
else:
    ensmem = ""
    ensmem2 = ""

taspath = (
    "/badc/deposited2021/chess-scape/data/rcp"
    + rcp2
    + ensmem2
    + "daily/tas/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_tas_uk_1km_daily_????????-????????.nc"
)
prepath = (
    "/badc/deposited2021/chess-scape/data/rcp"
    + rcp2
    + ensmem2
    + "daily/pr/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_pr_uk_1km_daily_????????-????????.nc"
)
tmnpath = (
    "/badc/deposited2021/chess-scape/data/rcp"
    + rcp2
    + ensmem
    + "daily/tasmin/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_tasmin_uk_1km_daily_????????-????????.nc"
)
tmxpath = (
    "/badc/deposited2021/chess-scape/data/rcp"
    + rcp2
    + ensmem2
    + "daily/tasmax/chess-scape_rcp"
    + rcp
    + "_"
    + ensmem
    + "_tasmax_uk_1km_daily_????????-????????.nc"
)
savedir = (
    "/gws/nopw/j04/ceh_generic/matbro/ecocrop/scores_rcp"
    + rcp
    + "_ens"
    + ensmem
    + "_"
    + ab
)
plotdir = (
    "/gws/nopw/j04/ceh_generic/matbro/ecocrop/plots_rcp"
    + rcp
    + "_ens"
    + ensmem
    + "_"
    + ab
)

yearaggmethod = "percentile"
precmethod = 2


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
sys.stdout.flush()

if not os.path.exists(savedir):
    os.makedirs(savedir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# open datafiles
print("Reading in met data")
print("Start: " + str(dt.datetime.now()))
sys.stdout.flush()
tas = xr.open_mfdataset(taspath).astype("float16")[tasvname]
tmn = xr.open_mfdataset(tmnpath).astype("float16")[tmnvname]
tmx = xr.open_mfdataset(tmxpath).astype("float16")[tmxvname]
pre = xr.open_mfdataset(prepath)[prevname]
if pf == "past":
    tas = tas.sel(time=slice(tas["time"][0], "2021-01-01")).load()
    tmn = tmn.sel(time=slice(tmn["time"][0], "2021-01-01")).load()
    tmx = tmx.sel(time=slice(tmx["time"][0], "2021-01-01")).load()
    pre = pre.sel(time=slice(pre["time"][0], "2021-01-01")).load()
elif pf == "future":
    tas = tas.sel(time=slice("2020-01-01", tas["time"][-1])).load()
    tmn = tmn.sel(time=slice("2020-01-01", tmn["time"][-1])).load()
    tmx = tmx.sel(time=slice("2020-01-01", tmx["time"][-1])).load()
    pre = pre.sel(time=slice("2020-01-01", pre["time"][-1])).load()
else:
    print("Past or future not selected so loading entire dataset")
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
sys.stdout.flush()
if method == "annual":
    topt_crop = score_temp2(tas, TMIN, TMAX, TOPMIN, TOPMAX).values
ktmp_crop = xr.where(tmn < KTMP, 1, 0).astype("uint16").values
kmax_crop = xr.where(tmx > KMAX, 1, 0).astype("uint16").values
print("End: " + str(dt.datetime.now()))
sys.stdout.flush()

# Determine growing season lengths to assess
# Intervals of 10 days are used to reduce
# computational cost
if GMAX - GMIN <= 15:
    gstart = np.int16(np.floor(GMIN / 10) * 10)
else:
    gstart = np.int16(np.ceil(GMIN / 10) * 10)
gend = np.int16(np.ceil(GMAX / 10) * 10)
allgtimes = list(np.arange(gstart, gend, 10, dtype="int16"))

# create arrays to store the total proportion of ktmp/kmax days amassed over all the gtimes
# for later calculating the average
print("Creating ktmp_days_prop and kmax_days_prop arrays")
print("Start: " + str(dt.datetime.now()))
ktmp_days_prop_total = ktmp_crop.copy()[: -allgtimes[-1] + 1].astype("float32")
kmax_days_prop_total = kmax_crop.copy()[: -allgtimes[-1] + 1].astype("float32")
ktmp_days_prop_total[:] = 0
kmax_days_prop_total[:] = 0
kdptlen = ktmp_days_prop_total.shape[0]
print("End: " + str(dt.datetime.now()))
sys.stdout.flush()

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
    sys.stdout.flush()
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
    sys.stdout.flush()
    tcoords_tmn = tmn["time"][: -gtime + 1]
    ycoords_tmn = tmn["y"]
    xcoords_tmn = tmn["x"]
    ktmp_days = frs3D(ktmp_crop, gtime, "uint16")
    ktmp_days_prop_total += (ktmp_days / gtime)[:kdptlen]
    ktmp_days = xr.DataArray(ktmp_days, coords=[tcoords_tmn, ycoords_tmn, xcoords_tmn])
    ktmp_days.name = "KTMP_days"
    print(ktmp_days_prop_total.dtype)
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
    kmax_days_prop_total += (kmax_days / gtime)[:kdptlen]
    kmax_days = xr.DataArray(kmax_days, coords=[tcoords_tmx, ycoords_tmx, xcoords_tmx])
    kmax_days.name = "KMAX_days"
    print("End: " + str(dt.datetime.now()))

    print("Calculating total precipitation")
    print("Start: " + str(dt.datetime.now()))
    sys.stdout.flush()
    # calculate total precipitation in gtime
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
    sys.stdout.flush()
    tempscore = xr.where(ktmp_days > np.uint8(0), np.uint8(0), tscore)
    print("End: " + str(dt.datetime.now()))

    print("Processing KMAX days penalty")
    print("Start: " + str(dt.datetime.now()))
    sys.stdout.flush()
    tempscore = tempscore - np.int8(kmax_days)
    tempscore = xr.where(tempscore < 0, 0, tempscore).astype("uint8")
    print("End: " + str(dt.datetime.now()))

    # Calculate precipitation suitability score
    print("Calculating precip suitability score using method " + str(precmethod))
    print("Start: " + str(dt.datetime.now()))
    sys.stdout.flush()
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
    sys.stdout.flush()
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
        sys.stdout.flush()
    counter += 1

# Combine the temperature and precipitation suitability scores
# by taking the minimum, as this will likely be the
# constraining factor on any crop growth
print("Calculating final combined crop suitability score")
print("Start: " + str(dt.datetime.now()))
sys.stdout.flush()
final_score_crop = xr.where(precscore < tempscore, precscore, tempscore)
print(final_score_crop.dtype)
print("End: " + str(dt.datetime.now()))

print("Calculating average ktmp_ and kmax_days proportions")
print("Start: " + str(dt.datetime.now()))
sys.stdout.flush()
ktmp_days_avg_prop = ktmp_days_prop_total / len(allgtimes)
kmax_days_avg_prop = kmax_days_prop_total / len(allgtimes)
tcoords_k = tmn["time"][: -allgtimes[-1] + 1]
ktmp_days_avg_prop = xr.DataArray(
    ktmp_days_avg_prop, coords=[tcoords_k, ycoords_tmx, xcoords_tmx]
)
kmax_days_avg_prop = xr.DataArray(
    kmax_days_avg_prop, coords=[tcoords_k, ycoords_tmx, xcoords_tmx]
)
print(ktmp_days_avg_prop)
print(ktmp_days_avg_prop.dtype)
print("End: " + str(dt.datetime.now()))

# Save outputs to file
print("Saving to netcdf")
print("Start: " + str(dt.datetime.now()))
sys.stdout.flush()
# save to netcdf
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

ktmp_days_avg_prop.name = "average_proportion_of_ktmp_days_in_gtime"
ktmp_days_avg_prop.encoding["zlib"] = True
ktmp_days_avg_prop.encoding["complevel"] = 1
ktmp_days_avg_prop.encoding["shuffle"] = False
ktmp_days_avg_prop.encoding["contiguous"] = False
ktmp_days_avg_prop.encoding["dtype"] = np.dtype("float32")
encoding = {}
encoding["average_proportion_of_ktmp_days_in_gtime"] = ktmp_days_avg_prop.encoding
ktmp_days_avg_prop.to_netcdf(
    os.path.join(savedir, cropname + "_ktmp_days_avg_prop.nc"),
    encoding=encoding,
)

kmax_days_avg_prop.name = "average_proportion_of_kmax_days_in_gtime"
kmax_days_avg_prop.encoding["zlib"] = True
kmax_days_avg_prop.encoding["complevel"] = 1
kmax_days_avg_prop.encoding["shuffle"] = False
kmax_days_avg_prop.encoding["contiguous"] = False
kmax_days_avg_prop.encoding["dtype"] = np.dtype("float32")
encoding = {}
encoding["average_proportion_of_kmax_days_in_gtime"] = kmax_days_avg_prop.encoding
kmax_days_avg_prop.to_netcdf(
    os.path.join(savedir, cropname + "_kmax_days_avg_prop.nc"),
    encoding=encoding,
)
print("End: " + str(dt.datetime.now()))


# calculate and plot monthly climos of ktmp & kmax days avg prop for each decade and their differences
print("Calculating monthly climo of ktmp/kmax proportions and decadal changes")
sys.stdout.flush()
(
    ktmpap_monavg_climo_diffs,
    kmaxap_monavg_climo_diffs,
) = calc_decadal_kprop_changes(
    ktmp_days_avg_prop,
    kmax_days_avg_prop,
    str(SOIL),
    lcmloc,
    bgsloc,
    cropname,
    savedir,
)
# for month in range(1, 13):
#    plot_decadal_changes(kmaxap_monavg_climo_diffs.sel(month=month),
#                         save=os.path.join(plotdir, cropname + '_kmaxdaysprop_decadal_change_month' + str(month) + '.png'),
#                         revcolbar = 1)
# for month in range(1, 13):
#    plot_decadal_changes(ktmpap_monavg_climo_diffs.sel(month=month),
#                         save=os.path.join(plotdir, cropname + '_ktmpdaysprop_decadal_change_month' + str(month) + '.png'),
#                         revcolbar = 1)

# calculate day of year of maximum score
print("Finding days of years of the maximum score")
sys.stdout.flush()
maxdoys, maxdoys_temp, maxdoys_prec = calculate_max_doy(
    final_score_crop, tempscore, precscore
)
print(
    "Calculating yearly average of this and decadal changes using modulo arithmetic/circular averaging"
)
sys.stdout.flush()
(
    maxdoys_decadal_changes,
    maxdoys_temp_decadal_changes,
    maxdoys_prec_decadal_changes,
) = calc_decadal_doy_changes(
    maxdoys,
    maxdoys_temp,
    maxdoys_prec,
    str(SOIL),
    lcmloc,
    bgsloc,
    cropname,
    savedir,
)
# plot_decadal_changes(maxdoys_decadal_changes, save=os.path.join(plotdir, cropname + '_maxdoys_decadal_changes.png'))
# plot_decadal_changes(maxdoys_temp_decadal_changes, save=os.path.join(plotdir, cropname + '_maxdoys_temp_decadal_changes.png'))
# plot_decadal_changes(maxdoys_prec_decadal_changes, save=os.path.join(plotdir, cropname + '_maxdoys_prec_decadal_changes.png'))

# calculate yearly scores and decadal changes
print("Calculating yearly scores and decadal changes")
sys.stdout.flush()
(
    allscore_decades,
    tempscore_decades,
    precscore_decades,
    allscore_decadal_changes,
    tempscore_decadal_changes,
    precscore_decadal_changes,
) = calc_decadal_changes(
    tempscore,
    precscore,
    str(SOIL),
    lcmloc,
    bgsloc,
    cropname,
    savedir,
    yearaggmethod,
)
# plot_decadal_changes(allscore_decadal_changes, save=os.path.join(plotdir, cropname + '_decadal_changes.png'))
# plot_decadal_changes(tempscore_decadal_changes, save=os.path.join(plotdir, cropname + '_tempscore_decadal_changes.png'))
# plot_decadal_changes(precscore_decadal_changes, save=os.path.join(plotdir, cropname + '_precscore_decadal_changes.png'))
# plot first decade's scores
plot_decade(
    allscore_decades[0, :, :],
    tempscore_decades[0, :, :],
    precscore_decades[0, :, :],
    save=os.path.join(plotdir, cropname + "_current_decade.png"),
)
