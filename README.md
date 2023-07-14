# EcoCrop Climactic Crop Suitability Scoring

MJB 14/7/23
-----------

# Overview

The Ecocrop suitability model assesses the changing climatic suitability of 182 crops. Only temperature and precipitation are assessed, meaning that the suitability only reflects that for un-irrigated crops grown outside. Other impacts on suitability such as changing soils and the spread of pests or diseases are not accounted for. 

The tool uses the following parameters from the EcoCrop database for the climate suitability calculation:
- TOPMN,optimal minimum temperature
- TOPMX,optimal maximum temperature
- TMIN,absolute minimum temperature
- TMAX,absolute maximum temperature
- KTMP,Killing temperature early growth
- ROPMN,optimal minimum rainfall
- ROPMX,optimal maximum rainfall
- RMIN,absolute minimum rainfall
- RMAX,absolute maximum rainfall
- GMIN,Minimum crop cycle,the minimum number of 'growing days' the crop needs
- GMAX,Maximum crop cycle,the maximum length of time the crop can grow in

The following parameters are used, optionally, for additional masking of suitable locations for crop growth:
- TEXT,Optimal soil texture
- TEXTR,Absolute soil texture

A suitability score out of 100 is calculated for temperature and precip separately, with the combined score the lower of these.


# Inputs and requirements

- The full FAO EcoCrop database is provided as EcoCrop_DB.csv.
- The code is currently set up to run on a trimmed-down version, EcoCrop_DB_secondtrim.csv
- Daily values of average, minimum, maximum temperature are required, along with daily precipitation, all on a grid in netCDF format. 
- Units of Kelvin and kg/m^2/s are expected
- A python environment with xarray, pandas, dask, cftime, cartopy and geopandas is required. Matplotlib also if plotting outputs.


# Running instructions

- This code is set up to run with the CHESS-SCAPE dataset, it remains an issue to generalise this to other climate datasets. 
- It is designed to run on a HPC due to the high memory requirements (approximately 1.5x the disk size of the meteorological data, ~500GB for the UK at 1km resolution for 60years of daily data)
- An example of a SLURM job submit script is provided as ecocrop_lotus_himem_template.sbatch
- This calls the main python script ecocrop_lotus_himem.py with the following arguments as inputs:
- 'cropind': Replace this with the EcoCrop_DB_secondtrim.csv row number (0-based, ignoring header row) of the spreadsheet in the sbatch template.
- 'rcp' and 'ensmem': variables are for the different RCP Scenarios and ensemble members of the CHESS-SCAPE dataset respectively. They only affect the input and output data directories.
- 'pf': handles the fact that the CHESS-SCAPE dataset was split up into before and after 2020, to help with memory limits. Again though it only affects the input and output data dirs. Can be set to 'past' or 'future' or anything else to ignore it.
- 'method': The temperature scoring method as described below. Can be 'annual' or 'perennial'.
- The following variables can be edited within the python script itself:
- ecocroploc: The location of the ecocrop database
- tasvname: Variable name of daily average temperature in the input netcdf files
- tmnvname: Variable name of daily minimum temperature in the input netcdf files
- tmxvname: Variable name of daily maximum temperature in the input netcdf files
- precname: Variable name of daily precipitation total in the input netcdf files
- lcmloc: Location of the arable land mask (provided in the repo)
- bgsloc: Location of the soil masks (provided in the repo)
You can also edit the taspath, tmnpath, tmxpath, precpath to point to your netcdf files as needed, and the plotloc and saveloc for where output plots and netcdf files are stored. 


# Method

The ECOCROP suitability model derives a suitability score based on daily temperature and daily precipitation values, using information on the required and optimal temperature and precipitation ranges, and the number of days within which the crop must grow. The temperature and precipitation suitability score for a given crop is calculated for each day and grid square in the CHESS-SCAPE (or provided) dataset and a selection of possible growing times (GTIMEs) between GMIN and GMAX by looking forward in time by GTIME days and calculating the scores for this period. We chose GTIMEs at an interval of 10days from GMIN onwards to balance accuracy against computational cost. The scores for each GTIME are first aggregated by taking the maximum, then the scores for each day are aggregated to yearly scores by taking the 95th centile over each year. Using the 95th centile ensures that the aggregated annual score represents the best possible score derived from the optimal timing of crop growth and harvest, without being overly sensitive to anomalous single days with high scores (as would be the case if the maximum was used). The minimum of the two scores at each grid square is then taken to give an overall score, as the lowest score is likely to be the limiting factor in the crop’s growth.

## Temperature Suitability Scoring Method

The temperature score $S_T$ is calculated in two different methods dependent on whether or not the crop is usually grown as an annual or perennial crop.

### Annual Scoring Method

For annual crops, $S_T$ is calculated using the following method:

For each day, 1km grid cell and GTIME length, an intermediate score between 0 and 1 is assigned using the following equations:

$D= \frac{T-TMIN}{TOPMN-TMIN} \text{ when }TMIN\lt T \lt TOPMN$

$D=1 \text{ when } TOPMN\lt T\lt TOPMX$

$D= \frac{TMAX-T}{TMAX-TOPMX} \text{ when }TOPMX\lt T\lt TMAX$

$D=0\text{ for all other }T$

Where $T$ is the average temperature of the given day. A score of 1 represents a day that is maximally temperature suitable for the given crop, and 0 not suitable. 

Then a sum of $D$ across the subsequent $GTIME$ days is calculated:

$$N_D=\sum_{days}^{GTIME}D$$

This sum, $N_D$, is the total number of suitable days within $GTIME$. 

If $N_D$ is greater than or equal to $GMIN$, I.e. if at least the minimum number of suitable days is achieved within $GTIME$, then a suitability score $S_T$, dependent only on the given $GTIME$, is assigned to the given day:

$S_T=100\left[1-\frac{GTIME-GMIN}{GMAX-GMIN}\right]\text{ where } N_D≥GMIN\text{ else } S_T=0$

The result of this calculation is that the fewer days it takes to amass $GMIN$ suitable days, the higher the suitability score ($S_T$).


Heat stress and frost penalties are then applied to the suitability score to account for temperature extremes. Daily minimum temperatures within the $GTIME$ window are checked and if there is a daily-minimum temperature below $KTMP$ then $S_T$ is set to 0. A heat stress penalty is also applied by subtracting the number of days with a daily maximum temperature above $TMAX$ from $S_T$.


### Perennial Scoring Method

The temperature score for a given $GTIME$, each day, grid square and crop is calculated as follows:
First, the average daily-average-temperature ($TAVG$) across $GTIME$ is calculated. Then the following equation is used to calculate the score, $S_T$:

$S_T=\frac{100}{0.5\left(TOPMX+TOPMN\right)-TMIN} \left(TAVG-TMIN\right)\text{ when } TMIN\lt TAVG\lt 0.5\left(TOPMX+TOPMN\right)$

$S_T=\frac{100}{TMAX-0.5\left(TOPMX+TOPMN\right)}\left(TMAX-TAVG\right)\text{ when } TMAX\gt TAVG\gt 0.5\left(TOPMX+TOPMN\right)$

$S_T=0\text{ for all other }TAVG$

The same heat and frost penalties as for the annual temperature suitability scoring method are then applied.


## Precipitation Suitability Scoring Method

The precipitation score is calculated in a similar way to the perennial temperature scoring method. The precipitation total ($PTOT$) is calculated over the $GTIME$ period then the following equation is used: 

$S_P=\frac{100}{0.5\left(POPMX+POPMN\right)-PMIN}\left(PTOT-PMIN\right)\text{ where }PMIN\lt PTOT\lt 0.5\left(POPMX+POPMN\right)$

$S_P=\frac{100}{PMAX-0.5\left(POPMX+POPMN\right)}\left(PMAX-PTOT\right)\text{ where }PMAX\gt PTOT\gt 0.5\left(POPMX+POPMN\right)$

$S_P=0\text{ for all other }PTOT$

# License information

This code is available under the [terms of the Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
By accessing or using this dataset, you agree to the [terms of the Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). 
Please cite this repository in any outputs produced from it.
