EcoCrop Climactic Crop Suitability Scoring
------------------------------------------

MJB 1/3/22
----------

Overview
--------

Tool that calculates how suitable given climate data is for various crops, using the FAO EcoCrop database of crop parameters.
The tool uses the following parameters from the EcoCrop database for the climate suitability calculation:
- TOPMN,optimal minimum temperature
- TOPMX,optimal maximum temperature
- TMIN,absolute minimum temperature
- TMAX,absolute maximum temperature
- KTMPR,Killing temperature during rest
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


Method
------

The suitability calculations use a variable 'GTIME'. 
This represents the time it might take a crop to grow and is varied incrementally from GMIN to GMAX (in steps of 10, to save computation time), to check how successful a crop will be if forced to grow in GTIME days

Temperature suitability is calculated using the following metrics:
- NOPTDAYS: A forward-rolling total of the number of days the average temperature is within the temperature range TOPMN and TOPMX, with window-size GTIME
- NKTMPDAYS: A forward-rolling total of the number of days the minimum temperature is below KTMPR, with window-size GTIME. The number of cold-stress days.
- NKMAXDAYS: A forward-rolling total of the number of days the maximum temperature is above TMAX, with window-size GTIME. The number of heat-stress days.

For each GTIME, the temperature suitability score is calculated as:

100*(1-(GTIME-GMIN)/(GMAX-GMIN)) - NKMAXDAYS

only where NOPTDAYS >= GMIN, otherwise it is 0.
The closer GTIME is to GMIN, the higher the score.
The suitability is also set to zero if NKTMPDAYS > 0

Once this has been calculated for all GTIMEs, the maximum is taken as the final temperature suitability score.
In essence, in the absence of frost or heat stress, the quicker a crop is able to amass the required (GMIN) days, the higher it's score.

Precipitation suitability is calculated using the forward-rolling total of precipitation (P) for each GTIME window-size
The score is calculated as:

(100/(PMINO-PMIN))(P-PMIN) for PMIN<=P<PMINO

100 for PMINO<=P<=PMAXO

(100/(PMAX-PMAXO))(PMAX-P) for PMAXO<P<=PMAX

As with temperature, the maximum over all the GTIME calculations is taken. 

The final temperature and precipitation suitability scores are combined by taking the minimum of the two, as this is assumed to be the limiting factor.


Inputs and requirements
-----------------------

- The FAO EcoCrop database is provided as EcoCrop_DB.csv.
- Daily values of average, minimum, maximum temperature are required, along with daily precipitation, all on a grid in netCDF format. 
- Units of Kelvin and kg/m^2/s are expected
- A python environment with xarray, pandas, dask, cftime, cartopy and geopandas is required. Matplotlib also if plotting outputs.


Running instructions
--------------------

- The tool is designed to run on a HPC due to the high memory requirements (approximately 1.5x the disk size of the meteorological data, ~500GB for the UK at 1km resolution for 60years of daily data)
- An example of a SLURM job submit script is provided as ecocrop_lotus_himem.sbatch
- This runs the python script ecocrop_lotus_himem.py
- Edit the variables at the top of the python script before running


