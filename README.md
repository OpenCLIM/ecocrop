# EcoCrop Climatic Crop Suitability Scoring

MJB 14/7/23
-----------

# Overview

The Ecocrop suitability model assesses the changing climatic suitability of a variety of crops. Temperature and precipitation are assessed, so the suitability is for un-irrigated crops grown outside. Other impacts on suitability such as changing soils and the spread of pests or diseases are not accounted for. 

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
- A python environment with xarray and pandas is required. Other optional packages are matplotlib and cartopy is making use of the plotting scripts and dask if attempting to process more data than will fit in memory.

# Installation
The current recommended method for testing the code is to run the 'Binder' instance associated with the code, which automatically installs all the dependencies and allows you to run the code from your web browser. Access the Binder here, or by clicking the binder button at the top of this README. 

Once the Binder environment has launched, open a terminal and run the testing script with the command `python ecocrop_testdata_run.py`. The code should run and produce suitability scores for the year 2020 over north Norfolk as a netcdf file and plot in the `outputs` folder.
The plot of the netcdf file should look like this:
**INSERT IMAGE**

Alternatively, it is possible to download the code and set up the environment required to run it manually using anaconda:
- First set up an anaconda or mamba environment with the required packages in. This can be done on Windows, Mac or Linux operating systems.
- Installation instructions and downloads for each operating system can be found on the [Anaconda website](https://www.anaconda.com/download)
- Download the EcoCrop repository to your local machine using `git clone https://github.com/OpenCLIM/ecocrop.git` or `git clone git@github.com:OpenCLIM/ecocrop.git` from the shell/terminal/commandline if git is installed, or `gh repo clone OpenCLIM/ecocrop` if you are on Windows and have the [Git Command Line client](https://cli.github.com/) installed. If git/Git CLI is not installed a zip file of the repository can be obtained from **LINK** zenodo.
- Once anaconda is installed, create a separate environment containing only the packages necessary to run EcoCrop. The correct environment can be set up using the environment.yml file provided in the EcoCrop repository by running `conda env create -f /path/to/environment.yml`. replacing `/path/to/` with the full path of the directory/folder of the repository or where the environment.yml file is if it has been moved. If on a windows machine it is recommended to do this in the 'Anaconda Prompt' that is installed when you install Anaconda. For Mac or Linux users this can be done in the terminal/shell. 
- Once the environment is installed, activate it using **conda activate envname** and you are ready to go!

# Test running instructions

A small dataset and script is provided for testing the code and checking it produces the expected outputs. The test script is set up to produce suitability scores for north Norfolk for 2020 for a wheat crop. 
The test script is ecocrop_testdata_run.py and the test data is in the testdata folder. It can be run on Windows, Mac or Linux operating systems.
Edits can be made to the variables at the start of the test script if necessary, but it is recommended to leave these as they are for the purposes of testing. 
- Ensure the **envname** environment is active by running **conda activate envname** in the Anaconda Prompt (Windows) or terminal/shell (Mac/Linux)
- Ensure you are in the directory containing all the repository files, including the test script. Use the 'cd' command in the prompt/terminal/shell to change directory. The current directory can be checked with the 'pwd' command and the files in the current directory can be viewed with 'ls'.
- Run the test script with the command `python ecocrop_testdata_run.py`
- There will be printouts on the screen displaying what the script is doing
- The suitability scores output from the script will be sent to the 'testoutputs' folder, along with an example plot.

# Full running instructions

- This code is set up to run with the full 100-year daily and 1km resolution CHESS-SCAPE dataset, but can be run with any dataset that has daily precipitation and daily average/max/min temperature.
- It is designed to run on a HPC due to the high memory requirements 
- An example of a SLURM job submit script is provided as ecocrop_lotus_himem_template.sbatch
- This calls the main python script ecocrop_lotus_himem.py with the following arguments as inputs:
- 'cropind': Replace this with the EcoCrop_DB_secondtrim.csv row number (0-based, ignoring header row) of the spreadsheet in the sbatch template.
- 'rcp' and 'ensmem': variables are for the different RCP Scenarios and ensemble members of the CHESS-SCAPE dataset respectively. They only affect the input and output data directories.
- 'pf': handles the fact that the CHESS-SCAPE dataset was originally split up into before and after 2020, to help with memory limits. Again though it only affects the input and output data dirs. Can be set to 'past' or 'future' or anything else to ignore it.
- 'method': The temperature scoring method as described below. Can be 'annual' or 'perennial'.
- The following variables can be edited within the python script itself:
- ecocroploc: The location of the ecocrop database (provided in the repo)
- tasvname: Variable name of daily average temperature in the input netcdf files
- tmnvname: Variable name of daily minimum temperature in the input netcdf files
- tmxvname: Variable name of daily maximum temperature in the input netcdf files
- precname: Variable name of daily precipitation total in the input netcdf files
- lcmloc: Location of the arable land mask (provided in the repo)
- bgsloc: Location of the soil masks (provided in the repo)
You can also edit the taspath, tmnpath, tmxpath, precpath to point to your netcdf files as needed, and the plotloc and saveloc for where output plots and netcdf files are to be stored. 


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
