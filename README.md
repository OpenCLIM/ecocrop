# EcoCrop Climatic Crop Suitability Scoring

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OpenCLIM/ecocrop/HEAD?labpath=ecocrop_testrun_notebook.ipynb)

[![DOI](https://zenodo.org/badge/633033201.svg)](https://zenodo.org/doi/10.5281/zenodo.10843625)

MJB 19/3/24
-----------

# Overview

The Ecocrop suitability model assesses the changing climatic suitability of a variety of crops using the [FAO EcoCrop database](https://gaez.fao.org/pages/ecocrop) and driving climate model or meterological data. Temperature and precipitation are assessed, so the suitability reflects that for un-irrigated crops grown outside. Other impacts on suitability such as changing soils and the spread of pests or diseases are not accounted for. A suitability score out of 100 is calculated for temperature and precipitation separately at each grid cell, for each day within the driving climate dataset. The combined suitability score is the gridpoint-wise lower of these.

# Inputs and requirements

- The full [FAO EcoCrop database](https://gaez.fao.org/pages/ecocrop) is provided as [EcoCrop_DB.csv](https://github.com/OpenCLIM/ecocrop/blob/main/EcoCrop_DB.csv).
- The code is currently set up to run on a trimmed-down version, [EcoCrop_DB_secondtrim.csv](https://github.com/OpenCLIM/ecocrop/blob/main/EcoCrop_DB_secondtrim.csv)
- Daily values of average, minimum, maximum temperature are required, along with daily precipitation, all on a grid in netCDF format.
- Units of Kelvin and kg/m^2/s are expected
- A python environment with xarray, rioxarray, dask, netcdf4, pandas, cartopy is required. An example [environment.yml](https://github.com/OpenCLIM/ecocrop/blob/main/environment.yml) file is provided.
- A mask for arable land ([provided in the repo](https://github.com/OpenCLIM/ecocrop/blob/main/Mask_arable_LCM2015_UK.tif), derived from the [UKCEH Land Cover Map 2015](https://doi.org/10.5285/6c6c9203-7333-4d96-88ab-78925e7a4e73))
- Masks for 'heavy', 'medium' and 'light' soil textures ([provided in the repo](https://github.com/OpenCLIM/ecocrop/tree/main/EU_STM_soildata), dervied from [BGS ESRI](https://www.bgs.ac.uk/download/esri-soil-parent-material-model-1km-resolution/) for England, Scotland and Wales, and the [European Soil Data Map](https://esdac.jrc.ec.europa.eu/content/european-soil-database-v2-raster-library-1kmx1km) for Northern Ireland)

# Installation and testing instructions

A small dataset and script is provided for testing the code and checking it produces the expected outputs. The test script is set up to produce suitability scores for north Norfolk for 2020 for a wheat crop. The test script is [ecocrop_testdata_run.py](https://github.com/OpenCLIM/ecocrop/blob/main/ecocrop_testdata_run.py) and the test data is in the [testdata folder](https://github.com/OpenCLIM/ecocrop/tree/main/testdata). The recommended way to run it is using the Binder instance associated with this repo, but it can also be run on Windows, Mac or Linux operating systems.
The script should output six netcdf files:
- **wheat.nc**: The combined suitability score for each day and gridpoint in the test dataset
- **wheat_temp.nc**: The temperature suitability score for each day and gridpoint in the test dataset
- **wheat_prec.nc**: The precipitation suitability score for each day and gridpoint in the test dataset
- **wheat_years.nc**: As wheat.nc but aggregated over the years in the test dataset
- **wheat_tempscore_years.nc**: As wheat_temp.nc but aggregated over the years in the test dataset
- **wheat_precscore_years.nc**: As wheat_prec.nc but aggregated over the years in the test dataset

and one plot, png file:
- **wheat_2020.png**: Plot of the aggregated score for 2020 for the combined, temperature and precipitation suitability scores

### Using Binder
The current recommended method for testing the code is to run the 'Binder' instance associated with the code, which automatically installs all the dependencies and allows you to run the code in a notebook from your web browser. [Access the Binder here](https://mybinder.org/v2/gh/OpenCLIM/ecocrop/HEAD?labpath=ecocrop_testrun_notebook.ipynb), or by clicking the binder button at the top of this README.

Once the Binder environment and python notebook has launched, which can take a few minutes, run the displayed notebook using the 'Run' menu at the top of the webpage to select 'Run All Cells' or by clicking on the displayed code cell and pressing Shift+Enter. The code should run and produce suitability scores for the year 2020 over north Norfolk as netcdf files and an example plot in the `testoutputs` folder. It should not take longer than a minute to run. The plot of the scores should display automatically when the notebook has completed running. Clicking the folder icon on the left-side vertical toolbar and navigating to the testoutputs folder will allow you to download the produced netcdf files (right click on a file and select 'Download' from the menu that appears), or you are welcome to use the notebook to view them yourself, no changes you make to the notebook will be saved to the repo. You are welcome to change the parameters and play around with the code, for example to calculate the suitability for different crops. Again, no changes you make will be saved to the repo.

### Using an Anaconda environment
Alternatively, it is possible to download the code and set up the environment required to run it manually using anaconda:
- First set up an anaconda environment with the required packages in. This can be done on Windows, Mac or Linux operating systems. Installation instructions and downloads for each operating system can be found on the [Anaconda website](https://www.anaconda.com/download). It should not take longer than an hour to install on most relatively modern computers.
- Download the EcoCrop repository to your local machine using `git clone https://github.com/OpenCLIM/ecocrop.git` or `git clone git@github.com:OpenCLIM/ecocrop.git` from the shell/terminal/commandline if git is installed, or `gh repo clone OpenCLIM/ecocrop` if you use the [Git Command Line client](https://cli.github.com/). If git/Git CLI is not installed a zip file of the repository can be obtained from [zenodo](https://zenodo.org/doi/10.5281/zenodo.10843625) or the DOI button at the top of this README.
- Once anaconda is installed, create a separate environment containing only the packages necessary to run EcoCrop. The correct environment can be set up using the [environment.yml](https://github.com/OpenCLIM/ecocrop/blob/main/environment.yml) (Mac or Linux) or [environment_windows.yml](https://github.com/OpenCLIM/ecocrop/blob/main/environment_windows.yml) file (Windows) provided in the repository by running `conda env create -f /path/to/environment.yml`, replacing `/path/to/` with the full path of the directory/folder of the repository or where the environment.yml or environment_windows.yml file is if it has been moved. If on a Windows machine it is recommended to do this in the 'Anaconda Prompt' that is installed when you install Anaconda. For Mac or Linux users this can be done in the terminal/shell. This should also not take longer than an hour to complete on most relatively modern computers. 
- Once the environment is installed, activate it using `conda activate ecocroptest` and you are ready to go!

Edits can be made to the variables at the start of the test script if necessary, but it is recommended to leave these as they are for the purposes of testing.
- Ensure the ecocroptest environment is active by running `conda activate ecocroptest` in the Anaconda Prompt (Windows) or terminal/shell (Mac/Linux)
- Ensure you are in the directory containing all the repository files, including the test script. Use the `cd` command in the prompt/terminal/shell to change directory. The current directory can be checked with the `pwd` command and the files in the current directory can be viewed with `ls` (`dir` on Windows)
- Run the test script with the command `python ecocrop_testdata_run.py`, it shouldn't take longer than a couple of minutes to run
- There will be printouts on the screen displaying what the script is doing
- The suitability scores output from the script will be written to the 'testoutputs' folder, along with an example plot.

The output plot should look like this:
![test ecocrop plot](testoutputs/verification/wheat_2020.png)
The script compares the output against a pre-existing verification file within `testoutputs/verification`, provided no parameters are changed within the `ecocrop_testdata_run.py` script. An error will be raised if the files do not match, unless a change to the parameters is detected. 

# Full running instructions

- The full version of the code is set up to run with the 100-year daily and 1km resolution [CHESS-SCAPE dataset](https://dx.doi.org/10.5285/8194b416cbee482b89e0dfbe17c5786c), but can be run with any dataset that has daily precipitation and daily average/max/min temperature.
- Note that the CHESS-SCAPE dataset is not provided in this repo due to it's size, but can be downloaded from the [CEDA Archive](https://dx.doi.org/10.5285/8194b416cbee482b89e0dfbe17c5786c)
- The full version of the code is identical to the test version except that it is designed to run on a HPC due to the high memory requirements of running with such a large dataset
- An example of a job submit script for a SLURM-based HPC system is provided as [ecocrop_lotus_himem_sbatch_template.sbatch](https://github.com/OpenCLIM/ecocrop/blob/main/ecocrop_lotus_himem_sbatch_template.sbatch)
- This calls the main python script [ecocrop_lotus_himem.py](https://github.com/OpenCLIM/ecocrop/blob/main/ecocrop_lotus_himem.py) with the following arguments as inputs:
  - **cropind**: The EcoCrop_DB_secondtrim.csv row number (0-based, ignoring header row) of the spreadsheet in the sbatch template, corresponding to the crop you wish to model
  - **rcp** and **ensmem**: variables are for the different RCP Scenarios and ensemble members of the CHESS-SCAPE dataset respectively. They only affect the input and output data directories
  - **pf**: handles the fact that the CHESS-SCAPE dataset was originally split up into before and after 2020, to help with memory limits. Again though it only affects the input and output data dirs. Can be set to 'past' or 'future', or anything else to ignore it, which is recommended
  - **method**: The temperature scoring method as described below. Can be 'annual' or 'perennial'
- The following variables can be edited within the python script itself:
  - **ecocroploc**: The location of the ecocrop database (provided in the repo)
  - **tasvname**: Variable name of daily average temperature in the input netcdf files
  - **tmnvname**: Variable name of daily minimum temperature in the input netcdf files
  - **tmxvname**: Variable name of daily maximum temperature in the input netcdf files
  - **precname**: Variable name of daily precipitation total in the input netcdf files
  - **lcmloc**: Location of the arable land mask (provided in the repo)
  - **bgsloc**: Location of the soil masks (provided in the repo)
You can also edit the **taspath**, **tmnpath**, **tmxpath**, **precpath** to point to your netcdf files as needed, and the **plotloc** and **saveloc** for where output plots and netcdf files are to be stored.

The outputs of the full version of the code are [as for the test version](https://github.com/OpenCLIM/ecocrop/blob/main/README.md#Installation-and-testing-instructions) with the addition of:
- **cropname_decades.nc**: The combined suitability score for each gridcell aggregated over the decades in the driving dataset
- **cropname_tempscore_decades.nc**: As cropname_decades.nc but for temperature suitability scores only
- **cropname_precscore_decades.nc**: As cropname_decades.nc but for precipitation suitability scores only
- **cropname_decadal_changes.nc**: The decadal changes in the combined suitability score from the first decade
- **cropname_tempscore_decadal_changes.nc**: As cropname_decadal_changes.nc but for temperature suitability scores only
- **cropname_precscore_decadal_changes.nc**: As cropname_decadal_changes.nc but for precipitation suitability scores only
- **cropname_ktmp_days_avg_prop.nc**: The proportion of days within the growing season that has a minimum temperature below KTMP, averaged across all the growing season lengths (gtimes) considered
- **cropname_kmax_days_avg_prop.nc**: As cropname_ktmp_days_avg_prop.nc but for the maximum temperature above TMAX
- **cropname_ktmpdaysavgprop_decades.nc**: The decadal climatology of cropname_ktmp_days_avg_prop.nc
- **cropname_kmaxdaysavgprop_decades.nc**: The decadal climatology of cropname_kmax_days_avg_prop.nc
- **cropname_ktmpdaysavgprop_decadal_changes.nc**: The decadal changes in cropname_ktmpdaysavgprop_decades.nc from the first decade
- **cropname_kmaxdaysavgprop_decadal_changes.nc**: The decadal changes in cropname_kmaxdaysavgprop_decades.nc from the first decade
- **cropname_max_score_doys.nc**: The day of year a particular gridpoint experiences it's maximum combined suitability score, for each year in the driving dataset
- **cropname_max_tempscore_doys.nc**: As cropname_max_score_doys but for the temperature suitability score only
- **cropname_max_precscore_doys.nc**: As cropname_max_score_doys but for the precipitation suitability score only
- **cropname_max_score_doys_decades.nc**: Decadally-averaged (using modulo/circular arithmetic) cropname_max_score_doys.nc
- **cropname_max_tempscore_doys_decades.nc**: Decadally-averaged (using modulo/circular arithmetic) cropname_max_tempscore_doys.nc
- **cropname_max_precscore_doys_decades.nc**: Decadally-averaged (using modulo/circular arithmetic) cropname_max_precscore_doys.nc
- **cropname_max_score_doys_decadal_changes.nc**: The decadal changes in cropname_max_score_doys_decades.nc from the first decade
- **cropname_max_tempscore_doys_decadal_changes.nc**: The decadal changes in cropname_max_tempscore_doys_decades.nc from the first decade
- **cropname_max_precscore_doys_decadal_changes.nc**: The decadal changes in cropname_max_precscore_doys_decades.nc from the first decade
where **cropname** is the name of the crop being modelled, taken from the EcoCrop database. 

The full version requires the same python environment as the test version, follow the instructions provided in the [Using an Anaconda environment](https://github.com/OpenCLIM/ecocrop/blob/main/README.md#Using-an-Anaconda-environment) section to set one up. 


# Method
The EcoCrop climatic suitability model uses the following parameters from the EcoCrop database for the climate suitability calculation:
- **TOPMN**: optimal minimum temperature
- **TOPMX**: optimal maximum temperature
- **TMIN**: absolute minimum temperature
- **TMAX**: absolute maximum temperature
- **KTMP**: killing temperature
- **ROPMN**: optimal minimum rainfall
- **ROPMX**: optimal maximum rainfall
- **RMIN**: absolute minimum rainfall
- **RMAX**: absolute maximum rainfall
- **GMIN**: minimum crop cycle, the minimum number of 'growing days' the crop needs
- **GMAX**: maximum crop cycle, the maximum length of time the crop can grow in

The following parameters are used for additional masking of suitable locations for crop growth:
- **TEXT**: Optimal soil texture
- **TEXTR**: Absolute soil texture

The suitability score is calculated using information on the required and optimal temperature (TMIN, TMAX, TOPMN, TOPMX) and precipitation (RMIN, RMAX, ROPMN, ROPMX) ranges, and the number of days within which the crop must grow (GMIN, GMAX). The temperature and precipitation suitability score for a given crop is calculated for each day and grid square in the CHESS-SCAPE (or other provided) dataset and a selection of possible growing times (GTIMEs) between GMIN and GMAX by looking forward in time by GTIME days and calculating the scores for this period. We chose GTIMEs at an interval of 10days from GMIN to balance accuracy against computational cost. The scores for each GTIME are first aggregated by taking the maximum, then the scores for each day are aggregated to yearly scores by taking the 95th centile over each year. Using the 95th centile ensures that the aggregated annual score represents the best possible score derived from the optimal timing of crop growth and harvest, without being overly sensitive to anomalous single days with high scores (as would be the case if the maximum was used). The minimum of the two scores at each grid square is then taken to give an overall score, as the lowest score is likely to be the limiting factor in the crop’s growth.

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

The temperature score for a given $GTIME$, day, grid square and crop is calculated as follows:
First, the average daily-average-temperature ($TAVG$) across $GTIME$ days into the future is calculated. Then the following equation is used to calculate the score, $S_T$:

$S_T=\frac{100}{0.5\left(TOPMX+TOPMN\right)-TMIN} \left(TAVG-TMIN\right)\text{ when } TMIN\lt TAVG\lt 0.5\left(TOPMX+TOPMN\right)$

$S_T=\frac{100}{TMAX-0.5\left(TOPMX+TOPMN\right)}\left(TMAX-TAVG\right)\text{ when } TMAX\gt TAVG\gt 0.5\left(TOPMX+TOPMN\right)$

$S_T=0\text{ for all other }TAVG$

The same heat and frost penalties as for the annual temperature suitability scoring method are then applied.


## Precipitation Suitability Scoring Method

The precipitation score is calculated in a similar way to the perennial temperature scoring method. The precipitation total ($PTOT$) is calculated over the $GTIME$ period then the following equation is used:

$S_P=\frac{100}{0.5\left(POPMX+POPMN\right)-PMIN}\left(PTOT-PMIN\right)\text{ where }PMIN\lt PTOT\lt 0.5\left(POPMX+POPMN\right)$

$S_P=\frac{100}{PMAX-0.5\left(POPMX+POPMN\right)}\left(PMAX-PTOT\right)\text{ where }PMAX\gt PTOT\gt 0.5\left(POPMX+POPMN\right)$

$S_P=0\text{ for all other }PTOT$

# Citation
If you use this repository in your work, or it helped you, please cite this repository by using the 'Cite this repository' button on the [main repository page](https://github.com/OpenCLIM/ecocrop), to the right of the files, or the CITATION.cff file in the root of the repository. 

# Disclaimer

THIS REPOSITORY IS PROVIDED THE AUTHORS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS REPOSITORY, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# License information

This code is available under the [terms of the Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
By accessing or using this dataset, you agree to the [terms of the Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Please cite this repository in any outputs produced from it.
