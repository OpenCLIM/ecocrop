mkdir testdata
cp -v /badc/deposited2021/chess-scape/data/rcp85/01/daily/tas/chess-scape_rcp85_01_tas_uk_1km_daily_2020*.nc testdata
cp -v /badc/deposited2021/chess-scape/data/rcp85/01/daily/tasmax/chess-scape_rcp85_01_tasmax_uk_1km_daily_2020*.nc testdata
cp -v /badc/deposited2021/chess-scape/data/rcp85/01/daily/tasmin/chess-scape_rcp85_01_tasmin_uk_1km_daily_2020*.nc testdata
cp -v /badc/deposited2021/chess-scape/data/rcp85/01/daily/pr/chess-scape_rcp85_01_pr_uk_1km_daily_2020*.nc testdata

cd testdata
mkdir cropped

for file in $(ls *.nc); do echo ${file}; cdo selindexbox,554,656,300,350 ${file} cropped/${file}; done

cp cropped/*.nc .
rm -r cropped
