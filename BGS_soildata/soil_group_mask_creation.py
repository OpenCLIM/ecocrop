import geopandas as gpd
import xarray as xr

sm = gpd.read_file("BGS_soildata/SoilParentMateriall_V1_portal1km.shp")
centre_points = sm.centroid
centre_points.name = "geometry"
centre_points = gpd.GeoDataFrame(centre_points)
centre_points.crs = sm.crs
smc = gpd.sjoin(sm, centre_points, how="right")
smc = smc.drop("index_left", axis=1)

smc = smc.drop(
    [
        "ESB_DESC",
        "CARB_CNTNT",
        "PMM_GRAIN",
        "SOIL_TEX",
        "SOIL_DEPTH",
        "PMM1K_UID",
        "VERSION",
    ],
    axis=1,
)

smc["HEAVY"] = 1
smc["MEDIUM"] = 1
smc["LIGHT"] = 1

smc["LIGHT"] = smc["LIGHT"].where(
    ["LIGHT" in sg for sg in list(smc["SOIL_GROUP"])], other=0
)
smc["MEDIUM"] = smc["MEDIUM"].where(
    ["MEDIUM" in sg for sg in list(smc["SOIL_GROUP"])], other=0
)
smc["HEAVY"] = smc["HEAVY"].where(
    ["HEAVY" in sg for sg in list(smc["SOIL_GROUP"])], other=0
)

smc_l = smc.drop(["SOIL_GROUP", "HEAVY", "MEDIUM"], axis=1)
smc_m = smc.drop(["SOIL_GROUP", "HEAVY", "LIGHT"], axis=1)
smc_h = smc.drop(["SOIL_GROUP", "MEDIUM", "LIGHT"], axis=1)

smc_l.to_file(driver="ESRI Shapefile", filename="BGS_soildata/light_soils.shp")
smc_m.to_file(driver="ESRI Shapefile", filename="BGS_soildata/medium_soils.shp")
smc_h.to_file(driver="ESRI Shapefile", filename="BGS_soildata/heavy_soils.shp")

# Then use 'gdal_rasterize' from bash to conver these to raster:
# e.g.: gdal_rasterize -tr 1000 1000 -te 0 0 700000 1250000 -a LIGHT -a_nodata 0 light_soils.shp test.tif

template = xr.open_dataset("scores/onions.nc")["crop_suitability_score"]
template = template[0, :, :]
template

x1 = template["x"].values[0]
x2 = template["x"].values[-1]
y1 = template["y"].values[0]
y2 = template["y"].values[-1]

l_mask = xr.open_rasterio("BGS_soildata/light_soils.tif")
l_mask = l_mask.drop("band").squeeze()
l_mask.name = "light_soil_mask"
l_mask = l_mask[::-1, :]
l_mask.sel(x=slice(x1, x2), y=slice(y1, y2))
l_mask = l_mask.astype("uint8")
l_mask.to_netcdf("BGS_soildata/light_soil_mask.nc")

m_mask = xr.open_rasterio("BGS_soildata/medium_soils.tif")
m_mask = m_mask.drop("band").squeeze()
m_mask.name = "medium_soil_mask"
m_mask = m_mask[::-1, :]
m_mask.sel(x=slice(x1, x2), y=slice(y1, y2))
m_mask = m_mask.astype("uint8")
m_mask.to_netcdf("BGS_soildata/medium_soil_mask.nc")

h_mask = xr.open_rasterio("BGS_soildata/heavy_soils.tif")
h_mask = h_mask.drop("band").squeeze()
h_mask.name = "heavy_soil_mask"
h_mask = h_mask[::-1, :]
h_mask.sel(x=slice(x1, x2), y=slice(y1, y2))
h_mask = h_mask.astype("uint8")
h_mask.to_netcdf("BGS_soildata/heavy_soil_mask.nc")
