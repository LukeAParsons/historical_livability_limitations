#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np 
import pylab as plt 
import glob
import pandas as pd
import pickle
import xarray as xr
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

path_era5 = '/uufs/chpc.utah.edu/common/home/u6053495/tnc-group1/data/era5_land/'

months = [
	'01', '02', '03',
	'04', '05', '06',
	'07', '08', '09',
	'10', '11', '12',
	]

#years = np.arange(1950,2024,1)

# Setup argument parser
parser = argparse.ArgumentParser(description="Calculate Relative Humidity for one year of ERA5-Land data.")
parser.add_argument('--year', type=int, required=True, help='Year to process')
args = parser.parse_args()

year = args.year

#for year in np.arange(1950,1951,1):
    
for mindex,month in enumerate(months):

    print('loading t2m and d2m data')

    ds_t2m = xr.open_dataset(path_era5 + '/t2m/' + 'era5_land_t2m_'+str(year)+'_'+str(month)+'_global.nc',
        chunks={"valid_time": 240})
    t2m = ds_t2m['t2m'] - 273.15

    ds_d2m = xr.open_dataset(path_era5 + '/d2m/' + 'era5_land_d2m_'+str(year)+'_'+str(month)+'_global.nc',
        chunks={"valid_time": 240})
    d2m = ds_d2m['d2m']  - 273.15

    print('calculating rh from t2m and d2m data')

    #calculate using ecmwf method
    rh_ecmwf = 100 * (np.exp((17.27 * d2m) / (237.3 + d2m)) / np.exp((17.27 * t2m) / (237.3 + t2m)))
    rh_ecmwf = rh_ecmwf.drop_vars(['number','expver'])

    print('saving rh data in nc file')

    #create xarray dataset
    rh = xr.Dataset(data_vars=dict(
    rh = (["valid_time","latitude", "longitude"],rh_ecmwf.data),),
    coords = dict(
        valid_time=(rh_ecmwf['valid_time']),
        latitude = (rh_ecmwf['latitude']),
        longitude = (rh_ecmwf['longitude'])),
    attrs=dict(description='2m Relative Humidity',units='%'))

    # set compression settings for saving the nc file
    compression_settings = {
    "dtype": "int16",  # Use 16-bit integers for storage
    "zlib": True,
    "complevel": 4,
    "scale_factor": 0.01,  # Adjust based on the precision you need
    "add_offset": 0.0     # Use to shift data range if necessary
    }

    # save netcdf file
    rh.to_netcdf(path_era5 + '/rh/' + 'era5_land_rh_'+str(year)+'_'+str(month)+'_global.nc' , 
                 format = 'NETCDF4_CLASSIC',
                 encoding={"rh": compression_settings})
