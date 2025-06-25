#!/usr/bin/env python
# coding: utf-8

#Loading needed packages to run the analysis as well as formatting for the plots (changes in the default runtime configuration rcParams).
import warnings
warnings.filterwarnings("ignore")
import HHB as PyHHB
import numpy as np 
import pylab as plt 
import glob
import argparse
import pandas as pd
import pickle
import matplotlib.gridspec as gridspec
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import xesmf as xe
import time as time

#Defining working directories 
path_era5 = '/uufs/chpc.utah.edu/common/home/u6053495/tnc-group1/data/era5_land/'
path_era5_output = '/scratch/general/vast/u6053495/output/'
workdir = '/uufs/chpc.utah.edu/common/home/u6053495/scripts/PyHHB/'
path_profiles  = workdir + 'personal_profiles/'
list_profiles = glob.glob(path_profiles+'*.txt') #To list all the profiles names in the folder
path_save = workdir + 'outputs/'#This is the folder in which all the outputs from this code will be saved.
path_icons = workdir + 'ancillary/'

parser = argparse.ArgumentParser(description="Calculate livability for one year and month of ERA5-Land data.")
parser.add_argument('--year', type=int, required=True, help='Year to process')
parser.add_argument('--month', type=str, required=True, help='Month to process (format: MM)')
args = parser.parse_args()
year = args.year
month = args.month

#Environmental settings
version = 'rh' # The humidity options are rh (relative humidity), wv (water vapor pressure), or q (specific humidity)
temperature = np.arange(25,60,0.1) #Create a range of temperatures we care about
sun = 'Night-Indoors' #Options are 'Night-Indoors','Day-Outdoors'
Av_ms = 1 #Air velocity in (m/s)
barometric_pressure = 1013 #in hPa ONLY to obtain water vapor pressure when specific humidity is used as humidity metric.
#Personal profile settings
name_profile = 'Young_adult' #Options are'Young_adult','65_over'
#Time of exposure (temporal resolution of the climate model output)
exp_time = 1 
#Read in the heat of vaporisation of sweat at 30⁰C, 2426 J/g
Lh_vap = PyHHB.Lh_vap

#creation of the temperature,humidity matrices according the values set in the previous cell.
#Dealing with different types of humidity (input model is Ambient vapour pressure in kPa)
if version == 'rh':
    humidity =  np.arange(0.5,100.5,0.5)
    hum_label = 'Relative humidity (%)'
    xx_temp, yy_humidity = np.meshgrid(temperature, humidity)
    Psa_kPa = 	PyHHB.Psa_kPa_from_TaC(xx_temp)#Saturated vapor pressure in kPa
    Pv  =   PyHHB.Pv_kPa_from_Psa_RH(Psa_kPa,yy_humidity) #Water vapor pressure in kPa

survivability      = pd.read_csv(path_save+version+'_survivability_array_'+sun+'_'+str(exp_time)+'H-'+name_profile+'.csv',index_col=0)  
flag_survivability = pd.read_csv(path_save+version+'_flag_survivability_'+sun+'_'+str(exp_time)+'H-'+name_profile+'.csv',index_col=0) 

profile = PyHHB.read_personal_profiles(path_profiles + name_profile+'_livability.txt')

if ~np.isfinite(float(profile['AD'])):
    AD    = PyHHB.AD_from_mass_height(float(profile['Mass']),float(profile['Height'])) #m2
else:
    AD = float(profile['AD'])

AD =  np.ones(xx_temp.shape)*AD

Ta_C = xx_temp
Pv_kPa  =   Pv
Av_ms  =  np.ones(xx_temp.shape)*Av_ms# Air velocity (m/s)

if sun == 'Night-Indoors':
    mrt_C  =  Ta_C#indoor mrt = Ta (⁰C)
elif sun == 'Day-Outdoors':
    mrt_C  =  Ta_C + 15 #outdoor mrt = Ta (⁰C) (Threshold to an hyphotetical Partly Cloudy Condition) based on the measurements from Guzman-Echavarria et al (2022)

else:
    raise ValueError("Non-valid radiation condition")
    
#Internal heat production at resting (W) equivalent to 1.5 METs
M =  np.ones(xx_temp.shape)* float(profile['M'])* float(profile['Mass']) #W/kg *Kg

# Defining properties to estimate the combined convective and radiative heat fluxes.
Tsk_C  = np.ones(xx_temp.shape)*float(profile['Tsk_C']) #Skin temperature, assumed to be held constant at 35°C
Emm_sk = np.ones(xx_temp.shape)*float(profile['Emm_sk']) #Area-weighted emissivity of the clothed body surface (dimensionless)
Ar_AD  = np.ones(xx_temp.shape)*float(profile['A_eff']) #Effective radiative area of the body (dimensionless)
Icl    = np.ones(xx_temp.shape)*float(profile['Icl']) #Insulation clothing value (CLO)


hc_cof  = PyHHB.hc_cof_from_Av(Av_ms) #W/m2K Convective heat transfer coefficient.
hr_cof  = PyHHB.hr_cof_from_radiant_features(mrt_C,Tsk_C,Emm_sk,Ar_AD) #Linear radiative heat transfer coefficient (W/m2K).
h_cof   = PyHHB.h_coef_from_hc_hr(hc_cof,hr_cof) # Combined heat transfer coefficient (W/m2K).
to_C    = PyHHB.to_from_hr_tr_hc_ta(hr_cof,mrt_C,hc_cof,Ta_C) #Operative temperature (⁰C)

#Estimation of combined dry heat loss via convection and radiation (W)
Dry_Heat_Loss =  PyHHB.Dry_Heat_Loss_c_plus_r(Tsk_C,to_C,Icl,h_cof,AD)

# Estimation of dry respiratory heat loss (W)
Cres = PyHHB.Cres_from_M_Ta(M,Ta_C,AD)
# Estimation of latent respiratory heat loss (W)
Eres = PyHHB.Eres_from_M_Pa(M,Pv_kPa,AD)

Re_cl = np.ones(xx_temp.shape)*float(profile['Re_cl']) #Evaporative resistance of clothing in m2·kPa/W

#Estimation of evaporative required heat loss (W) 
Ereq = PyHHB.Ereq_from_HeatFluxes(M,0,Dry_Heat_Loss,Cres, Eres)

#Water vapour pressure at the skin (kPa), assumed to be that of saturated water (100% HR) vapour at skin temperature.
Psk_s= PyHHB.Psa_kPa_from_TaC(Tsk_C) 
# Evaporative heat transfer coefficient (W/m2kPa).
he_cof = PyHHB.he_cof(hc_cof)

# Estimation of biophysical Emax (Env + clothing) (W)
Emax_env = PyHHB.Emax_env(Psk_s,Pv_kPa,Re_cl,he_cof,Icl,AD)
Emax_env[Emax_env<0] = 0 #This heat flux can not be negative.

# Maximum skin wettedness: maximum portion of total body surface area that can be saturated with sweat.
wmax = np.ones(xx_temp.shape)*PyHHB.wmax(profile['wmax_condition'])

# Physiological Emax (Env + clothing + wettedness) 
Emax_wettedness = PyHHB.Emax_wettedness(wmax,Psk_s,Pv_kPa,Re_cl,he_cof,Icl,AD)
Emax_wettedness[Emax_wettedness<0] = 0   #This heat flux can not be negative.  

# Estimation of required skin wettedness (dimensionless)
wreq = PyHHB.wreq_HSI_skin_wettedness(Ereq,Emax_env)
# Estimation of expected sweating efficiency (dimensionless)
r = PyHHB.Sweating_efficiency_r(wreq) 
#Estimation of required sweat rate  to maintain heat balance (L/h).
Sreq =  PyHHB.Sreq(Ereq,r,Lh_vap)

Smax = np.ones(Ta_C.shape)*float(profile['smax_rate']) #Maximum sweat rate (L/h).

#Estimation of the physiological evaporative heat loss based on capacity to secret sweat
Emax_sweat = PyHHB.Emax_sweat_rate(Smax,Lh_vap,1,r)

Mmax_W , survive_but_not_livable = PyHHB.livability_Mmax(flag_survivability,Ereq,Emax_wettedness,Emax_sweat,M)

Mmax_W[survivability == False] = np.nan
survive_but_not_livable = np.logical_and(survive_but_not_livable,survivability) #This line ensure too we doesn't extend past the new survivability curve

Mmax_MET = PyHHB.MetabolicRate_W_to_MET_Mass(Mmax_W,float(profile['Mass']))

def find_livable(t2m_in,rh_in):#,xx_temp,yy_humidity,Mmax_MET):
    xx_index = np.absolute(xx_temp[0,:] - t2m_in).argmin()
    yy_index = np.absolute(yy_humidity[:,0] - rh_in).argmin()
    livable = Mmax_MET[yy_index,xx_index]
    #Mmax_MET[yy_index,xx_index]
    return livable

# Load data with Dask and chunk for 24-hour periods
chunk_sizes = {"latitude": 100, "longitude": 100, "time": 24}  # Customize chunk sizes as needed
ds_mask = xr.open_dataset(path_era5 + '/masks/lsm_1279l4_0.1x0.1.grb_v4_unpack_ERA5LandGrid_New2025_01.nc')
mask = ds_mask['lsm'].mean(dim='time')

print(f"Processing month {month}")

# Open temperature and relative humidity data with chunking
ds_t2m = xr.open_dataset(
    path_era5 + f'/t2m/era5_land_t2m_{year}_{month}_global.nc',
    chunks=chunk_sizes
)
t2m = (ds_t2m['t2m'] - 273.15).where(mask > 0)

ds_rh = xr.open_dataset(
    path_era5 + f'/rh/era5_land_rh_{year}_{month}_global.nc',
    chunks=chunk_sizes
)
rh = ds_rh['rh'].where(mask > 0)

print("Data loaded and chunked")

# Apply the function over the entire chunked dataset
livable_map_month = xr.apply_ufunc(
    find_livable,  # Your livability function
    t2m,           # Temperature data
    rh,            # Relative humidity data
    input_core_dims=[[], []],  # Specify input dimensions
    output_core_dims=[[]],     # Specify output dimensions
    vectorize=True,            # Vectorize for Dask compatibility
    dask="parallelized",       # Enable parallel processing
    output_dtypes=[float]      # Specify output data type
)

# Wrap results in a dataset with 'mets' variable
mets = xr.Dataset(
    data_vars={
        "mets": (["time", "latitude", "longitude"], livable_map_month.data)
    },
    coords={
        "time": livable_map_month["valid_time"],
        "latitude": livable_map_month["latitude"],
        "longitude": livable_map_month["longitude"],
    },
    attrs={
        "description": "Maximum Metabolic Rate to Keep Compensable Heat Stress",
        "units": "METs"
    }
)

print(f"Saving results for month {month}")
compression_settings = {
    "dtype": "int16",
    "zlib": True,
    "complevel": 4,
    "scale_factor": 0.01,
    "add_offset": 0.0,
}

mets.to_netcdf(
    path_era5_output + f'/mets/mets_{year}_{month}_{exp_time}H_ERA5-Land_conditions{sun}_{exp_time}H-{name_profile}.nc',
    format="NETCDF4_CLASSIC",
    encoding={"mets": compression_settings}
)