import numpy as np 
import cv2
import netCDF4
from PIL import Image
from matplotlib import pyplot as plt
import glob 
from matplotlib.colors import DivergingNorm
import cartopy.crs as ccrs
import matplotlib




lama_reconstructed = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4_fixed_72_june_train2.nc')["tas"][:]
#lama_reconstructed_2 = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4_norm2.nc')["tas"][:]




lama_reconstructed = np.mean(np.reshape(lama_reconstructed, (2064,36,2,72)), axis=2)
plt.imshow(lama_reconstructed[(1979-1850)*12,:,:], vmin=-3, vmax=3)
plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/lama_36x72.png")
print(lama_reconstructed.shape)
#mask = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4.nc')["mask"][:]
mask = [netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'HadCRUT.4.6.0.0.anomalies.1.nc')["temperature_anomaly"][:] < -100]

kadow = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4.nc')["tas_kadow"][:]
kadow = kadow[:,::2,:]
plt.clf()
plt.imshow(kadow[(1979-1850)*12,:,:], vmin=-3, vmax=3)
plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/kadow_36x72.png")
hadcrut5 = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4_fixed_72_june_train2.nc')["tas_hc5"][:]
#hadcrut5 = np.roll(hadcrut5[:,:,:], int(hadcrut5.shape[1]/2), axis=2)
hadcrut5 = hadcrut5[:,::2,:]
plt.clf()
plt.imshow(hadcrut5[(1979-1850)*12,:,:], vmin=-3, vmax=3)
plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut5_36x72.png")
mask_hc5 = [hadcrut5<-100]
hc5_lama = lama_reconstructed - hadcrut5 

hc5_kadow = kadow - hadcrut5 
#era5_anomaly = np.mean(netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'era5_1960_90_72x72.nc')["t2m"][:], axis=0) - 273.15
#era5_tas = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'era5_72x72.nc')["t2m"][:] - 273.15 - era5_anomaly

#era5_lama = lama_reconstructed[(1979-1850)*12:(2005-1850)*12,:,:] - era5_tas[0:(2005-1979)*12,:,:]
#era5_kadow = kadow[(1979-1850)*12:(2005-1850)*12,:,:] - era5_tas[0:(2005-1979)*12,:,:]
#plt.clf()
#plt.imshow(era5_tas[0,:,:])
#plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/era5_tas.png")

#era5_lama[mask[(1979-1850)*12:(2005-1850)*12,:,:] == 255] = np.nan 
#era5_kadow[mask[(1979-1850)*12:(2005-1850)*12,:,:] == 255] = np.nan 

#number_of_measurements = np.count_nonzero(~np.isnan(era5_lama))
#rmse_spatial_array = np.sqrt((np.nansum((era5_lama[:, :, :])**2)))/np.sqrt(number_of_measurements)
#rmse_spatial_array_kadow = np.sqrt((np.nansum((era5_kadow[:, :, :])**2)))/np.sqrt(number_of_measurements)
#print("lama:", rmse_spatial_array)
#print("kadow:", rmse_spatial_array_kadow)
#sys.exit() 

hc5_lama[mask_hc5] = np.nan 
hc5_lama[mask == 255] = np.nan 
#hc5_lama[mask_hc4] = np.nan 
hc5_lama = hc5_lama[240:240+1632]

hc5_kadow[mask_hc5] = np.nan 
hc5_kadow[mask == 255] = np.nan 
#hc5_kadow[mask_hc4] = np.nan 
hc5_kadow = hc5_kadow[240:240+1632]

number_of_measurements = np.count_nonzero(~np.isnan(hc5_lama))
number_of_measurements_kadow = np.count_nonzero(~np.isnan(hc5_kadow))

#rmse_spatial_array = np.sqrt((np.nansum((difference[np.isnan(mask_3D)])**2, axis=0)))/np.sqrt(number_of_measurements)
rmse_spatial_array = np.sqrt((np.nansum((hc5_lama[:, :, :])**2)))/np.sqrt(number_of_measurements)
rmse_spatial_array_kadow = np.sqrt((np.nansum((hc5_kadow[:, :, :])**2)))/np.sqrt(number_of_measurements_kadow)
print("lama:", rmse_spatial_array)
print("kadow:", rmse_spatial_array_kadow)