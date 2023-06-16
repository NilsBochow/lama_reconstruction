import numpy as np 
import cv2
import netCDF4
from PIL import Image
from matplotlib import pyplot as plt
import glob 
from matplotlib.colors import DivergingNorm
import cartopy.crs as ccrs
import matplotlib

def get_min_max(): 
    img_file = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.4.6.0.0.anomalies.1.nc")["temperature_anomaly"][:,:,:]
    length = img_file.shape[0]
    img_file[img_file<-1000]=np.nan
    min_value = np.nanmin(img_file)
    #min_value = -33.276234
    #max_value = 56.640396
    max_value = np.nanmax(img_file-min_value)
    return min_value, max_value

def unnoramlize(array): 
    min_value, max_value = get_min_max()
    array = array/255*max_value + min_value
    return array



cmip_kadow = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/cmipAI_HadCRUT4_4.6.0.0_tas_mon_185001-201812.nc")["tas"]
print(cmip_kadow.shape)

#hadcrut5_infilled = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.5.0.1.0.analysis.anomalies.1.nc")["tas"]
hadcrut5_infilled = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc")["tas_mean"]

print(cmip_kadow.shape, hadcrut5_infilled.shape)

def rmse(name, index): 
    img_gt = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/'+ name + '_crop000.png')
    img_inpainted = Image.open('/p/tmp/bochow/LAMA/lama/inference/hadcrut/fixed_72_10days/' + name + '_crop000_mask000.png') 
    mask = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/' + name + '_crop000_mask000.png') 
    print(int(name[-5:]))
    kadow_inpainted = cmip_kadow[int(name[-5:])-240,:,:]
    kadow_inpainted = np.repeat(kadow_inpainted, 2,axis=0)

    hadcrut5_infilled_72 = np.repeat(hadcrut5_infilled[int(name[-5:]),:,:], 2,axis=0)

    img_gt = unnoramlize(np.array(img_gt))
    img_inpainted = unnoramlize(np.array(img_inpainted))[:,:,0]
    print(img_inpainted.shape)
    mask = (np.array(mask))
    print("mask shape:", mask.shape)
    difference = img_inpainted - kadow_inpainted
    #plt.imshow(difference)
    #plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/rmse_kadow_lama.png")
    mask_nan = np.zeros_like(mask, dtype="float32")
    print("mask nan shape:", mask_nan.shape)
    mask_nan[mask==255] = np.nan
    number_of_measurements = np.count_nonzero(np.isnan(mask_nan))
    print(number_of_measurements)
    rmse_array[index] = np.sqrt((np.nansum((difference[mask==255])**2)))/np.sqrt(number_of_measurements)
    print(rmse_array)
    
    
    
name_list = glob.glob("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/*crop000.png")
cropped_list =  [w[-17:-12] for w in name_list]
rmse_array = np.zeros(len(cropped_list))
rmse_spatial_array = np.zeros((72,72))
min_max = np.load("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_norm2/min_max_tas.npz")

def unnormalize_global(array, index):
    print(index)
    min_value = min_max["min_value"][index]
    max_value = min_max["max_value"][index]
    array = array/255*max_value + min_value
    return array
    
def save_netcdf_reconstructed():
    tas = np.zeros((len(cropped_list), 72, 72))
    mask_missing = np.zeros_like(tas)
    tas_gt = np.zeros_like(tas)
    ts_hc5 = np.zeros_like(tas)
    ts_hc5[:] = np.repeat(netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc")["tas_mean"][0:2064], 2,axis=1)
    ts_hc5 = np.roll(ts_hc5[:,:,:], int(ts_hc5.shape[1]/2), axis=2)
    ts_kadow = np.zeros_like(tas) 
    ts_kadow[:] = np.nan 
    kadow_inpainted = cmip_kadow
    kadow_inpainted = np.repeat(kadow_inpainted, 2,axis=1)
    ts_kadow[240:240+1632,:,:] = kadow_inpainted
    for i, name in enumerate(cropped_list): 
        name_string = "hadcrut0" +  name
        img_inpainted = Image.open('/p/tmp/bochow/LAMA/lama/inference/hadcrut/fixed_72_june_train2/' + name_string + '_crop000_mask000.png') 
        img_gt = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/'+ name_string + '_crop000.png')
        mask = np.array(Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/' + name_string + '_crop000_mask000.png'))
        #img_inpainted = unnoramlize(np.array(img_inpainted))[:,:,0] ## for local min/max 
        img_inpainted = unnormalize_global(np.array(img_inpainted), int(name))[:,:,0] ## for global max/min
        img_gt = unnoramlize(np.array(img_gt))[:,:,0]
        tas[int(name),:,:] = img_inpainted
        mask_missing[int(name),:,:] = mask[:,:]
        img_gt[mask==255] = np.nan
        tas_gt[int(name),:,:] = img_gt
        
    time = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.4.6.0.0.anomalies.1.nc")["time"][:]
    ncout = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4_fixed_72_june_train2.nc','w'); 

    ncout.createDimension('time', len(cropped_list));
    ncout.createDimension('lat', 72);
    ncout.createDimension('lon', 72);
    timevar = ncout.createVariable('time','float32',('time'));
    timevar.setncattr('units','days since 1850-1-1 00:00:00')
    timevar.setncattr('standard_name','time')
    timevar.setncattr('long_name','Time (days after 1850)')
    
    timevar.setncattr('start_year', '1850s') ;
    timevar.setncattr('end_year', '2021s') ;
    timevar.setncattr('start_month', '1s') ;
    timevar.setncattr('end_month', '12s') ;
    timevar.setncattr('calender','gregorian')

    timevar[:] = time


    myvar = ncout.createVariable('tas','float32',('time', 'lat', 'lon'));myvar.setncattr('units','Kelvin');
    myvar.setncattr('standard_name','temperature at surface (2m)')
    myvar.setncattr('long_name','surface_temperature')
    myvar[:] = tas;
    
    
    myvar_mask = ncout.createVariable('mask','float32',('time', 'lat', 'lon'));myvar_mask.setncattr('units','255=mising');
    myvar_mask.setncattr('standard_name','mask missing values)')
    myvar_mask.setncattr('long_name','missing_mask')
    myvar_mask[:] = mask_missing
    
    myvar_gt = ncout.createVariable('tas_gt','float32',('time', 'lat', 'lon'));myvar_gt.setncattr('units','Kelvin');
    myvar_gt.setncattr('standard_name','temperature at surface ground-truth (2m)')
    myvar_gt.setncattr('long_name','surface_temperature_gt')
    myvar_gt[:] = tas_gt;
        
    myvar_hc5 = ncout.createVariable('tas_hc5','float32',('time', 'lat', 'lon'));myvar_hc5.setncattr('units','Kelvin');
    myvar_hc5.setncattr('standard_name','temperature at surface HadCRUT5 (2m)')
    myvar_hc5.setncattr('long_name','surface_temperature_hc5')
    myvar_hc5[:] = ts_hc5;
        
    myvar_kadow = ncout.createVariable('tas_kadow','float32',('time', 'lat', 'lon'));myvar_kadow.setncattr('units','Kelvin');
    myvar_kadow.setncattr('standard_name','temperature at surface Kadow (2m)')
    myvar_kadow.setncattr('long_name','surface_temperature_kadow')
    myvar_kadow[:] = ts_kadow
        
    myvar_lat= ncout.createVariable('lat','float32',('lat'));myvar_lat.setncattr('units','degreeN');
    myvar_lat.setncattr('standard_name','latitude')
    myvar_lat.setncattr('long_name','Latitude')
    myvar_lat[:] = np.arange(-90,90,2.5);

    myvar_lon= ncout.createVariable('lon','float32',('lon'));myvar_lon.setncattr('units','degreeE');
    myvar_lon.setncattr('standard_name','longitude')
    myvar_lon.setncattr('long_name','Longitude')
    myvar_lon[:] = np.arange(0,360,5)
    ncout.close();

save_netcdf_reconstructed()
    
sys.exit() 
def rmse_spatial():  
    lama_reconstructed = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/" + 'reconstructed_lama_hadcrut4.nc')["tas"][:]

def rmse_spatial(): 
    difference = np.zeros((cmip_kadow.shape[0], 72,72))
    difference[:] = np.nan
    mask_3D = np.zeros((cmip_kadow.shape[0], 72,72))
    counter = 0
    for i, name in enumerate(cropped_list):
        if ((int(name) < 240) or (int(name)-240>1630)):
            continue
        else:
            name = "hadcrut0" +  name
            img_gt = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/'+ name + '_crop000.png')
            img_inpainted = Image.open('/p/tmp/bochow/LAMA/lama/inference/hadcrut/fixed_72_10days/' + name + '_crop000_mask000.png') 
            mask = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/' + name + '_crop000_mask000.png') 
            print(int(name[-5:]))
            kadow_inpainted = cmip_kadow[int(name[-5:])-240,:,:]
            kadow_inpainted = np.repeat(kadow_inpainted, 2,axis=0)

            hadcrut5_infilled_72 = np.repeat(hadcrut5_infilled[int(name[-5:]),:,:], 2,axis=0)

            img_gt = unnoramlize(np.array(img_gt))
            img_inpainted = unnoramlize(np.array(img_inpainted))[:,:,0]
            mask = (np.array(mask))
            mask_nan = np.zeros_like(mask, dtype="float32")
            mask_nan[mask==255] = np.nan

            difference[counter,mask==255] = img_inpainted[mask==255] - kadow_inpainted[mask==255]
            mask_3D[counter,:,:] = mask_nan
            counter = counter + 1
    #number_of_measurements = np.count_nonzero(np.isnan(mask_3D), axis=0 )
    number_of_measurements = np.count_nonzero(~np.isnan(difference), axis=0)
    #rmse_spatial_array = np.sqrt((np.nansum((difference[np.isnan(mask_3D)])**2, axis=0)))/np.sqrt(number_of_measurements)
    rmse_spatial_array = np.sqrt((np.nansum((difference[:, :, :])**2, axis = 0)))/np.sqrt(number_of_measurements)
    print(np.nanmax(rmse_spatial_array))
    plt.imshow(rmse_spatial_array)
    plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/rmse_kadow_lama_spatial.png")
    
rmse_spatial()
sys.exit()
for i, name in enumerate(cropped_list):
    if ((int(name) < 240) or (int(name)+240>1631)):
        continue 
    rmse("hadcrut0"+ name, i)

plt.clf()
plt.plot(rmse_array)
plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/rmse_kadow_lama_ts.png")