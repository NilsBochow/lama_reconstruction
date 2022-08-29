import numpy as np 
import netCDF4
from PIL import Image
from matplotlib import pyplot as plt
import glob 
from matplotlib.colors import DivergingNorm
import cartopy.crs as ccrs
import h5py 

mask = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missmask_1.nc")["tas"]
plt.imshow(mask[0,:,:])
plt.savefig("test_mask.png")

def upscale_hadcrut():
    tas = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.4.6.0.0.anomalies.1.nc")["temperature_anomaly"] 
    length = tas.shape[0]
    tas_72_file = np.repeat(tas, 2,axis=1)
    print(tas_72_file[0,:,:])
    plt.imshow(tas_72_file[0,:,:])
    plt.savefig("test_tas_2.png")
    
    mask = np.zeros(tas_72_file.shape)
    mask[tas_72_file<-2000] = 1
    tas_72_file[tas_72_file<-2000] = np.nan
    print(mask.shape)
    mask = np.roll(mask[:,:,:], int(mask.shape[1]/2), axis=2)
    plt.imshow(mask[0,:,:])
    plt.savefig("test_mask_2.png")
    hf = h5py.File('/p/tmp/bochow/LAMA/lama/hadcrut/mask_hadcrut_own.h5', 'w')
    hf.create_dataset('tas', data=mask)

    min_value = np.nanmin(tas_72_file)
    max_value = np.nanmax(tas_72_file-min_value)
    print(min_value, max_value)
    
    for i in range(length): 
        
        tas_72 = tas_72_file[i,:,:]
        tas_72[tas_72==np.nan] = 0
        print(np.nanmin(tas_72), np.nanmax(tas_72))
        tas_72 = (tas_72 -min_value)
        tas_72 = ((tas_72)/max_value*255).astype('uint8')
        tas_72 = np.roll(tas_72, int(tas_72.shape[1]/2), axis = 1)
        print(tas_72.shape)
        #mask = Image.fromarray(mask)

        tas_72 = np.repeat(tas_72[:, :, np.newaxis], 3, axis=2)
        tas_72 = Image.fromarray(tas_72)
        tas_72.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing/hadcrut{i:06}.png")
        #mask.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing/hadcrut{i:06}_mask.png")
    

upscale_hadcrut()