import numpy as np 
import h5py
from matplotlib import pyplot as plt
from PIL import Image
import netCDF4
from scipy.interpolate import NearestNDInterpolator

def normalize_sea_ice(array):
    return array/100*255 


def interpolate(array): 
    mask = np.where(~np.isnan(array))
    interp = NearestNDInterpolator(np.transpose(mask), array[mask])
    image_result = interp(*np.indices(array.shape))

    return image_result

def plot_cmip6():
    img_file_path = "/p/tmp/bochow/LAMA/lama/era5-dataset/ERA5/" + "1980" + "_masked.h5"
    print(img_file_path)
    h5_file = h5py.File(img_file_path, 'r')

    cmip256 = np.zeros((256,256))
    hdata = h5_file.get('Dataset1')
    hdata_copy = hdata[:,:,:]
    print(hdata_copy.shape)
    plt.imshow(hdata_copy[-1,:,:], vmin =0, vmax = 100)
    plt.savefig("/p/tmp/bochow/LAMA/lama/era5-dataset/test_plot.png")
    sic_cmip6 = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/sic/cmip6/" + "siconc_SImon_EC-Earth3_historical_r1i1p1f1_gn_12.nc")["siconc"][:]
    sic_cmip6_1month = sic_cmip6[6,0:180,:]
    sic_cmip6_1month[sic_cmip6_1month>100] = np.nan 

    cmip256[0:180,:] =sic_cmip6_1month
    cmip256[hdata_copy[-1,:,:] == 200] =120 
    cmip256 = interpolate(cmip256)
    plt.clf()
    plt.imshow(cmip256, vmin =0, vmax = 100)
    plt.savefig("/p/tmp/bochow/LAMA/lama/era5-dataset/test_plot_cmip6.png")
    

def save_cmip6(): 
    img_file_path = "/p/tmp/bochow/LAMA/lama/era5-dataset/ERA5/" + "1980" + "_masked.h5"
    h5_file = h5py.File(img_file_path, 'r')

    cmip256 = np.zeros((256,256))
    hdata = h5_file.get('Dataset1')
    hdata_copy = hdata[:,:,:]

    sic_cmip6 = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/sic/cmip6/" + "siconc_SImon_EC-Earth3_historical_r1i1p1f1_gn_12.nc")["siconc"][:]
    for i in range(sic_cmip6.shape[0]):
        sic_cmip6_1month = sic_cmip6[i,0:180,:]
        sic_cmip6_1month[sic_cmip6_1month>100] = np.nan 

        cmip256[0:180,:] =sic_cmip6_1month
        cmip256 = interpolate(cmip256)
        cmip256[hdata_copy[-1,:,:] == 200] =120 
        
        cmip256[cmip256<0] = 0
        cmip256[cmip256>101] = 0
        
        cmip256 = normalize_sea_ice(cmip256)
        
        img = np.repeat(cmip256[:, :, np.newaxis], 3, axis=2)
        img = Image.fromarray(img.astype(np.uint8))
        img.save("/p/tmp/bochow/LAMA/lama/sic/cmip6/raw/" + f"{i:02}_sic_cmip6.png")

save_cmip6()