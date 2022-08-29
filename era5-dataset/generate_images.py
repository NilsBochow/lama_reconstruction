import numpy as np 
import h5py
from matplotlib import pyplot as plt
from PIL import Image

def normalize_sea_ice(array):
    return array/100*255 


def create_train_images(month, year): 
    img_file_path = "/p/tmp/bochow/LAMA/lama/era5-dataset/" + year + "/"+ month + "-" + year + ".h5"
    print(img_file_path)
    h5_file = h5py.File(img_file_path, 'r')

    
    hdata = h5_file.get('Dataset1')
    hdata_copy = hdata[:,:,:]

    hdata_copy[hdata_copy<0] = 0
    hdata_copy[hdata_copy>120] = 0


    length = hdata_copy.shape[0]

    hdata_copy = normalize_sea_ice(hdata_copy)
    min_value = np.min(hdata_copy)
    max_value = np.max(hdata_copy-min_value)
    for i in range(length):
        hdata_daily = hdata_copy[i,:,:]

        img = np.repeat(hdata_daily[:, :, np.newaxis], 3, axis=2)

        img = Image.fromarray(img.astype(np.uint8))
        img.save("/p/tmp/bochow/LAMA/lama/era5-dataset/train/" + year + "_" + month + "_"  + f"{i:02}_era5.png")

        plt.imshow(hdata_copy[0,:,:])
        plt.savefig("test.png")
    """
    for i in range(length): 
        img = img_file[i,:,:]
    #normalize to positive
        img = (img -min_value)
        img = ((img)/max_value*255).astype('uint8')

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        img = Image.fromarray(img)
        #print(img.shape)
        #img = Image.fromarray(img)
        #img2 = img.convert('L')
        #img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/train/cmip{i:06}.png")
    """ 

for year in np.arange(2015,2021):
    print(year)
    for month in range(1,13):
        print(month)
        create_train_images(f"{month}", f"{year}")