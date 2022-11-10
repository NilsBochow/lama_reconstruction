import numpy as np 
from PIL import Image
import h5py
from matplotlib import pyplot as plt
#I used 0-100 for ice, 200 for continents and 122 for missing values


def normalize_sea_ice(array):
    return array/100*255 

def generate_sic_mask(): 
    img_file_path = "/p/tmp/bochow/LAMA/lama/sic/sea_ice_cover_3D_256x256_masked.h5"
    
    h5_file = h5py.File(img_file_path, 'r')

    
    hdata = h5_file.get('Dataset1')
    plt.imshow(hdata[700,:,:])
    plt.savefig("test_sic_1.png", dpi = 600)
    hdata_copy = hdata[:,:,:]
    mask = np.zeros_like(hdata_copy)
    mask[hdata_copy == 122.] = 1
    #hf = h5py.File('/p/tmp/bochow/LAMA/lama/sic/sic_missmask.h5', 'w')
    #hf.create_dataset('sic', data=mask, dtype=np.single)
    
    for i in range(hdata.shape[0]): 
        h_data_edited = hdata_copy[i,:,:]
        h_data_edited[h_data_edited>100] = 0
        h_data_edited[h_data_edited<0] = 0
        h_data_edited =  normalize_sea_ice(h_data_edited)
        img = np.repeat(h_data_edited[:, :, np.newaxis], 3, axis=2)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(f"/p/tmp/bochow/LAMA/lama/sic/raw/sic{i:06}.png")
    

    
generate_sic_mask()