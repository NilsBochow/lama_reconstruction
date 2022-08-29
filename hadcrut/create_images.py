import numpy as np 
import cv2
import netCDF4
from PIL import Image

def create_train_images(): 
    img_file = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/20cr-val.nc")["tas"] 
    length = img_file.shape[0]
    min_value = np.min(img_file)
    max_value = np.max(img_file-min_value)
    print(min_value, max_value)
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
        img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/val_source_cr/20cr{i:06}_val.png")

create_train_images()