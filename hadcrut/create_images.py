import numpy as np 
import cv2
import netCDF4
from PIL import Image

def create_train_images(): 
    img_file = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/20cr-val.nc")["tas"] 
    length = img_file.shape[0]
   
    print(min_value, max_value)
    for i in range(length): 
        img = img_file[i,:,:]
        min_value = np.min(img)
        max_value = np.max(img-min_value)
    #normalize to positive
        img = (img -min_value)
        img = ((img)/max_value*255).astype('uint8')

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        img = Image.fromarray(img)
        #print(img.shape)
        #img = Image.fromarray(img)
        #img2 = img.convert('L')
        img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/val_source_cr/20cr{i:06}_val.png")

#create_train_images()


def create_train_images_cmip(): 
    img_file = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/cmip-val.nc")["tas"] 
    length = img_file.shape[0]
    random_vector_eval = np.unique(np.random.randint(0, length, 2600))
    index_array_random = np.random.randint(0, random_vector_eval.size, 250)
    print(index_array_random.size, random_vector_eval.size)
    random_vector_visual_test = random_vector_eval[index_array_random]
    random_vector_eval = np.delete(random_vector_eval, index_array_random)
    print(random_vector_visual_test.size, random_vector_eval.size)


    for i in range(length): 
        img = img_file[i,:,:]
        min_value = np.min(img)
        max_value = np.max(img-min_value)
        #print(min_value, max_value)
        #normalize to positive
        img = (img -min_value)
        img = ((img)/max_value*255).astype('uint8')

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        img = Image.fromarray(img)

        if i in random_vector_eval: 
            img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/eval_source/cmip{i:06}_val.png")
        elif i in random_vector_visual_test:
            img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/visual_test_source/cmip{i:06}_val.png")
        else:
            img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/val_source/cmip{i:06}_val.png")

    img_file = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/cmip-train.nc")["tas"] 
    length = img_file.shape[0]
   

    for i in range(length): 
        img = img_file[i,:,:]
        min_value = np.min(img)
        max_value = np.max(img-min_value)
        #print(min_value, max_value)
        #normalize to positive
        img = (img -min_value)
        img = ((img)/max_value*255).astype('uint8')

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        img = Image.fromarray(img)

        img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/train/cmip{i:06}_train.png")

create_train_images_cmip()