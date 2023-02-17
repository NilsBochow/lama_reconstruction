import numpy as np 
import h5py
from matplotlib import pyplot as plt
from PIL import Image

def normalize_sea_ice(array):
    return array/100*255 


def create_train_images(year): 
    img_file_path = "/p/tmp/bochow/LAMA/lama/era5-dataset/ERA5/" + year + "_masked.h5"
    print(img_file_path)
    h5_file = h5py.File(img_file_path, 'r')

    
    hdata = h5_file.get('Dataset1')
    hdata_copy = hdata[:,:,:]

    length = hdata_copy.shape[0]
    random_vector_eval = np.unique(np.random.randint(0, length, 59)) #2100 random
    index_array_random_1 = np.random.randint(0, random_vector_eval.size, 6) #250 random from the 2100 random
    print(index_array_random_1.size, random_vector_eval.size)


    random_vector_visual_test = random_vector_eval[index_array_random_1] # 250 pics for visual test
    random_vector_eval = np.delete(random_vector_eval, index_array_random_1) #rest for eval 
    print(random_vector_eval.size, random_vector_eval)
    random_vector_val = random_vector_eval[0:int(random_vector_eval.size/2)]
    random_vector_eval = random_vector_eval[int(random_vector_eval.size/2)::]
    print(random_vector_val)
    hdata_copy[hdata_copy<0] = 0
    hdata_copy[hdata_copy>101] = 0

    #train_indices = np.delete(np.arange(0,length), np.array(random_vector_eval, random_vector_visual_test))
    #train_indices = np.delete(train_indices, random_vector_visual_test)
    hdata_copy = normalize_sea_ice(hdata_copy)
    min_value = np.min(hdata_copy)
    max_value = np.max(hdata_copy-min_value)
    for i in range(length):
        hdata_daily = hdata_copy[i,:,:]

        img = np.repeat(hdata_daily[:, :, np.newaxis], 3, axis=2)

        img = Image.fromarray(img.astype(np.uint8))
        if i in random_vector_eval: 
            print(i, "eval")
            img.save("/p/tmp/bochow/LAMA/lama/era5-dataset/eval/" + year  + "_"  + f"{i:02}_era5.png")
        elif i in random_vector_val:
            print(i, "val")
            img.save("/p/tmp/bochow/LAMA/lama/era5-dataset/val/" + year +  "_"  + f"{i:02}_era5.png")
        elif i in random_vector_visual_test:
            print(i, "visual test") 
            img.save("/p/tmp/bochow/LAMA/lama/era5-dataset/visual_test/" + year + "_"  + f"{i:02}_era5.png")
        else:
            img.save("/p/tmp/bochow/LAMA/lama/era5-dataset/train/" + year  + "_"  + f"{i:02}_era5.png")
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

for year in [1998, 1999]:
    create_train_images(f"{year}")