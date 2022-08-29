import numpy as np 
import h5py
from matplotlib import pyplot as plt
from PIL import Image


def test_masks(): 
    mask_file = h5py.File("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut4-missmask.h5", 'r')
    maskdata_1 = mask_file["tas"]

    mask_file = h5py.File("/p/tmp/bochow/LAMA/lama/hadcrut/mask_hadcrut_own.h5", 'r')
    maskdata_2 = mask_file["tas"]
    print(maskdata_1.shape, maskdata_2.shape)

    plt.imshow(maskdata_1[0,:,:])
    plt.colorbar()
    plt.savefig("maskdata1.png")

    plt.imshow(np.roll(maskdata_2[0,:,:], int(maskdata_2.shape[0]/2), axis=1))
    plt.colorbar()
    plt.savefig("maskdata2.png")


test_masks()