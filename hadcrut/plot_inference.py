import numpy as np 
import cv2
import netCDF4
from PIL import Image
from matplotlib import pyplot as plt
import glob 
from matplotlib.colors import DivergingNorm
import cartopy.crs as ccrs

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

def load_eval_images(name): 
    img_gt = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/eval/fixed_72.yaml/'+ name + '_val_crop000.png')
    img_inpainted = Image.open('/p/tmp/bochow/LAMA/lama/inference/hadcrut/fixed_72/' + name + '_val_crop000_mask000.png') 
    mask = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/eval/fixed_72.yaml/' + name + '_val_crop000_mask000.png') 
    
    img_gt = unnoramlize(np.array(img_gt))
    #im_gt[im_gt<-1000] = np.nan
    img_inpainted = unnoramlize(np.array(img_inpainted))
    mask = unnoramlize(np.array(mask))

    fig = plt.figure(figsize=(20, 5))
    ax0 = fig.add_subplot(1, 4, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 4, 2, projection=ccrs.Robinson())
    ax2 = fig.add_subplot(1, 4, 3, projection=ccrs.Robinson())
    ax3 = fig.add_subplot(1, 4, 4, projection=ccrs.Robinson())

    img_extent = (0, 360, -90, 90)

    

    print(np.min(im_gt))

    ax0.set_global()
    ax0.coastlines()
    ax1.set_global()
    ax1.coastlines()
    ax2.set_global()
    ax2.coastlines()
    ax3.set_global()
    ax3.coastlines()

    min_imshow = np.min(img_gt)
    max_imshow = np.max(img_gt)
    im1 = ax0.imshow(img_gt[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax1.imshow(img_inpainted[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax2.imshow(img_gt[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax2.imshow(mask, cmap = "binary", transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower", alpha = 0.2)
    ax3.imshow(img_gt[:,:,0] - img_inpainted[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax0.set_title("GT")
    ax1.set_title("inpainted")
    ax2.set_title("mask + GT")
    ax3.set_title("difference GT-inpainted")

    fig.subplots_adjust(wspace=0, hspace=0)

    #fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.1])
    fig.colorbar(im1, cax=cbar_ax, orientation="horizontal", label= "Temperature [K]", pad=0.01)

    plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/eval_pics/" + name + ".png", dpi=600, bbox_inches = "tight")

    """
    fig2, axs2 = plt.subplots(1, 3, figsize=(9, 3))
    im2 = axs2[0].imshow(img_gt[:,:,0], cmap = "coolwarm")
    axs2[1].imshow(img_gt[:,:,1], cmap = "coolwarm")
    axs2[2].imshow(img_gt[:,:,2], cmap = "coolwarm")
    axs2[0].set_title("GT1")
    axs2[1].set_title("GT2")
    axs2[2].set_title("GT3")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im2, cax=cbar_ax)

    plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/eval_pics/" + name + "_3channels.png")

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
        img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/train/cmip{i:06}.png")
    """



def load_eval_images_hadcrut(name): 
    img_gt = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/'+ name + '_crop000.png')
    img_inpainted = Image.open('/p/tmp/bochow/LAMA/lama/inference/hadcrut/fixed_72/' + name + '_crop000_mask000.png') 
    mask = Image.open('/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/' + name + '_crop000_mask000.png') 

    img_gt = unnoramlize(np.array(img_gt))
    img_inpainted = unnoramlize(np.array(img_inpainted))
    mask = unnoramlize(np.array(mask))

    fig = plt.figure(figsize=(20, 5))
    ax0 = fig.add_subplot(1, 3, 1, projection=ccrs.Robinson(central_longitude=180))
    ax1 = fig.add_subplot(1, 3, 2, projection=ccrs.Robinson(central_longitude=180))
    ax2 = fig.add_subplot(1, 3, 3, projection=ccrs.Robinson(central_longitude=180))
    #ax3 = fig.add_subplot(1, 4, 4, projection=ccrs.Robinson())

    img_extent = (0, 360, -90, 90)


    ax0.set_global()
    ax0.coastlines()
    ax1.set_global()
    ax1.coastlines()
    ax2.set_global()
    ax2.coastlines()
    #ax3.set_global()
    #ax3.coastlines()

    min_imshow = -3#np.min(img_gt)
    max_imshow = 3#np.max(img_gt)
    img_gt[mask==1] = np.nan
    im1 = ax0.imshow(img_gt[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax1.imshow(img_inpainted[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax2.imshow(img_gt[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax2.imshow(mask, cmap = "binary", transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower", alpha = 0.2)
    #ax3.imshow(img_gt[:,:,0] - img_inpainted[:,:,0], norm=DivergingNorm(0), cmap = "coolwarm", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "lower")
    ax0.set_title("GT")
    ax1.set_title("inpainted")
    ax2.set_title("mask + GT")
    #ax3.set_title("difference GT-inpainted")

    fig.subplots_adjust(wspace=0, hspace=0)

    #fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.1])
    fig.colorbar(im1, cax=cbar_ax, orientation="horizontal", label= "Temperature [K]", pad=0.01)

    plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/eval_pics/" + name + ".png", dpi=600, bbox_inches = "tight")

    """
    fig2, axs2 = plt.subplots(1, 3, figsize=(9, 3))
    im2 = axs2[0].imshow(img_gt[:,:,0], cmap = "coolwarm")
    axs2[1].imshow(img_gt[:,:,1], cmap = "coolwarm")
    axs2[2].imshow(img_gt[:,:,2], cmap = "coolwarm")
    axs2[0].set_title("GT1")
    axs2[1].set_title("GT2")
    axs2[2].set_title("GT3")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im2, cax=cbar_ax)

    plt.savefig("/p/tmp/bochow/LAMA/lama/hadcrut/eval_pics/" + name + "_3channels.png")

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
        img.save(f"/p/tmp/bochow/LAMA/lama/hadcrut/train/cmip{i:06}.png")
    """

name_list = glob.glob("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut_missing_masks/fixed_72.yaml/*crop000.png")
cropped_list =  [w[-17:-12] for w in name_list]
print(cropped_list)
print("fuck")

for i in ["00329","00330"]:#cropped_list:
    print(i)
    load_eval_images_hadcrut("hadcrut0"+ i)
