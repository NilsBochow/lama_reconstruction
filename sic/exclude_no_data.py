import numpy as np 
from PIL import Image
import h5py
from matplotlib import pyplot as plt
#I used 0-100 for ice, 200 for continents and 122 for missing values
import cartopy.crs as ccrs
import netCDF4

def exclude_no_data_images(): 
    img_file_path = "/p/tmp/bochow/LAMA/lama/sic/sea_ice_cover_3D_256x256_masked.h5"
    
    
    h5_file = h5py.File(img_file_path, 'r')
    hdata = h5_file.get('Dataset1')
    number_cells =  np.count_nonzero(hdata[0,:,:]==122)
    #test = np.count_nonzero((hdata[0,:,:]>0) & (hdata[0,:,:]<100))
    #print(test)
    number_per_month = np.zeros(hdata.shape[0])
    percentages_per_month = np.zeros_like(number_per_month)
    for i in range(hdata.shape[0]): 
        number_per_month[i] = np.count_nonzero(hdata[i,:,:]==122)
    percentages_per_month = number_per_month/number_cells * 100
    index_95 = np.argwhere(percentages_per_month<99)
    
    return index_95

def unnormalize_pic(array): 
    array = array*(100/255)
    return array[0:180,:]

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx / (1000**2)

    xda = DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r


def plot_sea_ice_inference(): 
    
    indices = exclude_no_data_images()[:,0]
    for i in indices: 
        img_gt = Image.open(f'/p/tmp/bochow/LAMA/lama/sic/raw/sic{i:06}.png') 
        img_inpainted = Image.open(f'/p/tmp/bochow/LAMA/lama/inference/sic/fixed_256/sic{i:06}_crop000_mask000.png')  

        img_gt = unnormalize_pic(np.array(img_gt)[:,:,0])
        img_inpainted = unnormalize_pic(np.array(img_inpainted)[:,:,0])
        
        lon = np.arange(90,45,-0.25)
        print(lon.shape)
        lat = np.linspace(0,360,256)
        lons, lats = np.meshgrid(lon, lat)

        print(img_gt[:,:].shape, lons.shape, lats.shape)
        fig = plt.figure(figsize=(20, 5))
        ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())#(45, 90))
        ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())#(45, 90))
 

        img_extent = (0, 360, 45, 90)
        #ax0.set_global()
        ax0.set_extent([0, 360, 45, 90], ccrs.PlateCarree())
        ax0.coastlines()
        #ax1.set_global()
        ax1.set_extent([0, 360, 45, 90], ccrs.PlateCarree())
        ax1.coastlines()
        min_imshow = 0
        max_imshow = 100
        im1 = ax0.imshow(img_gt[:,:], cmap = "viridis", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "upper")
        ax1.imshow(img_inpainted[:,:], cmap = "viridis", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree(), extent=img_extent, origin= "upper")
        
        #im1 = ax0.contourf(lats, lons, img_gt[:,:].T, cmap = "viridis", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree())#, extent=img_extent, origin= "upper")
        #ax1.contourf(lats, lons, img_inpainted[:,:].T, cmap = "viridis", vmin = min_imshow, vmax = max_imshow, transform = ccrs.PlateCarree())#, extent=img_extent, origin= "upper")
        
        plt.savefig(f"/p/tmp/bochow/LAMA/lama/sic/eval_pics/sic{i:06}.png", dpi=600, bbox_inches = "tight")
        


def timeseries(): 
    indices = exclude_no_data_images()[:,0]
    print(indices.size)
    sic = np.zeros((indices.size))
    lat = np.arange(45,90,0.25)
    lon = np.linspace(-180,180,256)
    area = np.flip(area_grid(lat, lon), axis = 0)
    total_area = area.sum(['latitude','longitude'])
    print(total_area)   
    for k,name in enumerate(indices): 
        img_gt = Image.open(f'/p/tmp/bochow/LAMA/lama/sic/raw/sic{name:06}.png') 
        img_inpainted = Image.open(f'/p/tmp/bochow/LAMA/lama/inference/sic/fixed_256/sic{name:06}_crop000_mask000.png')  

        img_gt = unnormalize_pic(np.array(img_gt)[:,:,0])
        img_inpainted = unnormalize_pic(np.array(img_inpainted)[:,:,0])/100
        sic[k] = np.sum((area * img_inpainted)) /1e6
    plt.imshow(img_inpainted)
    plt.savefig("/p/tmp/bochow/LAMA/lama/sic/eval_pics/test2.png")
    plt.imshow(area)
    plt.savefig("/p/tmp/bochow/LAMA/lama/sic/eval_pics/test.png")
    fig, ax = plt.subplots(figsize=(10,10))
    
    ax.scatter(indices, sic)
    ax.set_xlim(30*12,112*12)
    ax.set_xticks(np.arange(30*12, 112*12, 5*12))
    ax.set_xticklabels(np.arange(1931,2013, 5))
    plt.savefig(f"/p/tmp/bochow/LAMA/lama/sic/eval_pics/timeseries.png", dpi=600, bbox_inches = "tight")
        
        
#timeseries()


def timeseries_concentration(): 
    indices = exclude_no_data_images()[:,0]
    print(indices.size)
    sic = np.zeros((indices.size))
    lat = np.arange(45,90,0.25)
    lon = np.linspace(-180,180,256)
    area = np.flip(area_grid(lat, lon), axis = 0)
    total_area = area.sum(['latitude','longitude'])
    
    sic_reanalysis = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/sic/G10010_SIBT1850_v1.1.nc")["seaice_conc"][:]
    sic_reanalysis[sic_reanalysis<0] = 0
    lat_reanalysis = np.flip(netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/sic/G10010_SIBT1850_v1.1.nc")["latitude"][:])
    lon_reanalysis = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/sic/G10010_SIBT1850_v1.1.nc")["longitude"][:] -180

    area_reanalysis = np.flip(area_grid(lat_reanalysis, lon_reanalysis), axis = 0)
    total_area_reanalysis  = area_reanalysis.sum(['latitude','longitude'])

    plt.imshow(area_reanalysis)
    plt.savefig(f"/p/tmp/bochow/LAMA/lama/sic/eval_pics/test_area.png", dpi=600, bbox_inches = "tight")
    plt.clf() 
    plt.imshow(sic_reanalysis[-12,:,:]) 
    plt.savefig(f"/p/tmp/bochow/LAMA/lama/sic/eval_pics/test_sic_re.png", dpi=600, bbox_inches = "tight")
    plt.clf() 
    for k,name in enumerate(indices): 
        img_gt = Image.open(f'/p/tmp/bochow/LAMA/lama/sic/raw/sic{name:06}.png') 
        img_inpainted = Image.open(f'/p/tmp/bochow/LAMA/lama/inference/sic/fixed_256/sic{name:06}_crop000_mask000.png')  

        img_gt = unnormalize_pic(np.array(img_gt)[:,:,0])
        img_inpainted = unnormalize_pic(np.array(img_inpainted)[:,:,0])/100
        #print(np.sum(img_inpainted))
        #img_inpainted[img_inpainted<0.15] = 0
        #print(np.sum(img_inpainted))
        sic[k] = np.sum((area * img_inpainted)) /1e6
    
    sia_reanalysis_mean = np.zeros(sic_reanalysis.shape[0])
    #sic_reanalysis[sic_reanalysis<15] = 0
    img_inpainted = Image.open(f'/p/tmp/bochow/LAMA/lama/inference/sic/fixed_256/sic{indices[-1]:06}_crop000_mask000.png')  
    plt.imshow(img_inpainted) 
    plt.savefig(f"/p/tmp/bochow/LAMA/lama/sic/eval_pics/test_sic_last.png", dpi=600, bbox_inches = "tight")
    for i in range(sic_reanalysis.shape[0]):
        sia_reanalysis_mean[i] = np.sum((area_reanalysis* sic_reanalysis[i,:,:]/100)) /1e6

    fig, ax = plt.subplots(figsize=(10,10))
    print(sia_reanalysis_mean[-12], sic[-1])
    ax.scatter(indices+50*12, sic)
    ax.scatter(np.arange(sia_reanalysis_mean.size), sia_reanalysis_mean)
    #ax.set_xlim(30*12,112*12)
    ax.set_xticks(np.arange(30*12, 112*12, 5*12))
    ax.set_xticklabels(np.arange(1931,2013, 5))
    plt.savefig(f"/p/tmp/bochow/LAMA/lama/sic/eval_pics/timeseries_comparison.png", dpi=600, bbox_inches = "tight")
        
timeseries_concentration()

#plot_sea_ice_inference()