import numpy as np 
import h5py
from matplotlib import pyplot as plt
from PIL import Image
import netCDF4
import pandas as pd 


lat1d = np.arange(-90, 90, 2.5)
lon1d = np.arange(-180,180,5)
lon2d, lat2d = np.meshgrid(lon1d, lat1d)

img_file = netCDF4.Dataset("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.4.6.0.0.anomalies.1.nc")["temperature_anomaly"][:,:,:]
length = img_file.shape[0]
img_file[img_file<-1000]=np.nan
min_value = np.nanmin(img_file)
#min_value = -33.276234
#max_value = 56.640396
max_value = np.nanmax(img_file-min_value)
hadcrut4_monthly = pd.read_csv("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut4_monthly.csv", header = None, delimiter = "   ")
temp_anomaly_hadcrut4 = np.array(hadcrut4_monthly.iloc[:,1])
print(temp_anomaly_hadcrut4.shape,temp_anomaly_hadcrut4)
hadcrut5_monthly =pd.read_csv("/p/tmp/bochow/LAMA/lama/hadcrut/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv", delimiter = ",")
temp_anomaly_hadcrut5 = np.array(hadcrut5_monthly.iloc[:,1])[0:2064]


hadcrut4_monthly_kringing = pd.read_csv("/p/tmp/bochow/LAMA/lama/hadcrut/hadcrut4_monthly_kringing.csv", header = None, delimiter = "  ")
temp_anomaly_hadcrut4 = np.array(hadcrut4_monthly_kringing.iloc[:,1])[0:2052]

def unnormalize(array): 
    array = array/255*max_value + min_value
    return array

weights = np.cos(np.deg2rad(lat2d))
spatial_average = np.zeros(2064)

for i in range(2064): 
    img_inpainted = unnormalize(np.array(Image.open('/p/tmp/bochow/LAMA/lama/inference/hadcrut/fixed_72_10days/hadcrut' + f'{i:06}' + '_crop000_mask000.png')))
    array2d = np.array(img_inpainted)[:,:, 0]
    #print(i)
    spatial_average[i] = np.average(array2d, weights=weights)

yearly_average = np.mean(spatial_average.reshape(-1, 12), axis=1)
yearly_average_hadcrut4 = np.mean(temp_anomaly_hadcrut4.reshape(-1, 12), axis=1)
yearly_average_hadcrut5 = np.mean(temp_anomaly_hadcrut5.reshape(-1, 12), axis=1)
yearly_average_hadcrut4_kringing = np.mean(temp_anomaly_hadcrut4.reshape(-1, 12), axis=1)



fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlabel("Year after 1850")
ax.set_ylabel("Temperature anomaly (1960-1990) [K]")
ax.plot(yearly_average, label = "inpainted")
ax.plot(yearly_average_hadcrut4, label = "hadcrut4")
ax.plot(yearly_average_hadcrut5, label = "hadcrut5")
ax.axhline(np.mean(yearly_average))
ax.axhline(np.mean(yearly_average_hadcrut4), color = "C1")
ax.axhline(np.mean(yearly_average_hadcrut5), color = "C2")
#ax.plot(yearly_average_hadcrut4_kringing, label = "hadcrut4 kringing")
ax.legend()
plt.savefig("time_series_inpainted_yearly.pdf")

