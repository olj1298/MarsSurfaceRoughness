# %%
import sys
sys.path
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import numpy
import math
numpy.set_printoptions(threshold=sys.maxsize)
from matplotlib import interactive #visualizing plots
interactive(True) #visualizing plots

"""Load and read DTM data from header, take slope calculation, add information to header"""

#address for DTM file
fn = "C:/Users/orang/Downloads/499/DTMName/LavaChannelandCataractinSouth-CentralElysiumPlanitia/DTEEC_018339_1820_018194_1820_A01.img"
scale = 1 #distance to roll arrays for slope calculation
pixelsize=1.011852850773099943 #pixel resolution from DTM header
theta = 0 #global variable used later
ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open

#file details
#print("Band count:", ds.RasterCount)  # number of bands
#print("Projection: ", ds.GetProjection())  # get projection
#print("Columns:", ds.RasterXSize)  # number of columns
#print("Rows:", ds.RasterYSize)  # number of rows
#print("GeoTransform", ds.GetGeoTransform()) #tuple with 6 elements(below)
#[0] left X coordinate, [1] pixel width, [2] row rotation (usually zero)
#[3] top Y coordinate, [4] column rotation (usually zero), [5] pixel height, this will be negative for north up images
band = ds.GetRasterBand(1) #select raster with elevation values
ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
#print('No data value:', ndv)
#A = [[11,12,5,2],[15,6,10,4],[10,8,12,5],[12,15,8,6]]

#def simslope(data_nan,scale,pixelsize,theta):
subtractiny = np.roll(data_nan,-1*scale,axis=0) - np.roll(data_nan, scale,axis=0)#shift array up one minus shift array down one
subtractinx = np.roll(data_nan,-1*scale,axis=1) - np.roll(data_nan, scale,axis=1)#shift array right one minus shift array left one
yxinc = 2*pixelsize #baseline for calculation 
dzdy = subtractiny/yxinc #take difference over y baseline
dzdx = subtractinx/yxinc #take difference over x baseline
dysquared= dzdy**2 #square  y calc
dxsquared= dzdx**2 #square x calc
yplusx = dysquared + dxsquared #sum squares
sqyplusx = np.sqrt(yplusx) #square root squares
theta = np.arctan(sqyplusx) #slope in radians
slopedeg = theta * 180/np.pi
#return slopedeg

plt.figure(figsize=(58,60)) #columns=5953 row=5707, figsize=(58,60)
plt.imshow(slopedeg, cmap='RdBu', vmin=np.nanmin(slopedeg),vmax=np.nanmax(slopedeg))
plt.colorbar(ticks=np.linspace(np.nanmin(slopedeg),np.nanmax(slopedeg),10))

#aspect
#phi = math.atan2(-1*dzdy, -1*dzdx)
#aspect = 90 - phi

#plt.figure(figsize=(58,60))
#plt.imshow(aspect, cmap='plasma', vmin=np.nan(aspect), vmax=np.nan(aspect))
#plt.colorbar()

#driver_gtiff = gdal.GetDriverByName('GTiff')
#fn_copy = fn
#ds_copy = driver_gtiff.CreateCopy(fn_copy, ds)
#data_array = ds.GetRasterBand(1).ReadAsArray()  # read band data from the existing raster
#data_nan = np.where(data_array == ndv, np.nan, data_array)  # set all the no data values to np.nan so we can easily calculate the minimum elevation
#data_min = np.min(data_nan) # get the minimum elevation value (excluding nan)
#data_stretch = np.where(data_array == ndv, ndv, (data_array - data_min) * 1.5)  # now apply the strech algorithm
#ds_copy.GetRasterBand(1).WriteArray(slopedeg)  # write the calculated values to the raster
#ds_copy = gdal.Open(fn_copy)
#data_slope = ds_copy.GetRasterBand(1).ReadAsArray()
#plt.figure(figsize=(58, 60))
#plt.imshow(slopedeg, cmap='RdBu', vmin=np.nanmin(slopedeg),vmax=np.nanmax(slopedeg))
#plt.colorbar(ticks=np.linspace(np.nanmin(slopedeg),np.nanmax(slopedeg),10))

ds=None #closeraster



# %%
import numpy as np
scale = 1 #distance to roll arrays for slope calculation
pixelsize=1.011852850773099943 #pixel resolution from DTM header
data_nan = [[11,12,5,2], 
            [15,6,10,4], 
            [10,8,12,5], 
            [12,15,8,6]]
#shift array up one plus orginal array minus shift array down one plus original array
subtractiny = (np.roll(data_nan,-1*scale,axis=0) + data_nan) - (np.roll(data_nan,scale,axis=0) + data_nan)
#shift array right one minus shift array left one
subtractinx = (np.roll(data_nan,-1*scale,axis=1) + data_nan) - (np.roll(data_nan, scale,axis=1) + data_nan)
print(subtractiny)
print(subtractinx)


# %%
