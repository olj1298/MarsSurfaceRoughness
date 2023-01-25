# %%
import timeit
try:
    import gdal
except:
    from osgeo import gdal
x_block_size = 5
y_block_size = 5
# Function to read the raster as arrays for the chosen block size.
def read_raster(x_block_size, y_block_size):
    raster = "C:/Users/orang/Downloads/499/DTMName/LavaChannelandCataractinSouth-CentralElysiumPlanitia/DTEEC_018339_1820_018194_1820_A01.img" #file location
    ds = gdal.Open(raster) #open command
    band = ds.GetRasterBand(1) #file has one band
    xsize = band.XSize #number of pixels wide
    ysize = band.YSize #number of pixels tall
    blocks = 0 #begin counter
    for y in range(0, ysize, y_block_size): #start 0, stop at ysize, step in blocks
        if y + y_block_size < ysize: #when pixel location and block size before last column
            rows = y_block_size #rows is size of block
        else:
            rows = ysize - y #if pixel location and block size greater than or equal to the last column
        for x in range(0, xsize, x_block_size):
            if x + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x
            array = band.ReadAsArray(x, y, cols, rows)
            #del array
            blocks += 1 #add one to counter
    band = None
    ds = None
    print ("{0} blocks size {1} x {2}:".format(blocks, x_block_size, y_block_size))

# Function to run the test and print the time taken to complete.
def timer(x_block_size, y_block_size):
    t = timeit.Timer("read_raster({0}, {1})".format(x_block_size, y_block_size),
                     setup="from __main__ import read_raster")
    print ("\t{:.2f}s\n".format(t.timeit(1)))

raster = "C:/Users/orang/Downloads/499/DTM Name/DTEEC_064444_2210_029236_2210_A01.img"
ds = gdal.Open(raster)
band = ds.GetRasterBand(1)

# Get "natural" block size, and total raster XY size. 
block_sizes = band.GetBlockSize()
x_block_size = block_sizes[0]
y_block_size = block_sizes[1]
xsize = band.XSize
ysize = band.YSize
band = None
ds = None

# Tests with different block sizes.
timer(256,256)#x_block_size, y_block_size)
#timer(x_block_size*10, y_block_size*10)
#timer(x_block_size*100, y_block_size*100)
#timer(x_block_size*10, y_block_size)
#timer(x_block_size*100, y_block_size)
#timer(x_block_size, y_block_size*10)
#timer(x_block_size, y_block_size*100)
#timer(xsize, y_block_size)
#timer(x_block_size, ysize)
#timer(xsize, 1)
#timer(1, ysize)
# %%
import gdal
from gdalconst import *
from numpy import *
from gdalconst import *
import sys
import numpy as np
print(sys.path)

fn = "C:/Users/orang/Downloads/499/DTM Name/DTEEC_064444_2210_029236_2210_A01.img"
ds = gdal.Open(fn)
if ds is None:
    print (f'Could not open {fn}')
    sys.exit(1)
band = ds.GetRasterBand(1)
band.DataType #if 1, 8-bit inconsistent integers
cols = ds.RasterXSize
rows = ds.RasterYSize
#bands = ds.RasterCount
geotransform = ds.GetGeoTransform()
#originX = geotransform[0]
#originY = geotransform[3]
#pixelWidth = geotransform[1]
#pixelHeight = geotransform[5]
#xOffset = int((1000 - originX) / pixelWidth)
#yOffset = int((1000- originY) / pixelHeight)
#band = ds.GetRasterBand(1)
data = band.ReadAsArray(0,0,cols,rows)
print(data)
datashift = np.roll(data,2)
#print(datashift)
#print(data[1000,1000])
slope = data-datashift
#print(slope)
#value = data[0,0]
#data = band.ReadAsArray(0, 0, cols, rows)
#value = data[42, 94]
band = None
dataset = None
ds=None
dataset = gdal.Open("C:/temp/slope_gdal.tif")
dataset.ReadAsArray(12101,6475,3,3)
dataset=None
band=None
dataset = gdal.Open("C:/temp/slope_gdal.tif")
band = dataset.GetRasterBand(1)
band.ReadAsArray(100,100,5,5,10,10)
dataset=None
band=None

# %%
import utils
blockSize = utils.GetBlockSize(band)
xBlockSize = blockSize[0]
yBlockSize = blockSize[1]
for i in range(rows):
    data = band.ReadAsArray(0, i, cols, 1)
# do something with the data here, before
# reading the next row
bSize = 5   
for i in range(0, rows, bSize):
    if i + bSize < rows:
        size = bSize
    else:
        size = rows - i
    data = band.ReadAsArray(0, i, cols, size)
    # do something with the data here, before
    # reading the next set of blocks
rows = 13
cols = 11
xBSize = 5
yBSize = 5
for i in range(0, rows, yBSize):
    if i + yBSize < rows:
        numRows = yBSize
    else:
        numRows = rows - i
    for j in range(0, cols, xBSize):
        if j + xBSize < cols:
            numCols = xBSize
        else:
            numCols = cols - j
            data = band.ReadAsArray(j, i, numCols, numRows)

# %%
