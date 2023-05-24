#written by olivia jones at U of Arizona for HiRISE research project 05.12.2023
#advised by Dr. Shane Byrne and Dr. Sarah Sutton
import sys
sys.path
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import datetime

date_today = datetime.datetime.now() #adding the computer date to the filename when function runs
datestr = date_today.strftime('%m%d%Y') #format date

"""read DTM data from header, save file address. User needs to edit folder,fn and later pixsize variables depending on the DTM they are using. Take ZT and Horn slopes."""

####change this value depending on which DTM you are using
folder = '/UpliftedBlocksofLightTonedLayeredDeposits/DTEEC_054753_1825_055452_1825_A01.img' #specific DTM address

def getDTMinfo(folder):
    """Take directory address from user, apply to parent directory address, call open function for gdal, call header information to return pixelsize for taking slope.
        Inputs:
            :folder(string): directory address for specific DTM.img file from HiRISE website
        Returns:
            ::entire DTM header for user to copy paste pixel size"""
    
    ####change based on DTM directory address on user computer
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder #address for DTM file related to parent directory
    
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    print(gdal.Info(ds)) #print full DTM header
    #file details #print("Band count:", ds.RasterCount)  #number of bands #print("Projection: ", ds.GetProjection())  #get projection
    #print("Columns:", ds.RasterXSize)  #number of columns #print("Rows:", ds.RasterYSize)  #number of rows
    #print("GeoTransform", ds.GetGeoTransform()) #tuple with 6 elements(below) #[0] left X coordinate, [1] pixel width, [2] row rotation (usually zero)
    #[3] top Y coordinate, [4] column rotation (usually zero), [5] pixel height, this will be negative for north up images
    ds = None #close item
    with open(fn, 'r') as infile:
        for i in range(10):
            line = infile.readline()
            line = line.rstrip("\n")
            print(line)
            
    return

getDTMinfo(folder)

####change this value for separate DTMs
pixsize = 1.011848494027900 #pixel resolution from DTM header

#run data_nan for code below
fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
band = ds.GetRasterBand(1) #select raster with elevation values
ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN

def ZTslope(folder,data_nan,scale,pixsize):
    """Calculate Slope using Zevenbergen-Thorne formula from Dr. Shane Byrne's Lectures.
                f-d         b-h, 
      arctan( ------^2  + ------^2 ) ^1/2
               dz/dy       dzdx
        Inputs:
            :folder(string): directory address for specific DTM.img file from HiRISE website
            :data_nan(array): array of DTM with nan values
            :scale(float): step size for rolling arrays for slope calculation
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::slope of elevation points in degrees"""
    
    #access folder location, assign non data values, create base array of elevation data
    ####change based on directory address
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
    
    #manipulate elevation data with Zevenbergen Thorne method
    f = np.roll(data_nan,-1*scale,axis=0) #shift array up one
    d = np.roll(data_nan, scale,axis=0) #shift array down one
    b = np.roll(data_nan,-1*scale,axis=1) #shift array right one
    h = np.roll(data_nan, scale,axis=1) #shift array left one
    subtractiny = f - d #shift array up one minus shift array down one to take difference
    subtractinx = b - h #shift array right one minus shift array left one to take difference
    yxinc = 2*scale*pixsize #baseline for calculation 
    dzdy = subtractiny/yxinc #take difference over y baseline
    dzdx = subtractinx/yxinc #take difference over x baseline
    dysquared = dzdy**2 #square  y calc
    dxsquared = dzdx**2 #square x calc
    yplusx = dysquared + dxsquared #sum squares
    sqyplusx = np.sqrt(yplusx) #square root squares
    theta = np.arctan(sqyplusx) #slope in radians
    slopedeg = theta * 180/np.pi
    
    #plotting
    pltx = int(str(ds.RasterXSize)[:-2])+1 #first two integers in x size of raster plus one to set figure size
    plty = int(str(ds.RasterYSize)[:-2])+1 #first two intergers in y size of raster plus one to get figure size
    plt.figure(figsize=(pltx,plty)) #rows,columns
    label_size = 30
    plt.title(f"Zevenbergen-Thorne Slope Calculation Baseline of {2*scale}m",fontsize=label_size)
    plt.xlabel('distance (x) m', fontsize=label_size) #adjust x axis label
    plt.ylabel('distance (y) m', fontsize=label_size) #adjust y axis label 
    plt.rcParams['xtick.labelsize'] = label_size  #adjust x tick size
    plt.rcParams['ytick.labelsize'] = label_size #adjust y tick size
    plt.tick_params(label_size)
    plt.rcParams.update({'font.size': label_size}) #adjust x label
    #plt.imshow(slopedeg, cmap='magma', vmin=0,vmax=45)#could also use vmin=np.nanmin(slopedeg),vmax = np.nanmax(slopedeg)
    plt.colorbar(ticks=np.linspace(0,45,5)) #cutoff at 45 degree so all outputs at different baselines for one DTM is the same
    plt.savefig(fn + f'/ZTslope_{datestr}.png')
    
    #save tiff file
    tiffimg = Image.fromarray(slopedeg)
    tiffimg.save(fn + f'/slopedeg_{datestr}.tif')
    
    ds = None #close raster
    
    return slopedeg

ZTslope(folder,data_nan,4,pixsize) #run basic ZT slope 

def plotdiff(folder,scalearray,pixsize):
    """Calculate Differential Slope using formula from Krevalsky et al. 2001. z is elevation point. L is baseline (scale input).
            (z_L/2 - z_-L/2) - (z_L - z_-L)
            ----------------   ------------
                    L               2L
        Inputs:
            :folder(string): directory address for specific DTM.img file from HiRISE website
            :scalearray(array): numpy array of integers to take difference between slope baselines
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::png image of differential slope saved in folder designated by user"""
            
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN

    diff = ZTslope(folder,data_nan,2,pixsize) - ZTslope(folder,data_nan,4,pixsize) #4m baseline minus 8m baseline arrays
    diff2 = ZTslope(folder,data_nan,4,pixsize) - ZTslope(folder,data_nan,8,pixsize) #8m baseline minus 16m baseline arrays
    diff3 = ZTslope(folder,data_nan,8,pixsize) - ZTslope(folder,data_nan,16,pixsize) #16m baseline minus 32m baseline arrays

    #normalize data to get rid of any artifacts of extreme highs and lows of data
    normdiff = (diff-np.nanmean(diff))/np.nanstd(diff)
    normdiff2 = (diff2-np.nanmean(diff2))/np.nanstd(diff2)
    normdiff3 = (diff3-np.nanmean(diff3))/np.nanstd(diff3)

    #reduce arrays to ones and zeros, limit large outliers
    ####change based on scalesize to limit
    scalesize = 1
    
    data = (normdiff - (-1*scalesize))/(scalesize -(-1*scalesize) + 1e-9)
    data2 = (normdiff2 - (-1*scalesize))/(scalesize -(-1*scalesize) + 1e-9)
    data3 = (normdiff3 - (-1*scalesize))/(scalesize -(-1*scalesize) + 1e-9)
    result = (255*np.stack((data,data2,data3))).clip(0,255).astype(np.uint8) #stack 3 arrays on top of each other, multiplt by 255 so values range only from 0 to 255 for RGB conversion
    im = Image.fromarray(np.transpose(result,axes=(1,2,0))) #create image from stacked arrays and transpose

    #plotting
    pltx = int(str(ds.RasterXSize)[:-2])+1 # first two integers in x size of raster plus one to set figure size
    plty = int(str(ds.RasterYSize)[:-2])+1 # first two intergers in y size of raster plus one to get figure size
    plt.figure(figsize=(pltx,plty)) #columns=5953 row=5707, figsize=(58,60)
    label_size = 30
    
    ####change based on differential slopes chosen
    plt.title(f"Differential Zevenbergen-Thorne slope calculation Baseline of R=4/8m,G=8/16m,B=16/32m",fontsize=label_size) 
    
    plt.xlabel('distance (x) m', fontsize=label_size) #adjust x axis label font size
    plt.ylabel('distance (y) m', fontsize=label_size) #adjust y axis label font size
    plt.rcParams['xtick.labelsize'] = label_size #adjust x tick label font size
    plt.rcParams['ytick.labelsize'] = label_size #adjust y tick label font size
    plt.tick_params(label_size)
    plt.rcParams.update({'font.size': label_size})
    plt.imshow(im,vmin=np.nanmin(im),vmax=np.nanmax(im))
    im.savefig(fn + f'/ZTDifferential_{datestr}.png') #save file to view
    return im

scalearray = np.array([2,4],[4,8],[8,16])
def teststring(scalearray):
    
    print(scalearray[0])
    
plotdiff(folder,scalearray,pixsize) #run plot differential for different slopes

def Hornslope(folder,data_nan,scale,pixsize):
    """Calculate Slope using Horn formula from Dr. Shane Byrne Lectures.
            (c + 2*f + i) - (a + 2*d + g)   (a + 2*b + c) - (g + 2*h + i)
    arctan(------------------------------ + ------------------------------)^1/2
                 8*scale*pixelsize             8*scale*pixelsize
        Inputs:
            :folder(string): directory address for DTM file
            :data_nan(array): array of DTM with nan values
            :scale(float): step size for rolling arrays for slope calculation
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::Horn slope of elevation points in degrees"""
    
    #access folder location, assign non data values, create base array of elevation data
    ####change based on directory address for DTM on user computer
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
    
    #manipulate arrays
    a = np.roll(np.roll(data_nan,scale,axis=0),scale,axis=1) #shift array down one, right one
    b = np.roll(data_nan,scale,axis=0) #shift array down one
    c = np.roll(np.roll(data_nan,scale,axis=0),-1*scale,axis=1) #shift array down one, left one
    d = np.roll(data_nan,scale,axis=1) #shift array right one
    f = np.roll(data_nan,-1*scale,axis=1) #shift array left one
    g = np.roll(np.roll(data_nan,-1*scale,axis=0),scale,axis=1) #shift array up one, right one
    h = np.roll(data_nan,-1*scale,axis=0) #shift array up one
    i = np.roll(np.roll(data_nan,-1*scale,axis=0),-1*scale,axis=1) #shift array up one, left one

    #slope calculation
    dzdx = (c + 2*f + i) - (a + 2*d + g) #dzdx in Horn calculation
    dzdy = (a + 2*b + c) - (g + 2*h + i) #dzdy in Horn calculation
    xinc = 8*scale*pixsize #define baseline from pixelsize in DTM header    
    diffx = dzdx/xinc #take difference over baseline
    diffy = dzdy/xinc #take difference over baseline
    dxsquared = diffx**2 #square
    dysquared = diffy**2 #square
    yplusx = dxsquared+dysquared #sum squares
    sqyplusx = np.sqrt(yplusx) #square root squares
    theta = np.arctan(sqyplusx) #slope in radians
    slopedeg = theta * 180/np.pi #slope in degrees
    
    #plotting
    pltx = int(str(ds.RasterXSize)[:-2])+1 # first two integers in x size of raster plus one to set figure size
    plty = int(str(ds.RasterYSize)[:-2])+1 # first two intergers in y size of raster plus one to get figure size
    plt.figure(figsize=(pltx,plty)) #row,columns
    label_size = 30
    plt.title(f"Horn Slope Calculation Baseline of {2*scale}m",fontsize=label_size)
    plt.xlabel('slope (x) degrees', fontsize=label_size)
    plt.ylabel('slope (y) degrees', fontsize=label_size)
    plt.rcParams['xtick.labelsize'] = label_size 
    plt.rcParams['ytick.labelsize'] = label_size
    plt.tick_params(labelsize=label_size)
    plt.rcParams.update({'font.size': label_size})
    plt.imshow(slopedeg,cmap='magma',vmin=0,vmax=45) #can also use vmin=np.nanmin(slopedeg),vmax=np.nanmax(slopedeg)
    plt.colorbar(ticks=np.linspace(0,45,5))
    plt.savefig(fn + f'/Hornslope_{datestr}.png')
    
    tiffimg = Image.fromarray(slopedeg)
    tiffimg.savefig(fn + f'/Hornslopedeg_{datestr}.tif')
    
    ds = None #close raster
    
    return slopedeg

Hornslope(folder,data_nan,2,pixsize) #run Horn slope 

###################################
#Code below was used for information verification and was in tested by time the project was finished
###################################

def curvature(folder,data_nan,scale,pixsize):
    """Take curvature of ZT slope array.
        Inputs:
            :folder(string): directory address for DTM file
            :data_nan(array): array of DTM with nan values
            :scale(integer): step size for rolling arrays for slope calculation
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::Curvature of ZT slope for DTM"""
    
    #access folder location, assign non data values, create base array of elevation data
    ####change based on directory address for DTM on user computer
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
    
    #array manipulation
    f = np.roll(data_nan,-1*scale,axis=0) #shift array up one
    d = np.roll(data_nan, scale,axis=0) #shift array down one
    b = np.roll(data_nan,-1*scale,axis=1) #shift array right one
    h = np.roll(data_nan, scale,axis=1) #shift array left one
    subtractiny = f - d #shift array up one minus shift array down one
    subtractinx = b - h #shift array right one minus shift array left one
    yxinc = 2*scale*pixsize #baseline for calculation 
    dzdy = subtractiny/yxinc #take difference over y baseline
    dzdx = subtractinx/yxinc #take difference over x baseline
    dysquared = dzdy**2 #square  y calc
    dxsquared = dzdx**2 #square x calc
    yplusx = dysquared + dxsquared #sum squares
    sqyplusx = np.sqrt(yplusx) #square root squares
    theta = np.arctan(sqyplusx) #slope in radians
    slopedeg = theta * 180/np.pi
    nd = np.roll(slopedeg, scale,axis=1) #shift array left one
    e = slopedeg
    nf = np.roll(slopedeg, -1*scale, axis=1) #shift array right one
    nb = np.roll(slopedeg, -1*scale, axis=0) #shift array up one
    nh = np.roll(slopedeg, scale, axis=0) #shift array down one
    cx = nd - 2*e + nf / yxinc**2
    cy = nb - 2*e + nh / yxinc**2
    cyxsquared = cx**2 + cy**2
    sqcyx = np.sqrt(cyxsquared)
    
    #plotting
    pltx = ds.RasterXSize[2:]+1 #first two integers in x size of raster plus one to set figure size
    plty = ds.RasterYSize[2:]+1 #first two intergers in y size of raster plus one to get figure size
    plt.figure(figsize=(pltx,plty)) #rows,columns
    label_size = 30
    plt.title(f"Curvature calculation Baseline of {2*scale}m",fontsize=label_size)
    #Add axis labels
    plt.xlabel('distance (x) m', fontsize=label_size)
    plt.ylabel('distance (y) m', fontsize=label_size) #adjust tick label font size
    plt.rcParams['xtick.labelsize'] = label_size 
    plt.rcParams['ytick.labelsize'] = label_size
    plt.tick_params(label_size)
    plt.rcParams.update({'font.size': label_size})
    plt.imshow(sqcyx, cmap='magma', vmin=0,vmax=45) #vmin=np.nanmin(slopedeg) vmax = np.nanmax(slopedeg)
    plt.savefig(fn + f'/Curvature_{datestr}.png')
    
    return sqcyx

def getmode(folder,scale,pixsize):
    """Take mode of entire slope array to see if output for slope is realistic.
        Inputs:
            :folder(string): directory address for DTM file
            :scale(integer): step size for rolling arrays for slope calculation
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::mode of ZT slope array"""
            
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
    #manipulate elevation data with Zevenbergen Thorne method
    f = np.roll(data_nan,-1*scale,axis=0) #shift array up one
    d = np.roll(data_nan, scale,axis=0) #shift array down one
    b = np.roll(data_nan,-1*scale,axis=1) #shift array right one
    h = np.roll(data_nan, scale,axis=1) #shift array left one
    subtractiny = f - d #shift array up one minus shift array down one to take difference
    subtractinx = b - h #shift array right one minus shift array left one to take difference
    yxinc = 2*scale*pixsize #baseline for calculation 
    dzdy = subtractiny/yxinc #take difference over y baseline
    dzdx = subtractinx/yxinc #take difference over x baseline
    dysquared = dzdy**2 #square  y calc
    dxsquared = dzdx**2 #square x calc
    yplusx = dysquared + dxsquared #sum squares
    sqyplusx = np.sqrt(yplusx) #square root squares
    theta = np.arctan(sqyplusx) #slope in radians
    slopedeg = theta * 180/np.pi
    print(f"mode for {scale*2}m Z-T slope is:{stats.mode(np.round(slopedeg,1),axis=None,nan_policy='omit')}")

def takerow(folder,scale,pixsize):
    """Take single row of elevation and ZT slope data and graphs it.
        Inputs:
            :folder(string): directory address for DTM file
            :scale(integer): step size for rolling arrays for slope calculation
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::graph of elevation points"""
            
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
    #manipulate elevation data with Zevenbergen Thorne method
    f = np.roll(data_nan,-1*scale,axis=0) #shift array up one
    d = np.roll(data_nan, scale,axis=0) #shift array down one
    b = np.roll(data_nan,-1*scale,axis=1) #shift array right one
    h = np.roll(data_nan, scale,axis=1) #shift array left one
    subtractiny = f - d #shift array up one minus shift array down one to take difference
    subtractinx = b - h #shift array right one minus shift array left one to take difference
    yxinc = 2*scale*pixsize #baseline for calculation 
    dzdy = subtractiny/yxinc #take difference over y baseline
    dzdx = subtractinx/yxinc #take difference over x baseline
    dysquared = dzdy**2 #square  y calc
    dxsquared = dzdx**2 #square x calc
    yplusx = dysquared + dxsquared #sum squares
    sqyplusx = np.sqrt(yplusx) #square root squares
    theta = np.arctan(sqyplusx) #slope in radians
    slopedeg = theta * 180/np.pi

    zdatananrow = data_nan[101,:]
    datananrange = np.arange(0,5953,1)
    plt.plot(datananrange,zdatananrow, color='green',linewidth=2,label='datananslopecode')
    plt.xlabel('range')
    plt.ylabel('slope')
    plt.title("Elevation Profile original array slope calc")
    plt.legend()
    plt.savefig(fn + f'Data_nan_singlerow_{datestr}.png')

    ZTsloperow = slopedeg[101,:]
    ZTrange = np.arange(0,5953,1)
    plt.plot(ZTrange,ZTsloperow,color='red',linewidth=2,label='ZTslopecode')
    plt.xlabel('range')
    plt.ylabel('slope')
    plt.title("Elevation Profile ZT slope calc")
    plt.legend()
    plt.savefig(fn + f'ZT_singlerow_{datestr}.png')

def takehist(folder,scale,pixsize):
    """Get histogram of ZT slope to output distribution.
        Inputs:
            :folder(string): directory address for DTM file
            :scale(integer): step size for rolling arrays for slope calculation
            :pixsize(float): pixel resoultion taken from DTM header
        Returns:
            ::graph of ZT histogram"""
            
    fn = 'C:/Users/orang/Downloads/499/DTMName' + folder
    ds = gdal.Open(fn) #ds = None to close, gdal.Open(fn) to open
    band = ds.GetRasterBand(1) #select raster with elevation values
    ndv = data_array = ds.GetRasterBand(1).GetNoDataValue() #get non data value in DTM
    data_array = band.ReadAsArray(0,0,ds.RasterXSize,ds.RasterYSize) #numpy array, start at location 0,0 in array then go for all columns, rows
    data_nan = np.where(data_array == ndv, np.nan, data_array) #replace no data value with NaN
    #manipulate elevation data with Zevenbergen Thorne method
    f = np.roll(data_nan,-1*scale,axis=0) #shift array up one
    d = np.roll(data_nan, scale,axis=0) #shift array down one
    b = np.roll(data_nan,-1*scale,axis=1) #shift array right one
    h = np.roll(data_nan, scale,axis=1) #shift array left one
    subtractiny = f - d #shift array up one minus shift array down one to take difference
    subtractinx = b - h #shift array right one minus shift array left one to take difference
    yxinc = 2*scale*pixsize #baseline for calculation 
    dzdy = subtractiny/yxinc #take difference over y baseline
    dzdx = subtractinx/yxinc #take difference over x baseline
    dysquared = dzdy**2 #square  y calc
    dxsquared = dzdx**2 #square x calc
    yplusx = dysquared + dxsquared #sum squares
    sqyplusx = np.sqrt(yplusx) #square root squares
    theta = np.arctan(sqyplusx) #slope in radians
    slopedeg = theta * 180/np.pi
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(slopedeg[~np.isnan(slopedeg)], bins=int(np.nanmax(slopedeg)),edgecolor='black',color='magenta')
    ax.set(xlim=(0,68),ylim=(0,6500000),title=f'Distrib. of ZT slope values slope of {2*scale}m in DTM',
        xlabel='slope (deg)',ylabel='frequency')
    plt.savefig(fn + f'ZThist_{datestr}.png')
