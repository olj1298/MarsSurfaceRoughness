# MarsSurfaceRoughness
Calculation of surface roughness from Mars HiRISE DTM elevation data
The repository is split into the main Python file to access all functions and run code, and a folder of the different outputs from code while building project.
Digital Terrain Models(DTM) are freely available to all at https://www.uahirise.org/dtm/

Features
------
This software can import DTMs and read the header details. It can also convert raster band to a numpy array with non-data values being converted to NANs. Slope calculation is done in both Zevenbergen-Thorne(ZT) and Horn formulas that are converted from radians to degrees and plotted within a limit from zero to forty-five degrees. An option to save .tiff files is available as well. The differnential ZT slope bewtween baselines is computed by converting the array of baselines and normalizing their slopes.  

User will need to change folder variable depending on where their DTM files are saved on their system. Running the getDTMinfo is mostly to verify that the folder variable is correct and to get the pixel size of the DTM. The variable fn will need to be changed as well to the user's system for file access and plot saving. Pixel size changes based on the DTM used. User will need to copy and paste pixel size. In the plotdiff function, user will need to change the plot title if scales for differential slopes is not what is currently saved. Also, the scalesize variable can be edited based on what limits on the data the user would like to use.

After the Horn Slope function, the last part of the python file are functions to graph historgrams, curvature, and elevation data that was used to test project accuracy.

Installation
------------
Project was run in Windows 10 using Anaconda Navigator and Visual Studio Code. Any user needs to download a gdal python instance. Look at howtogetgdalinstance.txt to see how to replicate this. Will need to import sys, sys.path,osgeo, numpy, matplotlib.pyplot, scipy, PIL, time, and  datetime.

Contribute
----------

- Issue Tracker: https://github.com/olj1298/MarsSurfaceRoughness/issues
- Source Code: https://github.com/olj1298/MarsSurfaceRoughness
