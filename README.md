# MarsSurfaceRoughness
Calculation of surface roughness from Mars HiRISE DTM elevation data

Project was run in Windows 10 using Anaconda Navigator and Visual Studio Code. Any user needs to download a gdal python instance. Look at howtogetgdalinstance.txt to see how to replicate this.

The repository is split into the main Python file to access all functions and run code, and a folder of the different outputs from code while building project.

#Digital Terrain Models(DTM) are freely available to all at https://www.uahirise.org/dtm/

User will need to change folder variable depending on where their DTM files are saved on their system. Running the getDTMinfo is mostly to verify that the folder variable is correct and to get the pixel size of the DTM. The variable fn will need to be changed as well to the user's system for file access and plot saving. Pixel size changes based on the DTM used. User will need to copy and paste pixel size. In the plotdiff function, user will need to change the plot title if scales for differential slopes is not what is currently saved. Also, the scalesize variable can be edited based on what limits on the data the user would like to use.

After the Horn Slope function, the last part of the python file are functions to graph historgrams and elevation data that was used to test project accuracy.
