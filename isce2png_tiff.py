import os
import glob
import re

def make_png_kml(data, type):
    """ 
    Turn ISCE coherence or amplitude/phase images into png (with associated kml) or geotiff. 

    Parameter:
    data (str): amp_phase or coherence
    type (str): tiff or png


    Returns:
    if type is tiff: 2 channel tiff with amplitude and phase or 1 channel tiff with coherence
    if type is png: png with coherence or amplitude filtered phase plus corresponding kml 

    """
    all_directories = os.walk('.').next()[1]# list of all current directories
    regex = re.compile(r'\d{8}_\d{8}')
    directories = list(filter(regex.search, all_directories))
    for i in range(0,len(directories)):
            if type == "tiff":
                print(type)
                if data == "amp_phase":
                    f = "merged/filt_topophase.unw.geo.vrt"
                if data == "coherence":
                    f = "merged/phsig.cor.geo.vrt"
                print(data)
                filename = os.path.join(directories[i],f)
                command = "gdal_translate " + filename + " " + os.path.join(directories[i],directories[i] + data + '.tif')
                #print(command)
                os.system(command)
            else:
                if data == "coherence":
                    print(data)
                    filename = os.path.join(directories[i], "merged/phsig.cor.geo")
                    rename = "phsig.cor.geo.png"
                    #print(filename, rename)
                if data == "amp_phase":
                    print(data)
                    filename =os.path.join(directories[i], "merged/filt_topophase.unw.geo")
                    rename = "filt_topophase.unw.geo.png"
                command = "mdx.py " + filename + " -kml " + directories[i] + ".kml"
                os.system(command)
                with open(directories[i]+".kml", 'r') as file :
                    filedata = file.read()

                path = os.path.join(os.getcwd(),rename)
                    # Replace the target string
                filedata = filedata.replace(path, directories[i]+".png")
                    # Write the file out again
                with open(directories[i]+".kml", 'w') as file:
                    file.write(filedata)
                os.rename(rename, os.path.join(directories[i],directories[i]+"_"+ data +".png")) # rename file
                os.rename(directories[i]+".kml", os.path.join(directories[i],directories[i]+"_"+ data + ".kml")) # move xml to directory




#make_png_kml("coherence")
make_png_kml("coherence","tiff")


