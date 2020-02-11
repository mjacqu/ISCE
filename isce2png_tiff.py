import os
import glob
import re

def make_png_kml(type):
    """ 
    Turn ISCE coherence or amplitude/phase images into png (with associated kml) or geotiff. 

    Parameter:
    type (str): tiff, coherence or amp_phase

    Returns:
    if type is amplitude_tiff: 2 channel tiff with amplitude and phase
    if type is coherence or amp_phase: png + kml

    """
	all_directories = os.walk('.').next()[1]# list of all current directories
	regex = re.compile(r'\d{8}_\d{8}')
	directories = list(filter(regex.search, all_directories))
	for i in range(0,len(directories)):
            if type == "tiff":
                print(type)
                filename = os.path.join(directories[i],"merged/filt_topophase.unw.geo.vrt")
                command = "gdal_translate " + filename + " " +os.path.join(directories[i],directories[i]+'.amp.geo.tif')
                #print(command)
                os.system(command)
            else:
                if type == "coherence":
                    print(type)
                    filename = os.path.join(directories[i], "merged/phsig.cor.geo")
                    rename = "phsig.cor.geo.png"
                    #print(filename, rename)
                if type == "amp_phase":
                    print(type)
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
                os.rename(rename, os.path.join(directories[i],directories[i]+"_"+ type +".png")) # rename file
                os.rename(directories[i]+".kml", os.path.join(directories[i],directories[i]+"_"+ type + ".kml")) # move xml to directory




#make_png_kml("coherence")
make_png_kml("tiff")


