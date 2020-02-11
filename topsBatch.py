import glob, os
from writexml import makexml
import re

# 1. topsBatch.py takes .zip files in current directory and makes a director for each
# pair of images YYYYMMDD_YYYYMMDD, writes xml file for the corresponding pair
# and then runs topsApp.py in each directory.
# To avoid time-out, lauch tmux session before running topsBatch.py
# 2. TopsBatch requries writexml.py to be in same directory. Check writexml for dem and bounding box changes... (not in function yet)

datapath = "/mnt/MyleneShare/Chile/data/asc/" # path to data files
orbpath = "/net/tiampostorage/volume1/LabShare/orbits/aux_poeorb" # path to orbit files
auxpath = "/home/myja3483/Landslides/Bigsur/auxfiles" # path to auxfiles
swath_sel = "[3]" #select swath. Input type list i.e. [3]
az_look = "3" # number of azimuth looks
rng_look = "7" # number of range looks
n=3 #nth image to use for pair


files = glob.glob(datapath+"*.zip") # list of all filenames in data directory
r = re.compile('1S.V_([0-9]+?)T') # build sorting criterion
def key_func(m):
    return int(r.search(m).group(1))

files = sorted(files, key=key_func)
filenames = list(files)
for i in range(0,len(files)): #loop through and create list of dates only:
     m = re.search('1S.V_([0-9]+?)T', files[i]) # extract regex corresponding to date
     if m:
        files[i] = m.group(1)
'''
# for every one file (every file gets paired with the next file)
for i in range(0,len(filenames)-n):
    dir_name = files[i]+'_'+files[i+n] # construct directory name
    os.mkdir(dir_name)
    masterfile = os.path.join(os.getcwd(),filenames[i]) # master filename as string
    slavefile = os.path.join(os.getcwd(),filenames[i+n]) # slave filename as string
    makexml(orbpath,auxpath,masterfile,slavefile,swath_sel,az_look,rng_look)
    os.rename("topsApp.xml", os.path.join(dir_name,"topsApp.xml")) # move xml to directory

'''
# for every file with every file
for i in range (0,len(filenames)):
    for j in range (i+1,len(filenames)):
        dir_name=files[i]+'_'+files[j] # construct directory name
        os.mkdir(dir_name)
        masterfile = os.path.join(os.getcwd(),filenames[i]) # master filename as string
        slavefile = os.path.join(os.getcwd(),filenames[j]) # move xml to directory
        makexml(orbpath,auxpath,masterfile,slavefile,swath_sel,az_look,rng_look)
        os.rename("topsApp.xml", os.path.join(dir_name,"topsApp.xml")) # move xml to directory

'''

#2. go into each directory and run topsApp.py --steps
directories = os.walk('.').next()[1] # list of all current directories

for i in range(0,len(directories)): # loop over directories
     os.chdir(directories[i])    # and go into each one by one
     os.system("topsApp.py --steps")
     os.chdir("..")
'''
