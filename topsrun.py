import sys
sys.path.append('/home/myja3483/isce_tools/ISCE')
import tops
import numpy as np
import glob
import datetime
import os

'''
Example for usage of tops.Pair() and .run() to create topsApp xml files and run
isce with 'topsApp.py --steps'.
Use help(tops.Pair) and help(tops.makepairs) for details.
'''

data_path = '/net/tiampostorage/volume1/MyleneShare/data/bigsur_asc'
#dates = [
#    ('20151027', '20160119'),
#    ('20151027', '20160212')]
#maxdelta = datetime.timedelta(days = 48)
singlemaster = datetime.datetime(2015, 10, 27)

# set dict with all attributes of pair:
options = dict(
    path = '/net/tiampostorage/volume1/MyleneShare/Bigsur_asc/az1rng2',
    swaths = [3],
    orbit ='/net/tiampostorage/volume1/LabShare/orbits/aux_poeorb',
    auxiliary ='/home/myja3483/Landslides/Bigsur/auxfiles',
    unwrapper ='snaphu_mcf',
    dem = '/net/tiampostorage/volume1/MyleneShare/10mdem/crop.dem.wgs84',
    roi = [35.7622, 35.9909,-121.5001,-121.3209],
    bbox = [35.7622, 35.9909,-121.5001,-121.3209],
    az_looks = 1,
    rng_looks =2
    )


# Example 1: run with no date specification
#pairs = tops.make_pairs(path=data_path, options=options)

# Example 2: run with maxdelta
#pairs = tops.make_pairs(path=data_path, maxdelta  = maxdelta, options=options)

# Example 3: run with singlemaster
pairs = tops.make_pairs(path=data_path, singlemaster  = singlemaster, options=options)

# Example 4: run with dates
#pairs = tops.make_pairs(path = data_path, dates = dates, options = options)


for pair in pairs:
    print(pair.path)

prin(len(pairs))

for pair in pairs:
    if os.path.isdir(str(pair.path)) == True:
        print(str(pair.path) + ' exists, skipping')
    else:
       pair.run(overwrite=False)
       #print(str(pair.path) + ' will process')

#Reverse operation: make Pair objects from path to YYYYMMDD_YYYYMMDD directory
#mypair = tops.Pair.from_path('/net/tiampostorage/volume1/MyleneShare/Bigsur_asc/az1rng2/20170519_20170531')
#mypair = tops.Pair.from_path('/net/tiampostorage/volume1/MyleneShare/Bigsur_desc/faulty_processing_az1rng2/20160307_20160424')
# check to see if mypair processed correctly
#check = mypair.check_process()
