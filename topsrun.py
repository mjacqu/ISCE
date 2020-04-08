#import sys
#sys.path.append('/home/myja3483/isce_tools/ISCE')
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

data_path = '.'
dates = (
    ('20160822', '20160915'),
    ('20170113', '20160927'))
#maxdelta = datetime.timedelta(days = 48)
#singlemaster = datetime.datetime(2017, 5, 7)

# set dict with all attributes of pair:
options = dict(
    path = data_path,
    swaths = [3],
    orbit ='/Users/mistral/Documents/CUBoulder/',
    auxiliary ='/Users/mistral/Documents/',
    unwrapper ='snaphu_mcf',
    dem = '/Users/mistral/Documents/CUBoulder/',
    roi = [35.7622, 35.9909,-121.5001,-121.3209],
    bbox = [35.7622, 35.9909,-121.5001,-121.3209],
    az_looks = 3,
    rng_looks =7
    )

# Example 1: run with maxdelta
#pairs = tops.makepairs(path=data_path, maxdelta = maxdelta, options=options)

# Example 2: runt with dates
pairs = tops.make_pairs(path = data_path, dates = dates, options = options)

'''
for pair in pairs:
    print(pair.path)
'''

for pair in pairs:
    if os.path.isdir(str(pair.path)) == True:
        print(str(pair.path) + ' exists, skipping')
    else:
        pair.run(overwrite=False)

#Reverse operation: make Pair objects from path to YYYYMMDD_YYYYMMDD directory
mypair = tops.Pair.from_path('/net/tiampostorage/volume1/MyleneShare/Bigsur_desc/az1rng2/20150301_20150325')
# check to see if mypair processed correctly
check = mypair.check_process()
