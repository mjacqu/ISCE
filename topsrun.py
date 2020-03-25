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

data_path = '/Users/mistral/Documents/CUBoulder/Science/gernika/isce_tools/test'
maxdelta = datetime.timedelta(days = 19)
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

pairs = tops.makepairs(path=data_path, maxdelta = maxdelta, options=options)

for pair in pairs:
    if os.path.isdir(str(pair.path)) == True:
        print(str(pair.path) + ' exists, skipping')
    else:
        pair.run(overwrite=False)
