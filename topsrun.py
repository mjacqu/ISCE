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

data_path = '/net/tiampostorage/volume1/MyleneShare/BarryArm/data/ascending/F194'
dates = [('20200622', '20200704')]
maxdelta = datetime.timedelta(days = 12)
singleref = datetime.datetime(2017, 6, 26)
sequential = 1

# set dict with all attributes of pair:
options = dict(
    path = '/net/tiampostorage/volume1/MyleneShare/BarryArm/results/asc_194',
    swaths = [3],
    orbit ='/net/tiampostorage/volume1/LabShare/orbits/aux_resorb',
    auxiliary ='/home/myja3483/Landslides/Bigsur/auxfiles',
    unwrapper ='snaphu_mcf',
    dem = '/net/tiampostorage/volume1/MyleneShare/BarryArm/data/dem/stitched.dem.wgs84',
    roi = [61.08453, 61.19802,-148.27746,-148.03239],
    bbox = [61.08453, 61.19802,-148.27746,-148.03239],
    az_looks = 1,
    rng_looks =3,
    dense_offsets = False
    )


# Example 1: make pairs with dates
pairs = tops.make_pairs(path = data_path, dates = dates, options = options)

# Example 2: make pairs with maxdelta
pairs = tops.make_pairs(path=data_path, maxdelta  = maxdelta, options=options)

# Example 3: make pairs with singleref
pairs = tops.make_pairs(path=data_path, singleref = singleref, options=options)

# Example 4: make pairs with sequential pairing
pairs = tops.make_pairs(path=data_path, sequential = 1, options=options)

# Example 5: make pairs with any combination
pairs = tops.make_pairs(path=data_path, dates = dates, maxdelta  = maxdelta, singleref = singleref, sequential = 1, options=options)

for pair in pairs:
    print(pair.path)


print(len(pairs))

# run ISCE pairs with overwrite
#pair.run(overwrite = True)

#run ISCE pairs without overwrite
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
