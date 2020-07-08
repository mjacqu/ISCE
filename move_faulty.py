import sys
sys.path.append('/home/myja3483/isce_tools/ISCE')
import os
import numpy
import tops
import re

path = '/net/tiampostorage/volume1/MyleneShare/BarryArm/results/asc_199'
#path = '/net/tiampostorage/volume1/MyleneShare/Bigsur_desc/az1rng2'

dir_list = os.listdir(path)
regex = re.compile(r'\d{8}_\d{8}')
pair_dirs = [os.path.join(path, d) for d in list(filter(regex.search, dir_list))]
faulty = []
for p in pair_dirs:
    mypair = tops.Pair.from_path(p)
    check = mypair.check_process()
    if check == False:
        print(os.path.basename(mypair.path) + ' failed')
        faulty.append(os.path.basename(mypair.path))


#write faulty list to .txt file:
with open(os.path.join(path,'faulty_processing.txt'), 'w') as f:
    for item in faulty:
        f.write("%s\n" % item)
