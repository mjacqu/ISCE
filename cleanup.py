import sys
sys.path.append('/home/myja3483/isce_tools/MudCreek')
import move_dirs
import numpy as np
import os

#ascending data
root = '/net/tiampostorage/volume1/MyleneShare/Bigsur_asc'
file_list_asc = os.path.join(root,'az1rng2/low_coherence0.5.txt')
target_asc = os.path.join(root, 'low_coherence_az1rng2')
asc_path = os.path.join(root, 'az1rng2')
#move isce pairs
move_asc = np.loadtxt(file_list_asc, dtype = str)
#move_dirs.move_files(move_asc, target_asc)
#move coherence files
coh_path = os.path.join(asc_path, 'coherence')
asc_list = os.listdir(coh_path)
mv_files_asc = [os.path.join(coh_path, i) for i in move_dirs.Filter(asc_list, move_asc)]
move_dirs.move_files(mv_files_asc, target_asc)

#descending data
root = '/net/tiampostorage/volume1/MyleneShare/Bigsur_desc'
file_list_desc = os.path.join(root,'az1rng2/low_coherence0.5.txt')
target_desc = os.path.join(root, 'low_coherence_az1rng2')
desc_path = os.path.join(root, 'az1rng2')
#move isce pairs
move_desc = np.loadtxt(file_list_desc, dtype = str)
#move_dirs.move_files(move_desc, target_desc)
#move coherence files
coh_path = os.path.join(desc_path, 'coherence')
desc_list = os.listdir(coh_path)
mv_files_desc = [os.path.join(coh_path, i) for i in move_dirs.Filter(desc_list, move_desc)]
move_dirs.move_files(mv_files_desc, target_desc)
