import numpy as np
import os


def delete_extras(dir):
    dir_p = os.path.join(path, dir)
    bashCommand = f'rm -r {dir_p}/coarse* {dir_p}/ESD {dir_p}/fine* {dir_p}/geom_reference {dir_p}/PICKLE {dir_p}/reference {dir_p}/secondary'
    os.system(bashCommand)


def reduce_merged(dir):
    dir_p = os.path.join(path, dir)
    bashCommand = f'rm {dir_p}/merged/dem* {dir_p}/merged/topophase* {dir_p}/merged/filt_topophase.flat'
    os.system(bashCommand)

path = './'

dirs = os.listdir(path)

for d in dirs:
    delete_extras(d)
    reduce_merged(d)