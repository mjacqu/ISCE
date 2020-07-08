import numpy as np
import os
import shutil
import re


def move_files(move_list, target_path):
    '''
    Move directories listed in file_list to other target directory.
    '''
    if os.path.isdir(target_path) == True:
        [shutil.move(i, target_path) for i in move_list]
    else:
        print('Creating ' + os.path.basename(target_path) + ' in ' + os.path.dirname(target_path))
        [shutil.move(i, target_path) for i in move_list]



def Filter(string, substr):
    '''
    Find coherence files (.tif) listed in low_coherence.txt
    '''
    return [str for str in string
    if re.match(r'\d{8}_\d{8}', str).group(0) in substr]
