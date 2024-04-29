import os
from os import walk
import mne

def get_data_from_file(path):
    filenames = sorted(next(walk(path), (None, None, []))[2])
 
    res = []
    for filename in filenames:
        file_with_path = f'{path}/{filename}'
        data = mne.io.read_raw_edf(file_with_path)
        res.append(data)
    
    return res