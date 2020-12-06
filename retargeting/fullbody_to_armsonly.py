import json 
import glob
import os
from tqdm import tqdm

source_dir = '../DeepMimic/data/motions/MPI_Limits'

if not os.path.exists(source_dir+'_armsonly'):
    os.makedirs(source_dir+'_armsonly')

all_motion_files = glob.glob(source_dir+'/*')
for mf in all_motion_files:
    print('Processing ' + mf)
    with open(mf) as f:
        data = json.load(f)

    new_data = []
    for frame in data['Frames']:
    	# new_data.append([frame[0]] + [0, 0.78, 0] + [1,0,0,0] + frame[8:16] + frame[25:30] + frame[39:])
        new_data.append(frame[:16] + frame[25:30] + frame[39:])

    data['Frames'] = new_data

    with open(os.path.join(source_dir+'_armsonly', mf.split('/')[-1]), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
