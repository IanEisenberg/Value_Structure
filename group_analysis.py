from glob import glob
from os import path
import pandas as pd
from utils.data_preparation_utils import get_community, process_data

subjs = []
subj_data = []
group_structure = pd.DataFrame()
group_RL = pd.DataFrame()

for p in sorted(glob('Data/RawData/*RL*')):
    date = path.basename(p).split('_')[2]
    subj = path.basename(p).split('_')[0]
    subjs.append(subj)
    data = process_data(subj, overwrite=False)
    RL = data['RL']
    structure = data['structure']
    # add subject columns
    RL.loc[:, 'subj_id'] = subj
    structure.loc[:, 'subj_id'] = subj
    # add date columns
    RL.loc[:, 'date'] = date
    structure.loc[:, 'date'] = date
    RL.date = pd.to_datetime(RL.date)
    RL.structure = pd.to_datetime(RL.structure)
    group_structure = pd.concat([group_structure, structure], axis=0)
    group_RL= pd.concat([group_RL, RL], axis=0)
    
# filter by date
filter_date = '2018-01-15'
group_structure = group_structure[group_structure.date > filter_date]
group_RL= group_RL[group_RL.date > filter_date]
