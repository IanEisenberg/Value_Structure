from glob import glob
import numpy as np
from os import path
import pandas as pd
import pickle
from utils.utils import get_community

# get path of file
d = path.dirname(__file__)
pd.options.mode.chained_assignment = None 

def load_data(subj, post=True):
    RL_file, structure_file = sorted(glob(path.join(d, '../Data/RawData','*%s*' % subj)))
    assert 'RL' in RL_file
    #unpickle
    RL_unpickled = pickle.load(open(RL_file,'rb'))
    structure_unpickled = pickle.load(open(structure_file,'rb'))
    # combine metadata
    metadata = {'RL': RL_unpickled['taskdata'],
                'structure': structure_unpickled['taskdata']}
    #convert to dataframes
    RL_df = pd.DataFrame(RL_unpickled['RLdata'])
    structure_df = pd.DataFrame(structure_unpickled['structuredata'])
    # get descriptive stats
    descriptive_stats = get_descriptive_stats(RL_df, structure_df)
    # modify dataframes
    if post == True:
        RL_df = post_process_RL(RL_df)
        structure_df = post_process_structure(structure_df)
    return RL_df, structure_df, metadata, descriptive_stats

def get_descriptive_stats(RL_df, structure_df):
    stats = {}
    for name, df in [('RL', RL_df), ('structure', structure_df)]:
        stats[name] = {}
        stats[name]['missed_percent'] = np.mean(df.rt.isnull())
        stats[name]['avg_rt'] = df.rt.median()
        stats[name]['accuracy'] = df.correct.mean()
    return stats

def post_process_RL(RL_df):
    # scrub data
    RL_df = RL_df.query('exp_stage == "RL_task"')
    RL_df = RL_df[~RL_df.rt.isnull()]
    # add new columns
    stim_choices = RL_df.apply(lambda x: x.stim_indices[int(x.response)] \
                                         if not pd.isnull(x.response) else np.nan, axis=1)
    value_choices = RL_df.apply(lambda x: x['values'][int(x.response)] \
                                         if not pd.isnull(x.response) else np.nan, axis=1)
    
    RL_df.insert(0, 'selected_stim', stim_choices)
    RL_df.insert(0, 'selected_value', value_choices)
    # add community column
    community = RL_df.selected_stim.apply(get_community) 
    RL_df.insert(0, 'selected_community', community)
    
    # add trials since a stim set switch
    curr = RL_df.stim_set.iloc[0]
    trials_since_switch = []
    count=0
    for i in RL_df.stim_set:
        if i != curr:
            count=0
            curr = i
        trials_since_switch.append(count)
        count+=1
    RL_df.loc[:, 'trials_since_switch'] = trials_since_switch
    # add categorical stim_set column
    try:
        mapping = {k:v for v,k in enumerate(RL_df.stim_set.unique())}
    except TypeError:
        RL_df.stim_set = RL_df.stim_set.apply(lambda x: tuple(x))
        mapping = {k:v for v,k in enumerate(RL_df.stim_set.unique())}
    stim_set_cat = []
    for s in RL_df.stim_set:
        stim_set_cat.append(mapping[s])
    RL_df.insert(0, 'stim_set_cat', stim_set_cat)
    # convert column types
    RL_df.correct = RL_df.correct.astype(float)
    # drop unneeded
    RL_df.drop(['duration', 'feedback_duration', 
                'secondary_responses', 'secondary_rts'],
                axis=1,
                inplace=True)
    RL_df = RL_df.reindex(sorted(RL_df.columns), axis=1)
    return RL_df

def post_process_structure(structure_df):
    # scrub data
    structure_df = structure_df.query('exp_stage == "structure_learning"')
    # replace "correct" nans with False if there is a rt
    index = structure_df.query('rt==rt and correct!=correct').index
    structure_df.loc[index,'correct']=False
    
    # add new columns
    # add community column
    community = structure_df.stim_index.apply(get_community)
    structure_df.insert(0, 'community', community)
    structure_df.insert(0, 'community_transition',
                        structure_df.community.diff()!=0)
    structure_df.loc[0, 'community_transition'] = np.nan
    # add transition columns
    bridge_node = structure_df.stim_index.apply(lambda x: x in [0,4,5,9,10,14])
    structure_df.insert(0, 'bridge_node', bridge_node)
    # trials since last seen
    steps_since_seen = []
    last_seen = {}
    # ...for value data
    for i, stim in enumerate(structure_df.stim_index):
        if stim in last_seen.keys():
            steps_since = i-last_seen[stim]
        else:
            steps_since = 0
        last_seen[stim] = i
        steps_since_seen.append(steps_since)
    structure_df.loc[:,'steps_since_seen'] = steps_since_seen 
    # convert column types
    structure_df.correct = structure_df.correct.astype(float)
    # drop unneeded
    structure_df.drop(['duration', 'secondary_responses', 'secondary_rts'],
                axis=1,
                inplace=True)
    structure_df = structure_df.reindex(sorted(structure_df.columns), axis=1)
    return structure_df

def process_data(subj, overwrite=False):
    processed_file = path.join(d,'../Data/ProcessedData', 
                                     '%s_processed_data.pkl' % subj)
    if not overwrite and path.exists(processed_file):
        data = pickle.load(open(processed_file,'rb'))
    else:
        RL_df, structure_df, metadata, descriptive_stats = load_data(subj)
        data = {'RL': RL_df,
                'structure': structure_df,
                'meta': metadata,
                'descriptive_stats': descriptive_stats}
        pickle.dump(data, open(processed_file,'wb'))
    return data
    

