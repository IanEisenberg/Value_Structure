from glob import glob
import numpy as np
from os import path
import pandas as pd
import pickle
# get path of file
d = path.dirname(__file__)

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
    # convert stim set to categorical label
    RL_df.stim_set = RL_df.stim_set.astype(str)
    remapping = {k: str(v) for v,k in enumerate(RL_df.stim_set.unique())}
    for k,v in remapping.items():
        RL_df.stim_set.replace(k, v, inplace=True)
    RL_df.stim_set = RL_df.stim_set.astype(int)
    # add trials since a stim set switch
    switch_points = np.where(RL_df.stim_set.diff()==1)[0]
    trials_since_switch = []
    count=0
    for i in range(len(RL_df)):
        if i in switch_points:
            count=0
        trials_since_switch.append(count)
        count+=1
    RL_df.loc[:, 'trials_since_switch'] = trials_since_switch

    # drop unneeded
    RL_df.drop(['duration', 'feedback_duration', 
                'secondary_responses', 'secondary_rts'],
                axis=1,
                inplace=True)
    RL_df = RL_df.reindex_axis(sorted(RL_df.columns), axis=1)
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
    # add transition columns
    transition_node = structure_df.stim_index.apply(lambda x: x in [0,4,5,9,10,14])
    structure_df.insert(0, 'transition_node', transition_node)
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
        
    # drop unneeded
    structure_df.drop(['duration', 'secondary_responses', 'secondary_rts'],
                axis=1,
                inplace=True)
    structure_df = structure_df.reindex_axis(sorted(structure_df.columns), axis=1)
    return structure_df

def process_data(subj, post=True):
    RL_df, structure_df, metadata, descriptive_stats = load_data(subj, post)
    data = {'RL': RL_df,
            'structure': structure_df,
            'meta': metadata,
            'descriptive_stats': descriptive_stats}
    pickle.dump(data, open(path.join(d,'../Data/ProcessedData', 
                                     '%s_processed_data.pkl' % subj),'wb'))
    return data
    
# helper functions
def get_community(indices):
    communities = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14]]
    if type(indices) == list:
        community_labels = []
        for index in indices:
            community_labels.append([i for i, comm in enumerate(communities) if index in comm][0])
        return community_labels
        
    else:
        return [i for i, comm in enumerate(communities) if indices in comm][0]
            
