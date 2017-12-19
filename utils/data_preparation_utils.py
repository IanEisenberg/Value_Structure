from glob import glob
from os import path
import pandas as pd
import pickle

def load_data(subj):
    RL_file, structure_file = glob(path.join('../Data/RawData','*%s*' % subj))
    assert 'RL' in RL_file
    #unpickle
    RL_unpickled = pickle.load(open(RL_file,'rb'))
    structure_unpickled = pickle.load(open(structure_file,'rb'))
    #convert to dataframes
    RL_df = pd.DataFrame(RL_unpickled['RLdata'])
    structure_df = pd.DataFrame(structure_unpickled['structuredata'])
    # modify dataframes
    RL_df = post_process_RL(RL_df)
    structure_df = post_process_structure(structure_df)
    return RL_df, structure_df
    
def post_process_RL(RL_df):
    # add new columns
    stim_choices = RL_df.apply(lambda x: x['stim_indices'][x['response']], axis=1)
    RL_df.insert(0, 'selected_stim', stim_choices)
    # drop unneeded
    RL_df.drop(['duration', 'feedback_duration', 
                'secondary_responses', 'secondary_rts'],
                axis=1,
                inplace=True)
    RL_df = RL_df.reindex_axis(sorted(RL_df.columns), axis=1)
    return RL_df

def post_process_structure(structure_df):
    structure_df = structure_df.query('exp_stage == "structure_learning"')
    # add new columns
    # add community column
    community = structure_df.stim_index.apply(get_community)
    structure_df.insert(0, 'community', community)
    structure_df.insert(0, 'community_transition',
                        structure_df.community.diff())
    # drop unneeded
    structure_df.drop(['duration', 'secondary_responses', 'secondary_rts'],
                axis=1,
                inplace=True)
    structure_df = structure_df.reindex_axis(sorted(structure_df.columns), axis=1)
    return structure_df

def process_data(subj):
    RL_df, structure_df = load_data(subj)
    data = {'RL': RL_df,
            'structure': structure_df,
            'meta': {}}
    pickle.dump(data, open(path.join('../Data/ProcessedData', 
                                     '%s_processed_data.pkl' % subj),'wb'))
    
# helper functions
def get_community(index):
    communities = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14]]
    return [i for i, comm in enumerate(communities) if index in comm][0]