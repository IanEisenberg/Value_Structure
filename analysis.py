import cPickle
from glob import glob
import numpy as np
from numpy import log
from os import path
import pandas as pd
import statsmodels.formula.api as smf

# **********Processing*********************************
structuredata = []
valuedata = []
taskdata = {}
communities = {0: [0,1,2,3,4],
              1: [5,6,7,8,9],
              2: [10,11,12,13,14]}
node_lookup = {i:k for k,v in communities.items() for i in v }
f = lambda x: [k for k,v in communities.items() if x in v][0]

datafiles = sorted(glob('Data/RawData/*pkl'))
for exclude in ['CH']:
    datafiles = [i for i in datafiles if exclude not in i]

for filey in datafiles:
    subj = path.basename(filey).split('_')[0]
    data = cPickle.load(open(filey,'rb'))
    subj_structuredata = pd.DataFrame(data['structuredata'])
    subj_structuredata.loc[:,'subjid'] = subj
    subj_valuedata = pd.DataFrame(data['valuedata'])
    subj_valuedata.loc[:,'subjid'] = subj
    
    # add subject specific variables
    
    # for structuredata
    subj_structuredata.loc[:,'correct_shift'] = subj_structuredata.correct.shift()
    # add community columns
    subj_structuredata.loc[:,'community'] = subj_structuredata.stim_index.apply(f)
    subj_structuredata.loc[1:,'community_cross'] = (subj_structuredata.community.diff()!=0)[1:]
    # calculate steps within the community
    steps_in_community = []
    steps = 0
    for cross in subj_structuredata.community_cross:
        if cross==False:
            steps+=1
        else:
            steps=1
        steps_in_community.append(steps)
    subj_structuredata.loc[:,'steps_within_community'] = steps_in_community
    # trials since last seen
    steps_since_seen = []
    last_seen = {}
    # ...for value data
    for i, stim in enumerate(subj_structuredata.stim_index):
        if stim in last_seen.keys():
            steps_since = i-last_seen[stim]
        else:
            steps_since = 0
        last_seen[stim] = i
        steps_since_seen.append(steps_since)
    subj_structuredata.loc[:,'steps_since_seen'] = steps_since_seen  
    
    # for value data
    # regress out accuracy
    structure_acc = subj_structuredata[200:].groupby('stim_index').correct.mean()
    subj_valuedata.loc[:,'stim_acc'] = subj_valuedata.stim_index.apply(lambda x: structure_acc[x])
    rs = smf.ols('rating ~ stim_acc', subj_valuedata).fit()
    subj_valuedata.loc[:,'reg_rating'] = rs.resid
    
    # add stim repetitions
    n_stim = len(np.unique(subj_valuedata.stim_index))
    n_repeats = (len(subj_valuedata))//n_stim
    repeat_array = np.hstack([[i]*n_stim for i in range(n_repeats)])
    subj_valuedata.loc[:, 'stim_repetition'] = repeat_array
    # node values
    node_values = data['taskdata']['node_values']
    value_labels = subj_valuedata.stim_index. \
                    apply(lambda x: node_values.get(x,np.nan))
    subj_valuedata.loc[:,'labeled_value'] = value_labels
    # add community
    subj_valuedata.loc[:,'community'] = subj_valuedata.stim_index.apply(f)

    # concatenate with group
    structuredata.append(subj_structuredata)
    valuedata.append(subj_valuedata)
    taskdata[subj] = data['taskdata']

structuredata = pd.concat(structuredata, axis=0)
valuedata = pd.concat(valuedata, axis=0)

# save data
save_loc = path.join('Data','ProcessedData')
structuredata.to_csv(path.join(save_loc, 'structuredata.csv'))
valuedata.to_csv(path.join(save_loc, 'valuedata.csv'))
cPickle.dump(taskdata,open(path.join(save_loc,'taskdata.pkl'),'wb'))

# **********Analysis*********************************
regress_data = structuredata.query('rt!=-1')

# structure reaction time data
models = {}
for DV in ['rt', 'np.log(rt)']:
    rs = smf.mixedlm("%s ~ community_cross + steps_since_seen" % DV, 
                     regress_data, groups=regress_data["subjid"])
    # larger model
    rs_full = smf.mixedlm("%s ~ community_cross + steps_since_seen \
                          + correct_shift + C(rotation)" % DV, 
                     regress_data, groups=regress_data["subjid"])
    DV_name = DV[-7:]
    models[DV_name] = rs
    models[DV_name + '_full'] = rs_full
    
