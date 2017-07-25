import cPickle
from glob import glob
import numpy as np
from numpy import log
from os import path
import pandas as pd

# **********Processing*********************************
structuredata = []
valuedata = []
taskdata = {}
communities = {0: [0,1,2,3,4],
              1: [5,6,7,8,9],
              2: [10,11,12,13,14]}
node_lookup = {i:k for k,v in communities.items() for i in v }

for filey in sorted(glob('Data/RawData/*pkl')):
    subj = path.basename(filey).split('_')[0]
    data = cPickle.load(open(filey,'rb'))
    subj_structuredata = pd.DataFrame(data['structuredata'])
    subj_structuredata.loc[:,'subjid'] = subj
    subj_valuedata = pd.DataFrame(data['valuedata'])
    subj_valuedata.loc[:,'subjid'] = subj
    # add subject specific variables
    subj_structuredata.loc[:,'correct_shift'] = subj_structuredata.correct.shift()
    n_stim = len(np.unique(subj_valuedata.stim_index))
    n_repeats = (len(subj_valuedata))//n_stim
    repeat_array = np.hstack([[i]*n_stim for i in range(n_repeats)])
    subj_valuedata.loc[:, 'stim_repetition'] = repeat_array
    # node values
    node_values = data['taskdata']['node_values']
    value_labels = subj_valuedata.stim_index. \
                    apply(lambda x: node_values.get(x,np.nan))
    subj_valuedata.loc[:,'labeled_value'] = value_labels
    # concatenate with group
    structuredata.append(subj_structuredata)
    valuedata.append(subj_valuedata)
    taskdata[subj] = data['taskdata']

structuredata = pd.concat(structuredata, axis=0)
valuedata = pd.concat(valuedata, axis=0)

# add community columns
# ... for structure data
f = lambda x: [k for k,v in communities.items() if x in v][0]
structuredata.loc[:,'community'] = structuredata.stim_index.apply(f)
structuredata.loc[1:,'community_cross'] = (structuredata.community.diff()!=0)[1:]
# calculate steps within the community
steps_in_community = []
steps = 0
for cross in structuredata.community_cross:
    if cross==False:
        steps+=1
    else:
        steps=1
    steps_in_community.append(steps)
structuredata.loc[:,'steps_within_community'] = steps_in_community
# trials since last seen
steps_since_seen = []
last_seen = {}
# ...for value data
for i, stim in enumerate(structuredata.stim_index):
    if stim in last_seen.keys():
        steps_since = i-last_seen[stim]
    else:
        steps_since = 0
    last_seen[stim] = i
    steps_since_seen.append(steps_since)
structuredata.loc[:,'steps_since_seen'] = steps_since_seen  


valuedata.loc[:,'community'] = valuedata.stim_index.apply(f)

# save data
save_loc = path.join('Data','ProcessedData')
structuredata.to_csv(path.join(save_loc, 'structuredata.csv'))
valuedata.to_csv(path.join(save_loc, 'valuedata.csv'))
cPickle.dump(taskdata,open(path.join(save_loc,'taskdata.pkl'),'wb'))

# **********Analysis*********************************
import statsmodels.formula.api as smf
data = structuredata[1:].query('rt!=-1')

models = {}
for DV in ['rt', 'np.log(rt)']:
    rs = smf.mixedlm("%s ~ community_cross + steps_since_seen" % DV, 
                     data, groups=data["subjid"])
    # larger model
    rs_full = smf.mixedlm("%s ~ community_cross + steps_since_seen \
                          + correct_shift + C(rotation)" % DV, 
                     data, groups=data["subjid"])
    DV_name = DV[-7:]
    models[DV_name] = rs
    models[DV_name + '_full'] = rs_full
    
