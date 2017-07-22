import cPickle
from glob import glob
import numpy as np
from os import path
import pandas as pd

structuredata = []
pricedata = []
taskdata = {}
communities = {0: [0,1,2,3,4],
              1: [5,6,7,8,9],
              2: [10,11,12,13,14]}
node_lookup = {i:k for k,v in communities.items() for i in v }

for filey in sorted(glob('Data/RawData/*22_13*pkl')):
    subj = path.basename(filey).split('_')[0]
    data = cPickle.load(open(filey,'rb'))
    subj_structuredata = pd.DataFrame(data['structuredata'])
    subj_structuredata.loc[:,'subjid'] = subj
    subj_pricedata = pd.DataFrame(data['pricedata'])
    subj_pricedata.loc[:,'subjid'] = subj
    # add subject specific variables
    n_stim = int(np.max(subj_pricedata.stim_index)+1)
    n_repeats = (len(subj_pricedata)-1)//n_stim
    repeat_array = np.hstack([[i]*n_stim for i in range(1, n_repeats+1)])
    subj_pricedata.loc[1:, 'stim_repetition'] = repeat_array
    # node values
    node_values = dict(data['taskdata']['labeled_nodes'])
    price_labels = subj_pricedata.stim_index. \
                    apply(lambda x: node_values.get(x,np.nan))
    subj_pricedata.loc[:,'labeled_price'] = price_labels
    # concatenate with group
    structuredata.append(subj_structuredata)
    pricedata.append(subj_pricedata)
    taskdata[subj] = data['taskdata']

structuredata = pd.concat(structuredata, axis=0)
pricedata = pd.concat(pricedata, axis=0)

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
# ...for price data
pricedata.loc[1:,'community'] = pricedata.stim_index[1:].apply(f)

# save data
save_loc = path.join('Data','ProcessedData')
structuredata.to_csv(path.join(save_loc, 'structuredata.csv'))
pricedata.to_csv(path.join(save_loc, 'pricedata.csv'))
