import cPickle
from glob import glob
import os
import pandas as pd

structuredata = []
pricedata = []
communities = {0: [0,1,2,3,4],
              1: [5,6,7,8,9],
              2: [10,11,12,13,14]}

for filey in sorted(glob('Data/*/RawData/*')):
    subj = os.path.basename(filey).split('_')[0]
    data = cPickle.load(open(filey,'rb'))
    subj_structuredata = pd.DataFrame(data['structuredata'])
    subj_structuredata.loc[:,'subjid'] = subj
    subj_pricedata = pd.DataFrame(data['pricedata'])
    subj_pricedata.loc[:,'subjid'] = subj
    # concatenate with group
    structuredata.append(subj_structuredata)
    pricedata.append(subj_pricedata)

structuredata = pd.concat(structuredata, axis=0)
pricedata = pd.concat(pricedata, axis=0)

# add community columns
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
