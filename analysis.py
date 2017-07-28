import cPickle
from glob import glob
from itertools import permutations
import numpy as np
from numpy import log
from os import path
import pandas as pd
import statsmodels.formula.api as smf
from utils import scale

# **********Processing*********************************
structuredata = []
valuedata = []
taskdata = {}
winning = {}
communities = {0: [0,1,2,3,4],
              2: [5],
              1: [6,7,8,9,10]}
node_lookup = {i:k for k,v in communities.items() for i in v }
f = lambda x: [k for k,v in communities.items() if x in v][0]

datafiles = sorted(glob('Data/RawData/*pkl'))
for exclude in ['/CH', '/GL', '/YH']:
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
    subj_structuredata.loc[:,'congruent_rot'] = subj_structuredata.rotation==subj_structuredata.rotation.shift()
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
    stim_freq = subj_structuredata.groupby('stim_index').rt.count()
    subj_valuedata.loc[:,'stim_acc'] = subj_valuedata.stim_index.apply(lambda x: structure_acc[x])
    subj_valuedata.loc[:,'stim_freq'] = subj_valuedata.stim_index.apply(lambda x: stim_freq[x])
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
    winning[subj] = data['total_win']
    
structuredata = pd.concat(structuredata, axis=0)
valuedata = pd.concat(valuedata, axis=0)

# Remove subjects who had no variability on their ratings
rating_variability = valuedata.groupby(['subjid']).rating.std()
drop_subjects = list(rating_variability.index[rating_variability<.1])
valuedata = valuedata.query('subjid not in %s' % drop_subjects)
structuredata = structuredata.query('subjid not in %s' % drop_subjects)

# **********Analysis*********************************
results = {}

regress_data = structuredata.query('rt!=-1').dropna(subset=['correct_shift'])
#regress_data = structuredata.query('rt!=-1 and correct==True').dropna(subset=['correct_shift'])

# structure reaction time data
models = {}
for DV in ['rt']:
    rs = smf.mixedlm("%s ~ community_cross + steps_since_seen" % DV, 
                     data=regress_data, groups='subjid',
                     re_formula="~community_cross")
    # larger model
    rs_full = smf.mixedlm("%s ~ community_cross + steps_since_seen \
                          + correct_shift + C(rotation) + congruent_rot" % DV, 
                     data=regress_data, groups=regress_data["subjid"],
                     re_formula="~community_cross")
    DV_name = DV[-7:]
    models[DV_name] = rs
    models[DV_name + '_full'] = rs_full

# individual difference variables
rt_raneffects = models['rt_full'].fit().random_effects
structure_coefficients = {k:v.loc['community_cross[T.True]'] for k,v in rt_raneffects.items()}
results['structure_coefficients'] = structure_coefficients

# value effect as a function of community
valuedata.loc[:,'structure_coefficient'] = valuedata.subjid.apply(lambda x: structure_coefficients[x])
value_rs = smf.mixedlm('rating ~ C(community)+ stim_acc + C(stim_file)', 
                 data = valuedata.query('community!=2'), 
                 groups='subjid', re_formula="~C(community)").fit()

interaction_value_rs = smf.mixedlm('rating ~ C(community)*structure_coefficient + stim_acc + C(stim_file)', 
                 data = valuedata.query('community!=2'), 
                 groups='subjid').fit()

value_raneffects = value_rs.random_effects
value_coefficients = {k:v.loc['C(community)[T.1]'] for k,v in value_raneffects.items()}
results['value_coefficients'] = value_coefficients

# calculate standard deviation of clusters within community
group_stats = {}
for subj in valuedata.subjid.unique():
    subj_stats = {}
    permutes = list(permutations(range(7),7))
    ratings = valuedata.query('subjid == "%s"' % subj) \
                        .groupby(['subjid','stim_index','community']) \
                        .mean().loc[:,['rating','reg_rating']].reset_index()
    
    std_community = (ratings.query('community != 2').query('community != 2') \
                    .groupby(['subjid','community']) \
                    .reg_rating.std()).groupby('community').mean().mean()
    subj_stats['community_std'] = std_community
    permute_stds = []
    for permute in permutes:
        n = len(ratings.subjid.unique())
        permute_pop = ratings.copy().reset_index()
        permute_pop.loc[:, 'community'] = permute_pop.community.iloc[list(permute)].reset_index()
        permute_std = (permute_pop.groupby(['subjid','community']).reg_rating.std()).groupby('community').mean().mean()
        permute_stds.append(permute_std)
    subj_stats['permuted_array']  = permute_stds
    subj_stats['p_val'] = np.mean([std_community>=i for i in permute_stds])
    group_stats[subj] = subj_stats

results['clustering'] = {k:v['p_val'] for k,v in group_stats.items()}
results['example_clustering_dist'] = group_stats.values()[0]['permuted_array']

# structure data win stay/lose switch (or just change in attention following error?)
for rot in [0,90]:
    structuredata[(structuredata.rotation.shift()==rot) 
                    & (structuredata.rt!=-1)] \
                .groupby(['correct_shift','rotation']).correct.mean()

    structuredata[(structuredata.rotation.shift()==rot) 
                & (structuredata.rt!=-1)] \
            .groupby(['correct_shift','rotation']).rt.median()


# save data
save_loc = path.join('Data','ProcessedData')
structuredata.to_csv(path.join(save_loc, 'structuredata.csv'))
valuedata.to_csv(path.join(save_loc, 'valuedata.csv'))
cPickle.dump(taskdata,open(path.join(save_loc,'taskdata.pkl'),'wb'))
cPickle.dump(results, open(path.join('Analysis_Results','analysis.pkl'), 'wb'))










