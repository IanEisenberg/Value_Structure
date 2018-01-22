from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
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
    structure.date = pd.to_datetime(structure.date)
    group_structure = pd.concat([group_structure, structure], axis=0)
    group_RL= pd.concat([group_RL, RL], axis=0)
    
# filter by date
filter_date = '2017-01-15'
group_structure = group_structure[group_structure.date > filter_date]
group_RL= group_RL[group_RL.date > filter_date]


# descriptive plots
# structure
sns.set_context('poster')
sns.set_palette("Set1", 8, .75)
f, axes = plt.subplots(1,2, figsize=(12,6))
sns.boxplot(x='subj_id', y='rt', hue='community_transition', 
            data=group_structure, ax=axes[0])
sns.barplot(x='subj_id', y='correct', data=group_structure, 
              hue='community_transition', errwidth=3, ax=axes[1])
axes[1].legend().remove()
axes[0].legend(title='Community Transition', 
                bbox_to_anchor=(1.1, -.2), 
                loc='upper center')
plt.ylim(.5,1)
plt.tight_layout()
plt.suptitle('Effects on Community Transition', y=1.05)


# RL
# set up auxilary dataframes
probe_trials = group_RL.stim_set.apply(lambda x: len(x) < 3)
correct_community = (group_RL.selected_community == 
                     group_RL.stim_indices.apply(lambda x: np.max(get_community(x))))
correct_community.rename('correct_community', inplace=True)
correct_community = pd.concat([correct_community, group_RL], axis=1)
correct_community = correct_community[probe_trials]
group_data = correct_community.groupby(['subj_id', 'stim_set', 'stim_set_cat']).mean().reset_index()
# descriptive stats
plt.figure(figsize=(24,8))
plt.subplot(1,2,1)
sns.pointplot(x='stim_set', y='correct_community', 
              data=group_data,
              hue='stim_set_cat',
              join=False,
              scale=1,
              err_width=3)
plt.title('Choices Between Identically Valued Stim')
group_data = group_RL[~probe_trials].query('display_reward==0').groupby(['stim_set','stim_set_cat','subj_id']).mean().reset_index()
plt.subplot(1,2,2)
sns.pointplot(x='stim_set', y='correct', 
              data=group_data,
              hue='stim_set_cat',
              join=False,
              scale=1,
              err_width=3)
plt.title('Choices during Blackout')