import matplotlib.pyplot as plt
from models import BasicRLModel, GraphRLModel, SR_RLModel
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils.data_preparation_utils import get_community, process_data

#test code

data = process_data('502', overwrite=True)

RL = data['RL']
structure = data['structure']
meta = data['meta']
descriptive_stats = data['descriptive_stats']

# *****************************************************************************
# analyze structure task
# *****************************************************************************
if 'nback_match' in structure.columns:
    m = smf.ols(formula='rt ~ C(nback_match) + correct.shift() + community_transition + steps_since_seen',
                data = structure)
else:
    m = smf.ols(formula='rt ~ C(rotation) + correct.shift() + community_transition + steps_since_seen',
            data = structure)
res = m.fit()
res.summary()

# descriptive plots
sns.set_context('poster')
sns.set_palette("Set1", 8, .75)
plt.figure(figsize=(24,16))
plt.subplot(2,2,1)
sns.boxplot(x='stim_index', y='rt', hue='bridge_node', data=structure)

plt.subplot(2,2,2)
sns.pointplot(x='stim_index', y='correct', data=structure, join=False, 
              hue='bridge_node', scale=1, errwidth=3)
# average across stim
plt.subplot(2,2,3)
sns.boxplot(x='community_transition', y='rt', data=structure)

plt.subplot(2,2,4)
sns.pointplot(x='community_transition', y='correct', data=structure, join=False, 
              hue='community_transition', scale=1, errwidth=3)

# *****************************************************************************
# analyze RL task
# *****************************************************************************
# set up auxilary dataframes
probe_trials = RL.stim_set.apply(lambda x: len(x) < 3)
correct_community = (RL.selected_community == 
                     RL.stim_indices.apply(lambda x: np.max(get_community(x))))
correct_community = pd.concat([correct_community, RL.loc[:,['stim_set', 'stim_set_cat']]], axis=1)
correct_community = correct_community[probe_trials]
correct_community.columns = ['correct', 'stim_set','stim_set_cat']

# descriptive stats
plt.figure(figsize=(24,8))
plt.subplot(1,2,1)
sns.pointplot(x='stim_set', y='correct', 
              data=correct_community,
              hue='stim_set_cat',
              join=False,
              scale=1,
              err_width=3)
plt.title('Choices Between Identically Valued Stim')
plt.subplot(1,2,2)
sns.pointplot(x='stim_set', y='correct', 
              data=RL[~probe_trials].query('display_reward==0'),
              hue='stim_set_cat',
              join=False,
              scale=1,
              err_width=3)
plt.title('Choices during Blackout')

# evaluate whether performance went up over time
m = smf.glm(formula='correct ~ trials_since_switch', data=RL[~probe_trials], family=sm.families.Binomial())
res = m.fit()
res.summary()

# fit models
switch_points = np.where(RL.stim_set_cat.diff()==1)[0]
# total
# before switch
models = {}
basic_m = BasicRLModel(RL)
SR_m = SR_RLModel(RL, structure)
for model in [basic_m, SR_m]:
    model.optimize()
    p, vals = model.run_data()
models['all'] = {'basic': basic_m,
                  'SR': SR_m}

