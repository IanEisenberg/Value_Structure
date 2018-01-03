import matplotlib.pyplot as plt
from models import BasicRLModel, GraphRLModel, SR_RLModel
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils.data_preparation_utils import get_community, process_data

#test code

data = process_data('651')

RL = data['RL']
structure = data['structure']
meta = data['meta']
descriptive_stats = data['descriptive_stats']

# *****************************************************************************
# analyze structure task
# *****************************************************************************
m = smf.ols(formula='rt ~ C(nback_match) + correct.shift() + community_transition + steps_since_seen',
            data = structure)
res = m.fit()
res.summary()

# descriptive plots
sns.set_context('poster')
sns.set_palette("Set1", 8, .75)
plt.figure(figsize=(24,16))
plt.subplot(2,2,1)
sns.boxplot(x='stim_index', y='rt', hue='transition_node', data=structure)

plt.subplot(2,2,2)
sns.pointplot(x='stim_index', y='correct', data=structure,   join=False, 
              hue='transition_node', scale=1, errwidth=3)
# average across stim
plt.subplot(2,2,3)
sns.boxplot(x='transition_node', y='rt', data=structure)

plt.subplot(2,2,4)
sns.pointplot(x='transition_node', y='correct', data=structure, join=False, 
              hue='transition_node', scale=1, errwidth=3)

# *****************************************************************************
# analyze RL task
# *****************************************************************************
# descriptive stats
plt.figure(figsize=(12,8))
RL.rt.hist(bins=50)


RL.correct = RL.correct.astype(int)
m = smf.glm(formula='correct ~ trials_since_switch * stim_set', data=RL.query('stim_set<4'), family=sm.families.Binomial())
res = m.fit()
res.summary()

# fit models
switch_points = np.where(RL.stim_set.diff()==1)[0]
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

# before switch
basic_m = BasicRLModel(RL)
SR_m = SR_RLModel(RL, structure)
for model in [basic_m, SR_m]:
    model.optimize(stop=switch_points[0])
    p, vals = model.run_data()
models['before_switch'] = {'basic': basic_m,
                           'SR': SR_m}
# after switch
basic_m = BasicRLModel(RL)
SR_m = SR_RLModel(RL, structure)
for model in [basic_m, SR_m]:
    model.optimize(start=switch_points[0], stop=None)
    p, vals = model.run_data()
models['after_switch'] = {'basic': basic_m,
                           'SR': SR_m}

for key,vals in models.items():
    print('*'*79)
    if key == 'after_switch':
        start = switch_points[0]
        stop = switch_points[1]
    else:
        start = None
        stop = switch_points[0]
    print('%s: start %s, stop %s' % (key, start, stop))
    for name, m in vals.items():
        print('%s: ' % name, m.get_log_likelihood(start, stop))


# drive M from one step transition probabilities
from utils.graph_utils import graph_to_matrix
from models import SR_from_transition
graph = meta['structure']['graph']
onestep = graph_to_matrix(graph)/5.0
TrueM = SR_from_transition(onestep, SR_m.gamma)
M = SR_m.get_M()
np.corrcoef(M.flatten(),TrueM.flatten())
