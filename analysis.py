import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from utils.data_preparation_utils import process_data

#test code

data = process_data('test')

RL = data['RL']
structure = data['structure']
meta = data['meta']
descriptive_stats = data['descriptive_stats']

# *****************************************************************************
# analyze structure task
# *****************************************************************************
m = smf.ols(formula='rt ~ C(rotation) + correct.shift() + community_transition + steps_since_seen',
            data = structure)
res = m.fit()
res.summary()

# descriptive plots
sns.set_context('poster')
sns.set_palette("Set1", 8, .75)
plt.figure(figsize=(12,8))
sns.boxplot(x='stim_index', y='rt', hue='transition_node', data=structure)

plt.figure(figsize=(12,8))
sns.pointplot(x='stim_index', y='correct', data=structure,   join=False, 
              hue='transition_node', scale=1, errwidth=3)

# *****************************************************************************
# analyze RL task
# *****************************************************************************
plt.figure(figsize=(12,8))
RL.rt.hist(bins=50)


from models import BasicRLModel, GraphRLModel


basic_m = BasicRLModel(RL)
graph_m = GraphRLModel(RL, meta['structure']['graph'])
for model in [basic_m, graph_m]:
    model.optimize()
    p, vals = model.run_data()

compare_df = RL.loc[:,['response','stim_indices', 'selected_stim']]
ignored = compare_df.apply(lambda x: x.stim_indices[int(1-x.response)] \
                                    if not pd.isnull(x.response) else np.nan, axis=1)