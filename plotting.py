from os import path
import pandas as pd
import seaborn as sns
from utils import get_node_colors

# load data
data_loc = path.join('Data','ProcessedData')
structuredata = pd.read_csv(path.join(data_loc, 'structuredata.csv'))
valuedata = pd.read_csv(path.join(data_loc, 'valuedata.csv'))


sns.plt.figure(figsize=(12,8))
for subset in valuedata.groupby('stim_repetition'):
    label = subset[0]
    sns.kdeplot(subset[1].rating, label = label, linewidth=3)
sns.plt.legend(fontsize=20, title='Stimulus Repetition')
sns.plt.xlabel('value Rating', fontsize = 18)
sns.plt.title('Value Rating as a function of Stimulus Repetition',
              fontsize = 22)

sns.plt.figure(figsize=(12,8))
for subset in valuedata.groupby('community'):
    label = subset[0]
    sns.kdeplot(subset[1].rating, label = label, linewidth=3)
sns.plt.legend(fontsize=20, title='Community')
sns.plt.xlabel('value Rating', fontsize = 18)
sns.plt.title('Value Rating as a function of Community',
              fontsize = 22)

sns.boxplot('stim_repetition','reg_rating',data=valuedata, hue = 'community')

# plot based on average community value
valuedata.loc[:,'avg_community_value'] = \
        valuedata.community.apply(
        lambda x: valuedata.groupby('community').\
        labeled_value.mean()[x])

colors = get_node_colors()
for group in valuedata.groupby('stim_index'):
    stim_i = int(group[0])
    subset = group[1]
    sns.kdeplot(subset.rating, label = stim_i, 
                linewidth=3, color = colors[stim_i])

