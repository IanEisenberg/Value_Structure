from os import path
import pandas as pd
import seaborn as sns
from utils import get_node_colors

# load data
data_loc = path.join('Data','ProcessedData')
structuredata = pd.read_csv(path.join(data_loc, 'structuredata.csv'))
pricedata = pd.read_csv(path.join(data_loc, 'pricedata.csv'))


sns.plt.figure(figsize=(12,8))
for subset in pricedata.groupby('stim_repetition'):
    label = subset[0]
    sns.kdeplot(subset[1].rating, label = label, linewidth=3)
sns.plt.legend(fontsize=20, title='Stimulus Repetition')
sns.plt.xlabel('Price Rating', fontsize = 18)
sns.plt.title('Price Rating as a function of Stimulus Repetition',
              fontsize = 22)

sns.plt.figure(figsize=(12,8))
for subset in pricedata.groupby('community'):
    label = subset[0]
    sns.kdeplot(subset[1].rating, label = label, linewidth=3)
sns.plt.legend(fontsize=20, title='Community')
sns.plt.xlabel('Price Rating', fontsize = 18)
sns.plt.title('Price Rating as a function of Community',
              fontsize = 22)

# plot based on average community price
pricedata.loc[:,'avg_community_price'] = \
        pricedata.community.apply(
        lambda x: pricedata.groupby('community').\
        labeled_price.mean()[x])

colors = get_node_colors()
for group in pricedata.groupby('stim_index'):
    stim_i = int(group[0])
    subset = group[1]
    sns.kdeplot(subset.rating, label = label, 
                linewidth=3, color = colors[stim_i])
