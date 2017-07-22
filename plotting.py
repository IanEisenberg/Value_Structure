from os import path
import pandas as pd
import seaborn as sns

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

c_colors = [[.8,0,.2], [0,.8,.4], [0,.1,1]]
colors = [c_colors[0]]*5 + [c_colors[1]]*5 + [c_colors[2]]*5
colors[0] = [i*.7+y*.3 for i,y in zip(c_colors[0],c_colors[2])]
colors[4] = [i*.7+y*.3 for i,y in zip(c_colors[0],c_colors[1])]
colors[5] = [i*.7+y*.3 for i,y in zip(c_colors[1],c_colors[0])]
colors[9] = [i*.7+y*.3 for i,y in zip(c_colors[1],c_colors[2])]
colors[10] = [i*.7+y*.3 for i,y in zip(c_colors[2],c_colors[1])]
colors[14] = [i*.7+y*.3 for i,y in zip(c_colors[2],c_colors[0])]
sns.palplot(colors)

for group in pricedata.groupby('stim_index'):
    stim_i = int(group[0])
    subset = group[1]
    sns.kdeplot(subset.rating, label = label, 
                linewidth=3, color = colors[stim_i])
