import cPickle
from matplotlib import pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from utils import get_node_colors

# load data
data_loc = path.join('Data','ProcessedData')
structuredata = pd.read_csv(path.join(data_loc, 'structuredata.csv'))
valuedata = pd.read_csv(path.join(data_loc, 'valuedata.csv'))
analysis = cPickle.load(open(path.join('Analysis_Results','analysis.pkl'),'rb'))

plotvaluedata = valuedata.query('community!=2')
colors = get_node_colors()

pal = sns.color_palette([colors[1],colors[8]])


f = sns.plt.figure(figsize=(12,8))
for subset in plotvaluedata.groupby('community'):
    label = ['High', 'Low'][subset[0]]
    sns.kdeplot(subset[1].rating, label = label, linewidth=3, color=pal[subset[0]])
leg=sns.plt.legend(fontsize=20, title='Community')
leg.get_title().set_fontsize(20)
sns.plt.xlabel('Value Rating', fontsize = 24)
sns.plt.title('Value Rating as a function of Community',
              fontsize = 30)
f.savefig('Plots/community_values.png')

# plot based on average community value
f=sns.plt.figure(figsize=(12,10))
colors = get_node_colors()
for group in plotvaluedata.groupby('stim_index'):
    stim_i = int(group[0])
    subset = group[1]
    sns.kdeplot(subset.rating, label = stim_i, 
                linewidth=5, color = colors[stim_i])
sns.plt.xlim(0,10)
sns.plt.title('Value Distributions for Individual Stimuli', fontsize=30)
sns.plt.xlabel('Value', fontsize=24)
leg=sns.plt.legend(fontsize=20, title='Stim Identity')
leg.get_title().set_fontsize(20)
f.savefig('Plots/Stim_Values.png')

# plot interaction
struc_coef = plotvaluedata.structure_coefficient
plotvaluedata.loc[:,'binned_struc_coef'] = list(np.digitize(struc_coef,[np.percentile(struc_coef,i) for i in [33,66]]))
sns.factorplot(x='community',y='rating',hue='binned_struc_coef',data=plotvaluedata)

"""
# unused figures
sns.plt.figure(figsize=(12,8))
for subset in plotvaluedata.groupby('stim_repetition'):
    label = subset[0]
    sns.kdeplot(subset[1].rating, label = label, linewidth=3)
sns.plt.legend(fontsize=20, title='Stimulus Repetition')
sns.plt.xlabel('value Rating', fontsize = 18)
sns.plt.title('Value Rating as a function of Stimulus Repetition',
              fontsize = 22)


f=plt.figure(figsize=(12,10))
sns.boxplot('community','rating',
            data=plotvaluedata.query('stim_index in [2,3,8,9]'))
plt.xlabel('Stimulus Repetition', fontsize=20)
plt.ylabel('Community (corrected) Rating', fontsize=20)
f.savefig('Community_Rating_Box.png')
"""